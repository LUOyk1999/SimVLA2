#!/usr/bin/env python3
"""SimVLA2 Fractal evaluation client (SimplerEnv Google robot, variant aggregation).

    python fractal_client.py \
        --host 127.0.0.1 --port 8000 --task coke_can --output_dir ./results

Talks to serve_simvla2_fractal.py over a websocket; run_eval.sh puts the
SimplerEnv checkout on PYTHONPATH. Results go to <output_dir>/VA/<task>/:
google_results.txt (one JSON line per episode) and one mp4 per episode,
which --resume uses to skip finished episodes.
"""
import argparse
import os
import time
import json
import numpy as np
import collections
from sapien.core import Pose
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_video
import simpler_env
import sys
from itertools import product
from transforms3d.euler import euler2quat
import itertools
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from evaluation.policy_client import PolicyClient

SCRIPT_DIR = Path(__file__).parent  # evaluation/fractal/

TASK_CONFIGS = {
    "coke_can":   SCRIPT_DIR / "configs/VA/coke_can.json",
    "move_near":  SCRIPT_DIR / "configs/VA/move_near.json",
    "open_close": SCRIPT_DIR / "configs/VA/open_close.json",
}

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def parse_range_tuple(t):
    if isinstance(t, (int, float)):
        return [t]
    return np.linspace(t[0], t[1], int(t[2])).tolist()

def generate_robot_init_quats(quat_center, rpy_range):
    r_range = parse_range_tuple(rpy_range[:3])
    p_range = parse_range_tuple(rpy_range[3:6])
    y_range = parse_range_tuple(rpy_range[6:])
    return [
        (Pose(q=euler2quat(r, p, y)) * Pose(q=quat_center)).q
        for r, p, y in product(r_range, p_range, y_range)
    ]

def get_expected_episodes(scenario_config: dict) -> int:
    """Number of episodes one scenario expands to."""
    # robot init combinations
    robot_init_x_count = len(parse_range_tuple(scenario_config["robot_init_x"]))
    robot_init_y_count = len(parse_range_tuple(scenario_config["robot_init_y"]))
    robot_init_quats = generate_robot_init_quats(
        scenario_config["robot_init_rot_quat_center"], 
        scenario_config["robot_init_rot_rpy_range"]
    )
    robot_init_count = robot_init_x_count * robot_init_y_count * len(robot_init_quats)
    
    # obj reset combinations
    if scenario_config["obj_variation_mode"] == "episode":
        obj_reset_count = scenario_config.get("episode_nums", 1)
    elif scenario_config["obj_variation_mode"] == "xy":
        x_count = len(parse_range_tuple(scenario_config["obj_init_x_range"]))
        y_count = len(parse_range_tuple(scenario_config["obj_init_y_range"]))
        obj_reset_count = x_count * y_count
    else:
        obj_reset_count = 1
    
    return robot_init_count * obj_reset_count

# -----------------------------------------------------------------------------
# Policy client
# -----------------------------------------------------------------------------
class FractalClient:
    """Turns the server's predicted chunks into absolute end-effector commands.

    The server returns ``[T, 7]``: the xyz delta from the position at query
    time, absolute Euler xyz and a continuous gripper value. The first
    ``exec_chunk`` rows are anchored at ``current_xyz`` and executed before the
    next query; the gripper is thresholded to +-1. Proprio is all zeros, as in
    training.
    """

    def __init__(self, host: str, port: int, gripper_threshold: float = 0.5, exec_chunk: int = 20):
        self.client = PolicyClient(host, port)
        self.gripper_threshold = gripper_threshold
        self.exec_chunk = exec_chunk
        self.proprio = np.zeros(7, dtype=np.float32)
        self.reset()

    def reset(self, instruction=None, current_xyz=None):
        self.instruction = instruction
        self.action_plan = collections.deque()
        self.current_xyz = current_xyz

    def set_instruction(self, instruction: str):
        self.instruction = instruction

    def step(self, image: np.ndarray) -> np.ndarray:
        if not self.action_plan:
            result = self.client.infer({
                "proprio": self.proprio,
                "language_instruction": self.instruction,
                "image": image,
            })
            action_seq = np.array(result["actions"], dtype=np.float32)[:self.exec_chunk]
            action_seq[:, :3] += self.current_xyz
            self.action_plan.extend(action_seq.tolist())

        action_pred = np.array(self.action_plan.popleft(), dtype=np.float32)
        gripper_action = 1.0 if action_pred[6] > self.gripper_threshold else -1.0
        action_final = np.concatenate([action_pred[:6], np.array([gripper_action])])

        # The next chunk is anchored at this commanded position, not at the
        # measured one.
        self.current_xyz = action_final[:3].reshape(1, 3)
        return action_final

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def evaluate_scenario(client, output_dir: str, scenario_config: dict, scenario_name: str,
                      completed_episodes: set = None, episode_ids_filter: set = None):
    """Evaluate a single scenario.
    
    Args:
        completed_episodes: Episode IDs to skip (resume mode).
        episode_ids_filter: If set, only these episode IDs run (e.g. to score
            move_near one instruction at a time).
    """
    if completed_episodes is None:
        completed_episodes = set()
    
    max_steps = scenario_config["max_episode_steps"] * 2
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "google_results.txt")
    
    print("\n" + "=" * 70)
    print(f"🧩 [Eval] Scenario: {scenario_name} | Max Steps: {max_steps}")
    if completed_episodes:
        print(f"   📂 Resume: Skipping {len(completed_episodes)} completed episodes")
    print("=" * 70)
    
    scenario_config["control_freq"] = 3
    scenario_config["sim_freq"] = 513
    robot_init_quats = generate_robot_init_quats(
        scenario_config["robot_init_rot_quat_center"], 
        scenario_config["robot_init_rot_rpy_range"]
    )
    if "rgb_overlay_path" in scenario_config:
        if "rgb_overlay_cameras" not in scenario_config:
            if "google_robot_static" in scenario_config["robot_name"]:
                scenario_config["rgb_overlay_cameras"] = ["overhead_camera"]

    ep_count = 0
    success_count = 0
    robot_init_x_range = scenario_config["robot_init_x"]
    robot_init_y_range = scenario_config["robot_init_y"]
    
    for robot_init_x in parse_range_tuple(robot_init_x_range):
        for robot_init_y in parse_range_tuple(robot_init_y_range):
            for robot_init_quat in robot_init_quats:
                scenario_config["robot_init_rot_quat"] = robot_init_quat
                
                make_kwargs = dict(
                    robot=scenario_config["robot_name"],
                    sim_freq=scenario_config["sim_freq"],
                    control_freq=scenario_config["control_freq"],
                    control_mode="arm_pd_ee_base_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner",
                    scene_name=scenario_config["scene_name"],
                    camera_cfgs={"add_segmentation": True},
                    rgb_overlay_path=scenario_config.get("rgb_overlay_path", None),
                    rgb_overlay_cameras=scenario_config.get("rgb_overlay_cameras", None),
                )

                env = simpler_env.make(scenario_config["env_name"], **make_kwargs, **scenario_config["additional_env_build_kwargs"])
                options = {
                    "robot_init_options": {
                        "init_xy": np.array([robot_init_x, robot_init_y]),
                        "init_rot_quat": robot_init_quat,
                    }
                }
                
                reset_combinations = []
                if scenario_config["obj_variation_mode"] == "episode":
                    for ep_id in range(scenario_config["episode_nums"]):
                        if episode_ids_filter is not None and ep_id not in episode_ids_filter:
                            continue
                        reset_combinations.append({"episode_id": ep_id})
                elif scenario_config["obj_variation_mode"] == "xy":
                    x_list = parse_range_tuple(scenario_config["obj_init_x_range"])
                    y_list = parse_range_tuple(scenario_config["obj_init_y_range"])
                    xy_combinations = list(itertools.product(x_list, y_list))
                    for x, y in xy_combinations:
                        reset_combinations.append({"init_xy": np.array([x, y])})
                        
                for obj_reset_option in reset_combinations:
                    if ep_count in completed_episodes:
                        print(f"   ⏭️  Skipping completed: {scenario_name} episode {ep_count}")
                        ep_count += 1
                        continue
                    
                    options["obj_init_options"] = obj_reset_option
                    obs, _ = env.reset(options=options)
                    print(f"Eval scenario: {scenario_name} for {ep_count}-th episode......")
                    images = []
                    instruction = env.unwrapped.get_language_instruction()
                    print(f"📝 Now Instruction: {instruction}")
                    
                    ee_pose_wrt_base = Pose(p=obs['agent']['base_pose'][:3], q=obs['agent']['base_pose'][3:]).inv() * Pose(p=obs['extra']['tcp_pose'][:3], q=obs['extra']['tcp_pose'][3:])
                    current_xyz = np.array(ee_pose_wrt_base.p).reshape(1, 3)
                    client.reset(instruction, current_xyz)
                    
                    task_start = time.time()
                    for step_idx in range(max_steps):
                        instruction = env.unwrapped.get_language_instruction()
                        if instruction != client.instruction:
                            client.set_instruction(instruction)
                        image = get_image_from_maniskill2_obs_dict(env.unwrapped, obs)
                        action = client.step(image)
                        obs, reward, done, _, _ = env.step(action)
                        images.append(image.copy())

                        if done:
                            print(f"✅ Scenario {scenario_name} completed in {step_idx+1} steps (suc={done})")
                            break
                        
                    duration = time.time() - task_start
                    out_video = os.path.join(output_dir, f"{scenario_name}_{ep_count}_{done:.2f}.mp4")
                    write_video(out_video, images, fps=10)

                    result = {
                        "scenario": scenario_name,
                        "episode_id": ep_count,
                        "reward": float(reward),
                        "done": bool(done),
                        "steps": step_idx + 1,
                        "duration_sec": duration,
                        "output": out_video,
                    }
                    if done:
                        success_count += 1

                    with open(log_path, "a+") as f:
                        f.write(json.dumps(result) + "\n")

                    print(f"🎥 Saved video to {out_video}")
                    print(f"🕒 Current episode duration: {duration:.1f}s")
                    ep_count += 1
                    
                env.close()
    
    return {"total_episodes": ep_count, "success_count": success_count}

def parse_video_filename(filename):
    """Parse ``{scenario}_{episode_id}_{success}.mp4``; None if it does not match.

    Example: ``google_robot_move_near_va_8_52_0.00.mp4`` gives scenario
    ``google_robot_move_near_va_8``, episode_id 52, success 0.00.
    """
    if not filename.endswith('.mp4'):
        return None
    name = filename[:-4]
    parts = name.rsplit('_', 2)
    if len(parts) < 3:
        return None
    try:
        scenario = parts[0]
        episode_id = int(parts[1])
        success = float(parts[2])
        return {"scenario": scenario, "episode_id": episode_id, "success": success}
    except (ValueError, IndexError):
        return None

def get_completed_from_videos(output_dir):
    """``{scenario: {episode_id: success}}`` read from the video filenames."""
    completed = {}  # {scenario_name: {episode_id: success}}
    if not os.path.exists(output_dir):
        return completed
    
    for filename in os.listdir(output_dir):
        parsed = parse_video_filename(filename)
        if parsed:
            scenario = parsed["scenario"]
            ep_id = parsed["episode_id"]
            success = parsed["success"]
            if scenario not in completed:
                completed[scenario] = {}
            completed[scenario][ep_id] = success
    return completed

def get_stats_from_videos(output_dir):
    """Episode and success counts, read from the video filenames."""
    total_episodes = 0
    success_count = 0
    
    if not os.path.exists(output_dir):
        return total_episodes, success_count
    
    for filename in os.listdir(output_dir):
        parsed = parse_video_filename(filename)
        if parsed:
            total_episodes += 1
            if parsed["success"] >= 0.99:  # 1.00 means success
                success_count += 1
    
    return total_episodes, success_count

def main():
    parser = argparse.ArgumentParser("SimVLA2 Fractal Evaluation Client")
    parser.add_argument("--task", type=str, required=True,
                        choices=["coke_can", "move_near", "open_close"],
                        help="Task to evaluate")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Policy server host")
    parser.add_argument("--port", type=int, default=8000, help="Policy server port")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--scenario_start", type=int, default=0,
                        help="Start scenario index (for parallel execution)")
    parser.add_argument("--scenario_end", type=int, default=-1,
                        help="End scenario index, -1 means all")
    parser.add_argument("--episode_ids", type=str, default="",
                        help="Comma-separated episode IDs to evaluate within each scenario (for move_near per-instruction eval)")
    parser.add_argument("--gripper_threshold", type=float, default=0.5,
                        help="Threshold for gripper action")
    parser.add_argument("--exec_chunk", type=int, default=20,
                        help="Actions executed from each predicted chunk before re-querying")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous run, skip completed scenarios")
    args = parser.parse_args()

    config_path = TASK_CONFIGS[args.task]
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, "r") as f:
        env_dict = json.load(f)
    
    all_scenario_names = list(env_dict.keys())
    total_scenarios = len(all_scenario_names)
    
    start_idx = args.scenario_start
    end_idx = args.scenario_end if args.scenario_end > 0 else total_scenarios
    scenario_names = all_scenario_names[start_idx:end_idx]
    
    task_output_dir = os.path.join(args.output_dir, "VA", args.task)
    os.makedirs(task_output_dir, exist_ok=True)
    
    completed_videos = {}
    if args.resume:
        completed_videos = get_completed_from_videos(task_output_dir)
        if completed_videos:
            total_completed = sum(len(eps) for eps in completed_videos.values())
            print(f"📂 Resume mode: Completed {total_completed} episodes ({len(completed_videos)} scenarios)")
    
    # A scenario is done once its videos cover every expected episode.
    scenarios_to_run = []
    for name in scenario_names:
        expected_eps = get_expected_episodes(env_dict[name])
        if name in completed_videos:
            completed_eps = len(completed_videos[name])
            if completed_eps >= expected_eps:
                print(f"   ⏭️  Skipping completed: {name} ({completed_eps}/{expected_eps} episodes)")
            else:
                print(f"   🔄 Needs continuation: {name} ({completed_eps}/{expected_eps} episodes)")
                scenarios_to_run.append(name)
        else:
            scenarios_to_run.append(name)
    
    print(f"🚀 Starting evaluation: VA/{args.task}")
    print(f"📊 Scenarios: {len(scenarios_to_run)} to evaluate (Skipping {len(scenario_names) - len(scenarios_to_run)} completed)")
    print(f"📁 Output: {task_output_dir}")
    
    if not scenarios_to_run:
        print("✅ All scenarios completed, no evaluation needed")
        sys.exit(0)

    client = FractalClient(args.host, args.port, gripper_threshold=args.gripper_threshold,
                           exec_chunk=args.exec_chunk)

    episode_ids_filter = None
    if args.episode_ids:
        episode_ids_filter = set(int(i) for i in args.episode_ids.split(",") if i.strip())
        print(f"🎯 Episode IDs filter: {sorted(episode_ids_filter)}")

    results = []
    for scenario_name in scenarios_to_run:
        completed_eps = set(completed_videos.get(scenario_name, {}).keys()) if args.resume else set()
        result = evaluate_scenario(client, task_output_dir, env_dict[scenario_name], scenario_name, completed_eps,
                                   episode_ids_filter=episode_ids_filter)
        results.append(result)
    
    # Counted from the videos, so episodes from earlier resumed runs are included.
    total_eps, total_success = get_stats_from_videos(task_output_dir)
    overall_rate = total_success / total_eps if total_eps > 0 else 0
    
    current_eps = sum(r["total_episodes"] for r in results)
    current_success = sum(r["success_count"] for r in results)
    
    print(f"\n{'='*70}")
    print(f"🏁 Evaluation Complete: VA/{args.task}")
    print(f"   Current run: {current_eps} episodes, {current_success} successes")
    print(f"   Cumulative stats (from videos): {total_eps} episodes, {total_success} successes")
    print(f"   Success Rate: {overall_rate*100:.1f}%")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

