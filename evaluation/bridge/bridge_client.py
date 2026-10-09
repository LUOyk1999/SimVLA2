#!/usr/bin/env python3
"""SimVLA2 Bridge evaluation client (WidowX in SimplerEnv, websocket).

    python bridge_client.py --host 127.0.0.1 --port 8000 \
        --task widowx_carrot_on_plate --output_dir out/widowx_carrot_on_plate

Talks to serve_simvla2_bridge.py. Per task it appends one JSON line per
episode to <task>_results.txt and writes one mp4 per episode, which --resume
uses to skip finished episodes.

Observation format:
1. State: [x(1), y(1), z(1), rx(1), ry(1), rz(1), gripper(1)] = 7D
2. Action: absolute action (7D)
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")

import time
import json
import numpy as np
import collections
import gymnasium as gym
from scipy.spatial.transform import Rotation as R
from sapien.core import Pose
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_video
import simpler_env

sys.path.append(str(Path(__file__).resolve().parents[2]))
from evaluation.policy_client import PolicyClient

# -----------------------------------------------------------------------------
# Tasks
# -----------------------------------------------------------------------------
TASKS = {
    "widowx_stack_cube": {
        "name": "widowx_stack_cube",
        "description": "Stack green cube on yellow cube",
    },
    "widowx_put_eggplant_in_basket": {
        "name": "widowx_put_eggplant_in_basket",
        "description": "Put eggplant in basket",
    },
    "widowx_carrot_on_plate": {
        "name": "widowx_carrot_on_plate",
        "description": "Put carrot on plate",
    },
    "widowx_spoon_on_towel": {
        "name": "widowx_spoon_on_towel",
        "description": "Put spoon on towel",
    },
}

# General Configurations
GRIPPER_THRESHOLD = 0.91
AXIS_TRANSFORM = np.array([0, np.pi / 2, 0])
GRIPPER_QPOS_MIN = 0.015
GRIPPER_QPOS_MAX = 0.037
MAX_STEPS = 1200
POS_LIMIT = 1.0
EULER_LIMIT = np.pi / 2
NUM_EPISODES = 24


# -----------------------------------------------------------------------------
# Policy client
# -----------------------------------------------------------------------------
class BridgeClient:
    def __init__(self, host: str, port: int):
        self.client = PolicyClient(host, port)
        self.reset()

    def reset(self, proprio=None, instruction=None):
        self.proprio = proprio
        self.instruction = instruction
        self.action_plan = collections.deque()

    def step(self, image: np.ndarray) -> np.ndarray:
        if not self.action_plan:
            result = self.client.infer({
                "image": image,
                "proprio": self.proprio,
                "language_instruction": self.instruction,
            })
            action_seq = np.array(result["actions"], dtype=np.float32)
            self.action_plan.extend(action_seq.tolist())

        action_pred = np.array(self.action_plan.popleft(), dtype=np.float32)
        # The next query's proprio is the last commanded pose, not the measured one.
        self.proprio[:7] = action_pred[:7]

        pos = np.clip(action_pred[:3], -POS_LIMIT, POS_LIMIT)
        euler_sim = np.clip(action_pred[3:6] + AXIS_TRANSFORM, -EULER_LIMIT, EULER_LIMIT)
        gripper_env = -1.0 if action_pred[6] < GRIPPER_THRESHOLD else 1.0

        return np.concatenate([pos, euler_sim, [gripper_env]])


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def get_completed_episodes(output_dir: str, task_name: str) -> set:
    """Scan output dir for completed episode IDs."""
    completed = set()
    if not os.path.exists(output_dir):
        return completed
    
    import re
    # Video filename format: {task_name}_{proc_id}_{success:.2f}.mp4
    pattern = re.compile(rf"^{re.escape(task_name)}_(\d+)_[\d.]+\.mp4$")
    
    for fname in os.listdir(output_dir):
        match = pattern.match(fname)
        if match:
            completed.add(int(match.group(1)))
    
    return completed


def load_existing_results(output_dir: str, task_name: str) -> tuple:
    """Read stats from existing results.txt."""
    log_path = os.path.join(output_dir, f"{task_name}_results.txt")
    success_count = 0
    total_count = 0
    
    if not os.path.exists(log_path):
        return success_count, total_count
    
    try:
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                result = json.loads(line)
                total_count += 1
                if result.get("success", False):
                    success_count += 1
    except Exception as e:
        print(f"⚠️ Error reading {log_path}: {e}")
    
    return success_count, total_count


def evaluate_task(client, task_name: str, output_dir: str, episode_start: int, episode_end: int, max_steps: int = MAX_STEPS, resume: bool = False, cuda_device: int = 0):
    """Evaluate a single task."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"{task_name}_results.txt")
    
    success_count = 0
    total_count = 0
    
    completed_episodes = set()
    if resume:
        completed_episodes = get_completed_episodes(output_dir, task_name)
        if completed_episodes:
            success_count, total_count = load_existing_results(output_dir, task_name)
            print(f"📦 Resume mode: found {len(completed_episodes)} completed episodes (success={success_count}, total={total_count})")

    print(f"\n{'='*70}")
    print(f"🧩 Task: {task_name}")
    print(f"📊 Episodes: {episode_start} - {episode_end}")
    if resume and completed_episodes:
        print(f"⏭️  Skipping: {sorted(completed_episodes)}")
    print(f"{'='*70}")

    # One env for all episodes; a new env per episode exhausts Vulkan handles.
    env_name, env_kwargs = simpler_env.ENVIRONMENT_MAP[task_name]
    env_kwargs = dict(env_kwargs)
    env_kwargs["prepackaged_config"] = True
    env_kwargs["renderer_kwargs"] = {
        "offscreen_only": True,
        "device": f"cuda:{cuda_device}",
    }
    env = gym.make(env_name, obs_mode="rgbd", **env_kwargs)

    try:
        for proc_id in range(episode_start, episode_end):
            if resume and proc_id in completed_episodes:
                print(f"⏭️ Episode {proc_id}: already completed, skipping")
                continue
            try:
                images = []
                obs, _ = env.reset(options={"obj_init_options": {"episode_id": proc_id}})
                instruction = env.unwrapped.get_language_instruction()

                ee_pose_wrt_base = Pose(
                    p=obs["agent"]["base_pose"][:3],
                    q=obs["agent"]["base_pose"][3:]
                ).inv() * Pose(
                    p=obs["extra"]["tcp_pose"][:3],
                    q=obs["extra"]["tcp_pose"][3:]
                )

                # sapien quaternions are wxyz; scipy reads xyzw.
                ee_euler_model = R.from_quat(ee_pose_wrt_base.q[[1, 2, 3, 0]]).as_euler('xyz') - AXIS_TRANSFORM
                gripper_qpos = obs["agent"]["qpos"][-1]
                gripper_normalized = np.clip(
                    (gripper_qpos - GRIPPER_QPOS_MIN) / (GRIPPER_QPOS_MAX - GRIPPER_QPOS_MIN), 0.0, 1.0
                )

                proprio = np.concatenate([
                    ee_pose_wrt_base.p, ee_euler_model, [gripper_normalized]
                ]).astype(np.float32)

                client.reset(proprio, instruction)

                task_start = time.time()
                done = False
                for step_idx in range(max_steps):
                    image = get_image_from_maniskill2_obs_dict(env.unwrapped, obs)
                    action = client.step(image)
                    obs, reward, done, _, _ = env.step(action)
                    images.append(image.copy())
                    if done:
                        break

                duration = time.time() - task_start
                total_count += 1
                if done:
                    success_count += 1

                out_video = os.path.join(output_dir, f"{task_name}_{proc_id}_{done:.2f}.mp4")
                write_video(out_video, images, fps=10)

                result = {
                    "task": task_name,
                    "episode": proc_id,
                    "success": bool(done),
                    "steps": step_idx + 1,
                    "duration": duration,
                }

                with open(log_path, "a+") as f:
                    f.write(json.dumps(result) + "\n")

                status = "✅" if done else "❌"
                print(f"{status} Episode {proc_id}: steps={step_idx+1}, time={duration:.1f}s")

            except Exception as e:
                print(f"⚠️ Episode {proc_id} error: {e}")
                continue
    finally:
        env.close()

    success_rate = success_count / total_count if total_count > 0 else 0
    print(f"\n📊 {task_name} Summary: {success_count}/{total_count} ({success_rate*100:.1f}%)")
    
    return {"task": task_name, "success": success_count, "total": total_count, "rate": success_rate}


def main():
    parser = argparse.ArgumentParser("SimVLA2 Bridge Multi-Task Evaluation")
    parser.add_argument("--task", type=str, default="all",
                        choices=list(TASKS.keys()) + ["all"],
                        help="Task to evaluate (or 'all' for all tasks)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Policy server host")
    parser.add_argument("--port", type=int, default=8000, help="Policy server port")
    parser.add_argument("--output_dir", type=str, default="logs-all")
    parser.add_argument("--episode_start", type=int, default=0)
    parser.add_argument("--episode_end", type=int, default=None)
    parser.add_argument("--num_episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--resume", action="store_true", help="Resume from completed episodes (skip if video exists)")
    parser.add_argument("--cuda_device", type=int, default=0,
                        help="CUDA device index for SAPIEN renderer (default: 0, should match the GPU running the server)")
    args = parser.parse_args()

    episode_end = args.episode_end if args.episode_end else args.num_episodes
    
    print("🚀 SimVLA2 Bridge Evaluation")
    print(f"🔗 Server: {args.host}:{args.port}")
    print(f"📁 Output: {args.output_dir}")

    os.environ["VK_ICD_FILENAMES"] = os.environ.get("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")

    client = BridgeClient(args.host, args.port)

    tasks_to_eval = list(TASKS.keys()) if args.task == "all" else [args.task]
    
    all_results = []
    for task_name in tasks_to_eval:
        if args.task == "all":
            task_output = os.path.join(args.output_dir, task_name)
        else:
            task_output = args.output_dir
            
        result = evaluate_task(client, task_name, task_output, args.episode_start, episode_end,
                               resume=args.resume, cuda_device=args.cuda_device)
        all_results.append(result)

    # Print only; run_eval.sh writes summary.json from the per-task results.
    print("\n" + "="*70)
    print("📊 Overall Summary")
    print("="*70)
    total_success = sum(r["success"] for r in all_results)
    total_episodes = sum(r["total"] for r in all_results)
    for r in all_results:
        print(f"  {r['task']}: {r['success']}/{r['total']} ({r['rate']*100:.1f}%)")
    if total_episodes > 0:
        print(f"\n  Total: {total_success}/{total_episodes} ({total_success/total_episodes*100:.1f}%)")
    else:
        print(f"\n  Total: 0/0 (no episodes completed)")


if __name__ == "__main__":
    main()

