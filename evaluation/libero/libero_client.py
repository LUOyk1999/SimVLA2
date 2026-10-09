#!/usr/bin/env python3
"""SimVLA2 LIBERO evaluation client (websocket).

    python libero_client.py --host 127.0.0.1 --port 8000 \
        --task_suite libero_spatial --output_dir out/

Talks to serve_simvla2_libero.py and writes <output_dir>/<task_suite>/:
results.json (success, total, rate, expected, per-task counts) and, unless
--no_video, one mp4 per episode under videos/.

The policy gets both camera images rotated 180 degrees and the 8-D state
[eef_pos(3), axis_angle(3), gripper_qpos(2)], as in the training data
(datasets/domain_handler/libero_hdf5.py). Its 7-D delta actions go to the
simulator's default delta controller unchanged.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from typing import Deque, Dict


import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

sys.path.append(str(Path(__file__).resolve().parents[2]))
from evaluation.policy_client import PolicyClient

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

MAX_STEPS = {
    "libero_spatial": 800,
    "libero_object": 800,
    "libero_goal": 800,
    "libero_10": 900,
    "libero_90": 900,
}

NUM_STEPS_WAIT = 10  # Wait for objects to stabilize

benchmark_dict = benchmark.get_benchmark_dict()


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to axis-angle representation.
    
    Uses the same convention as robosuite for consistency with training data.
    """
    # Keep w inside the domain of acos.
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _resize_with_pad(image: np.ndarray, size: int) -> np.ndarray:
    """Bilinear resize of a uint8 HWC image to ``size`` x ``size``, aspect kept, zero-padded.

    The reported LIBERO numbers depend on this exact resize; keep it
    bit-identical.
    """
    if image.shape[:2] == (size, size):
        return image
    pil = Image.fromarray(image)
    ratio = max(pil.width / size, pil.height / size)
    resized = pil.resize((int(pil.width / ratio), int(pil.height / ratio)), resample=Image.BILINEAR)
    padded = Image.new(resized.mode, (size, size), 0)
    padded.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return np.asarray(padded)


# -----------------------------------------------------------------------------
# Policy client
# -----------------------------------------------------------------------------

class WebSocketClient:
    """Websocket client for the SimVLA2 policy server.

    Executes the first ``replan_steps`` actions of each returned chunk before
    querying the server again.
    """
    def __init__(self, host: str, port: int, replan_steps: int = 5, resize_size: int = 224):
        self.client = PolicyClient(host, port)
        self.replan_steps = replan_steps
        self.resize_size = resize_size
        self.reset()

    def reset(self) -> None:
        self.action_plan: Deque[np.ndarray] = collections.deque()

    def step(self, obs: Dict, goal: str) -> np.ndarray:
        if not self.action_plan:
            img = _resize_with_pad(obs["image"], self.resize_size)
            wrist_img = _resize_with_pad(obs["wrist_image"], self.resize_size)
            
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": obs["state"],
                "prompt": goal,
            }
            
            result = self.client.infer(element)
            action_chunk = result["actions"]
            
            if not isinstance(action_chunk, np.ndarray):
                action_chunk = np.array(action_chunk)
            
            assert len(action_chunk) >= self.replan_steps, \
                f"Need {self.replan_steps} steps but got {len(action_chunk)}"
            
            for i in range(min(self.replan_steps, len(action_chunk))):
                self.action_plan.append(action_chunk[i])

        return self.action_plan.popleft()


# -----------------------------------------------------------------------------
# Evaluator
# -----------------------------------------------------------------------------
def get_libero_env(task, resolution: int, seed: int):
    """Initialize a LIBERO environment."""
    task_description = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def eval_libero(
    client,
    task_suite_name: str,
    num_trials: int = 50,
    seed: int = 7,
    video_out_path: str = "data/libero/videos",
    save_video: bool = True,
) -> Dict:
    """Evaluate every task in a suite; returns the counts written to results.json."""
    np.random.seed(seed)
    
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = MAX_STEPS.get(task_suite_name, 400)
    
    if save_video:
        Path(video_out_path).mkdir(parents=True, exist_ok=True)
    
    print(f"Task suite: {task_suite_name}")
    print(f"   Tasks: {num_tasks}, Trials per task: {num_trials}")
    print(f"   Max steps: {max_steps}")
    
    total_episodes, total_successes = 0, 0
    per_task = {}
    
    for task_id in tqdm(range(num_tasks - 1, -1, -1), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
        
        task_successes = 0
        for ep in tqdm(range(num_trials), desc=f"{task_description[:30]}...", leave=False):
            env.reset()
            client.reset()
            obs = env.set_init_state(initial_states[ep % len(initial_states)])
            
            replay_images = []
            t = 0
            done = False
            
            # A server or simulator error must end the run, not count as a
            # failed episode.
            while t < max_steps + NUM_STEPS_WAIT:
                if t < NUM_STEPS_WAIT:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # Rotated 180 degrees, like the training images.
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                if save_video:
                    replay_images.append(img)

                state = np.concatenate([
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ])

                obs_dict = {
                    "image": img,
                    "wrist_image": wrist_img,
                    "state": state,
                }

                action = client.step(obs_dict, task_description)
                obs, reward, done, info = env.step(action.tolist())

                if done:
                    task_successes += 1
                    total_successes += 1
                    break

                t += 1

            total_episodes += 1
            
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")[:50]
            video_path = Path(video_out_path) / f"{task_segment}_ep{ep}_{suffix}.mp4"
            if replay_images and save_video:
                imageio.mimwrite(str(video_path), replay_images, fps=10)
            
            status_icon = "[OK]" if done else "[FAIL]"
            print(f"  {status_icon} Task {task_id} Ep {ep}: {suffix.upper()} (steps={t})")

        env.close()
        per_task[str(task_id)] = {"task": task_description, "success": task_successes, "total": num_trials}
        print(f"   Task {task_id}: {task_successes}/{num_trials} ({task_successes/num_trials*100:.1f}%)")
    
    success_rate = total_successes / max(total_episodes, 1)
    print(f"\nTotal success rate: {total_successes}/{total_episodes} ({success_rate*100:.1f}%)")
    
    return {
        "suite": task_suite_name,
        "success": total_successes,
        "total": total_episodes,
        "rate": success_rate,
        "expected": num_tasks * num_trials,
        "tasks": per_task,
    }


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("LIBERO Evaluation Client")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Policy server host")
    parser.add_argument("--port", type=int, default=8000, help="Policy server port")
    parser.add_argument("--task_suite", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Results go to <output_dir>/<task_suite>/")
    parser.add_argument("--no_video", action="store_true", help="Disable video recording for faster evaluation")

    args = parser.parse_args()

    print(f"Starting LIBERO evaluation client")
    print(f"   Server: ws://{args.host}:{args.port}")
    print(f"   Task suite: {args.task_suite}")
    print(f"   Replan steps: {args.replan_steps}")
    print()
    
    client = WebSocketClient(args.host, args.port, replan_steps=args.replan_steps)
    
    suite_dir = Path(args.output_dir) / args.task_suite
    results = eval_libero(
        client=client,
        task_suite_name=args.task_suite,
        num_trials=args.num_trials,
        seed=args.seed,
        video_out_path=str(suite_dir / "videos"),
        save_video=not args.no_video,
    )
    suite_dir.mkdir(parents=True, exist_ok=True)
    (suite_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Results written to {suite_dir / 'results.json'}")


if __name__ == "__main__":
    main()
