#!/usr/bin/env python3
"""SimVLA2 CALVIN evaluation client (ABC->D, websocket).

    python calvin_client.py --host 127.0.0.1 --port 8000 \
        --dataset_path calvin/dataset/task_ABC_D/validation --output_dir out/

Talks to a running :mod:`serve_simvla2_calvin` policy server over a
websocket, drives CALVIN's multi-step protocol (1000 chains by default), and
writes per-worker logs and ``summary.json`` (score = mean completed chain
length, sr = success rate at chain lengths 1-5) into ``--output_dir``.
"""
import argparse
import json
import os
import queue
import sys
import traceback
import importlib.util
import multiprocessing as mp
from pathlib import Path

import imageio
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm
from omegaconf import OmegaConf
import hydra
from pytorch_lightning import seed_everything as reset_everything
import PIL.Image as Image

from calvin_agent.models.calvin_base_model import CalvinBaseModel
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
)
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_env.envs.play_table_env import get_env

sys.path.append(str(Path(__file__).resolve().parents[2]))
from evaluation.policy_client import PolicyClient

# -----------------------------------------------------------------------------
# Global config
# -----------------------------------------------------------------------------
EP_LEN = 720
NUM_SEQUENCES = 1000
SEQGEN_WORKERS = int(os.environ.get("CALVIN_SEQGEN_WORKERS", "8"))


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def euler_xyz_to_rotate6D(q: np.ndarray) -> np.ndarray:
    return R.from_euler("xyz", q, degrees=False).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))


def rotate6D_to_quat(v6: np.ndarray) -> np.ndarray:
    a1, a2 = v6[..., 0:5:2], v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = (a2 - proj) / np.linalg.norm(a2 - proj, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return R.from_matrix(np.stack((b1, b2, b3), axis=-1)).as_quat()


def _pad_frames_to_macro_block(frames, macro_block_size: int = 16):
    if not frames:
        return frames
    padded = []
    for frame in frames:
        h, w = frame.shape[:2]
        pad_h = (macro_block_size - h % macro_block_size) % macro_block_size
        pad_w = (macro_block_size - w % macro_block_size) % macro_block_size
        if pad_h == 0 and pad_w == 0:
            padded.append(frame)
            continue
        if frame.ndim == 2:
            pad_width = ((0, pad_h), (0, pad_w))
        else:
            pad_width = ((0, pad_h), (0, pad_w), (0, 0))
        padded.append(np.pad(frame, pad_width, mode="edge"))
    return padded


def save_video(path, frames, fps=30):
    frames = _pad_frames_to_macro_block(frames, macro_block_size=16)
    imageio.mimsave(path, frames, fps=fps)


def _default_dataset_path() -> Path:
    for cand in (
        Path("calvin/dataset/task_ABC_D/validation"),
        Path("task_ABC_D/validation"),
        Path("ABC_D/validation"),
    ):
        if cand.exists():
            return cand
    # Nothing on disk: fall back to a repo-relative path so the caller gets a
    # clear "not found" instead of a host-specific absolute path.
    return Path("task_ABC_D/validation")


def resolve_calvin_dataset_path(dataset_path) -> Path:
    if not dataset_path:
        return _default_dataset_path()
    resolved = Path(dataset_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset path not found: {resolved}")
    return resolved


class ClientModel(CalvinBaseModel):
    def __init__(
        self,
        host,
        port,
        *,
        plan_len: int = 15,
        gripper_threshold: float = 0.0,
    ):
        self.client = PolicyClient(host, port)
        self.action_plan = []
        self.proprio = None
        self.plan_len = plan_len
        self.gripper_threshold = gripper_threshold

    def reset(self, obs):
        # Proprio: position(3) + rotation 6D(6) + gripper(1), zero-padded to 20.
        self.action_plan.clear()
        self.proprio = np.concatenate([
            obs["robot_obs"][:3],
            euler_xyz_to_rotate6D(obs["robot_obs"][3:6]),
            obs["robot_obs"][-1:],
        ])
        self.proprio = np.concatenate([self.proprio, np.zeros_like(self.proprio)])

    def step(self, obs, goal):
        if not self.action_plan:
            result = self.client.infer({
                "language_instruction": goal,
                "proprio": np.asarray(self.proprio, dtype=np.float64),
                "image0": obs["rgb_obs"]["rgb_static"],
                "image1": obs["rgb_obs"]["rgb_gripper"],
            })
            self.action_plan = result["actions"][: self.plan_len]
        action_predict = np.array(self.action_plan.pop(0))
        if action_predict.shape[-1] < 10:
            raise ValueError(f"Expected >=10D action, got shape {action_predict.shape}")
        self.proprio[:10] = action_predict[:10]
        return (
            action_predict[:3],
            rotate6D_to_quat(action_predict[3:9]),
            1 if float(action_predict[9]) > self.gripper_threshold else -1,
        )


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def evaluate_policy(
    model,
    env,
    output_dir,
    *,
    ep_len: int,
    num_sequences: int,
    save_video_flag: bool,
    eval_start=0,
    eval_end=NUM_SEQUENCES,
):
    conf_dir = Path(__file__).resolve().parent / "ABC_D/validation"
    task_cfg = OmegaConf.load(conf_dir / "new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "new_playtable_validation.yaml")
    # get_sequences defaults its pool to multiprocessing.cpu_count(), which
    # forks far more workers than the ~24 initial states can use. Cap it via
    # SEQGEN_WORKERS.
    eval_sequences = list(get_sequences(num_sequences, num_workers=SEQGEN_WORKERS))

    results = []
    total = max(0, eval_end - eval_start)

    for idx in tqdm(range(eval_start, eval_end), total=total, desc="Evaluating", ncols=100):
        init_state, seq = eval_sequences[idx]
        results.append(evaluate_sequence(
            env,
            model,
            task_oracle,
            init_state,
            seq,
            val_annotations,
            output_dir,
            ep_len=ep_len,
            save_video_flag=save_video_flag,
        ))
        with open(f"{output_dir}/log.txt", 'a+') as f:
            list_r = count_success(results)
            list_r.append(sum(list_r))
            print(f"{idx}: " ," ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(list_r)]) + "|", file=f)

    return results


def evaluate_sequence(
    env,
    model,
    oracle,
    init_state,
    seq,
    annotations,
    output_dir,
    *,
    ep_len: int,
    save_video_flag: bool,
):
    robot_obs, scene_obs = get_env_state_for_initial_condition(init_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success = 0
    for subtask in seq:
        model.reset(env.get_obs())
        ok, frames, lang = rollout(
            env,
            model,
            oracle,
            subtask,
            annotations,
            ep_len=ep_len,
            save_video_flag=save_video_flag,
        )
        if save_video_flag:
            save_video(f"{output_dir}/{lang}_{ok}.mp4", frames)
        if ok:
            success += 1
        else:
            break
    return success


def rollout(
    env,
    model,
    oracle,
    subtask,
    annotations,
    *,
    ep_len: int,
    save_video_flag: bool,
):
    obs = env.get_obs()
    lang = annotations[subtask][0].split("\n")[0].replace("’", "'")
    start_info = env.get_info()
    frames = []
    for step in range(ep_len):
        action = model.step(obs, lang)
        obs, _, _, info = env.step(action)
        if save_video_flag:
            main = obs["rgb_obs"]["rgb_static"]
            wrist = np.asarray(Image.fromarray(obs["rgb_obs"]["rgb_gripper"]).resize(main.shape[:2]))
            frames.append(np.concatenate([main, wrist], axis=1))
        if oracle.get_task_info_for_set(start_info, info, {subtask}):
            return True, frames, lang
    return False, frames, lang


def _split_eval_ranges(start: int, end: int, num_workers: int):
    total = max(0, end - start)
    if total == 0:
        return []
    num_workers = max(1, int(num_workers))
    base = total // num_workers
    extra = total % num_workers
    ranges = []
    cur = start
    for i in range(num_workers):
        size = base + (1 if i < extra else 0)
        if size <= 0:
            break
        ranges.append((cur, cur + size))
        cur += size
    return ranges


def _run_worker(
    worker_id: int,
    host: str,
    port: int,
    args_dict: dict,
    dataset_path: str,
    disable_tactile: bool,
    eval_start: int,
    eval_end: int,
    result_queue,
):
    try:
        reset_everything(0, workers=True)
        env = get_env(dataset_path, show_gui=False, disable_tactile=disable_tactile)
        model = ClientModel(
            host,
            port,
            plan_len=args_dict["plan_len"],
            gripper_threshold=args_dict["gripper_threshold"],
        )
        out_dir = Path(args_dict["output_dir"]) / f"worker_{worker_id}_s{eval_start}_e{eval_end}"
        out_dir.mkdir(parents=True, exist_ok=True)
        results = evaluate_policy(
            model,
            env,
            str(out_dir),
            ep_len=args_dict["ep_len"],
            num_sequences=args_dict["num_sequences_total"],
            save_video_flag=args_dict["save_video"],
            eval_start=eval_start,
            eval_end=eval_end,
        )
    except Exception:
        result_queue.put({"worker_id": worker_id, "error": traceback.format_exc()})
        return
    result_queue.put(
        {
            "worker_id": worker_id,
            "eval_start": eval_start,
            "eval_end": eval_end,
            "results": results,
            "output_dir": str(out_dir),
        }
    )


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("SimVLA2 CALVIN Evaluation Client")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Policy server host")
    parser.add_argument("--port", type=int, default=8000, help="Policy server port")
    parser.add_argument("--output_dir", type=str, default="logs/",
                        help="Directory for saving evaluation videos and logs")
    parser.add_argument("--plan_len", type=int, default=30,
                        help="Number of actions to consume per server call")
    parser.add_argument("--ep_len", type=int, default=EP_LEN,
                        help="Max steps per subtask during evaluation")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Parallel evaluation workers (each runs its own env)")
    parser.add_argument("--num_sequences", type=int, default=NUM_SEQUENCES,
                        help="Number of CALVIN sequences to evaluate")
    parser.add_argument("--eval_start", type=int, default=0,
                        help="Start index for evaluation sequences")
    parser.add_argument("--eval_end", type=int, default=None,
                        help="End index for evaluation sequences (exclusive)")
    parser.add_argument(
        "--gripper_threshold",
        type=float,
        default=0.0,
        help="Threshold for gripper (open if g > thr)",
    )
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to extracted validation folder")
    parser.add_argument("--no_video", action="store_true", default=False,
                        help="Disable mp4 rendering to speed up evaluation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("🚀 [Client] Starting SimVLA2-CALVIN evaluation client...")

    host, port = args.host, args.port
    print(f"🔗 SimVLA2 server at {host}:{port}")

    print("🧩 Loading CALVIN environment...")
    dataset_path = resolve_calvin_dataset_path(args.dataset_path)
    print(f"🧩 Using CALVIN dataset at: {dataset_path}")
    disable_tactile = importlib.util.find_spec("pyrender") is None
    if disable_tactile:
        print("WARNING: pyrender not found. Disabling tactile sensor.")

    print(f"🎯 Starting CALVIN evaluation, saving to {args.output_dir}")
    eval_end = args.eval_end
    if eval_end is None:
        eval_end = args.eval_start + int(args.num_sequences)
    num_sequences_total = max(int(args.num_sequences), int(eval_end))

    def _run_eval_once(run_output_dir: str) -> dict:
        reset_everything(0, workers=True)
        Path(run_output_dir).mkdir(parents=True, exist_ok=True)

        if args.num_workers <= 1:
            env = get_env(dataset_path, show_gui=False, disable_tactile=disable_tactile)
            model = ClientModel(
                host,
                port,
                plan_len=args.plan_len,
                gripper_threshold=args.gripper_threshold,
            )
            results = evaluate_policy(
                model,
                env,
                run_output_dir,
                ep_len=args.ep_len,
                num_sequences=num_sequences_total,
                save_video_flag=not args.no_video,
                eval_start=args.eval_start,
                eval_end=eval_end,
            )
            list_r = count_success(results)
            score = sum(list_r)
            summary = {
                "num_workers": args.num_workers,
                "eval_start": args.eval_start,
                "eval_end": eval_end,
                "num_sequences": len(results),
                "score": score,
                "sr": list_r,
            }
            summary_path = Path(run_output_dir) / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2))
            print(f"\n✅ Evaluation completed! score={score:.3f}")
            print(f"📝 Summary written to {summary_path}")
            return summary

        # Parallel workers
        ranges = _split_eval_ranges(args.eval_start, eval_end, args.num_workers)
        if not ranges:
            print("⚠️ Empty evaluation range.")
            return {
                "num_workers": args.num_workers,
                "eval_start": args.eval_start,
                "eval_end": eval_end,
                "num_sequences": 0,
                "score": 0.0,
                "sr": [],
            }
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        args_dict = {
            "plan_len": args.plan_len,
            "gripper_threshold": args.gripper_threshold,
            "ep_len": args.ep_len,
            "num_sequences_total": num_sequences_total,
            "output_dir": run_output_dir,
            "save_video": not args.no_video,
        }
        procs = []
        for worker_id, (start, end) in enumerate(ranges):
            p = ctx.Process(
                target=_run_worker,
                args=(
                    worker_id,
                    host,
                    port,
                    args_dict,
                    str(dataset_path),
                    disable_tactile,
                    start,
                    end,
                    result_queue,
                ),
            )
            p.start()
            procs.append(p)

        worker_results = []
        while len(worker_results) < len(procs):
            try:
                worker_results.append(result_queue.get(timeout=60))
            except queue.Empty:
                # A worker that died without reporting would otherwise leave
                # this loop waiting forever.
                if not any(p.is_alive() for p in procs):
                    break
        for p in procs:
            p.join()
        errors = [item for item in worker_results if "error" in item]
        for item in errors:
            print(f"❌ Worker {item['worker_id']} failed:\n{item['error']}")
        if errors or len(worker_results) < len(procs):
            raise SystemExit(f"{len(errors) + len(procs) - len(worker_results)} of "
                             f"{len(procs)} CALVIN workers failed; no summary written")

        # Aggregate results
        total_results = []
        for item in worker_results:
            total_results.extend(item["results"])
        list_r = count_success(total_results) if total_results else []
        score = sum(list_r)
        summary = {
            "num_workers": args.num_workers,
            "eval_start": args.eval_start,
            "eval_end": eval_end,
            "num_sequences": len(total_results),
            "score": score,
            "sr": list_r,
        }
        summary_path = Path(run_output_dir) / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\n✅ Evaluation completed! score={score:.3f}")
        print(f"📝 Summary written to {summary_path}")
        return summary

    print(f"🔁 Evaluating (output_dir={args.output_dir})")
    _run_eval_once(args.output_dir)


if __name__ == "__main__":
    main()
