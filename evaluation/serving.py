"""Policy server code shared by every ``evaluation/<bench>/serve_simvla2_*.py``:
loading the checkpoint, and a websocket server with request-level dynamic
batching.

Besides the standard library, only numpy, torch, msgpack, websockets and
``models`` are imported at module level, so a server can also run in a
simulator's env that lacks the training stack.

One eval job runs several simulators as concurrent clients against one server,
so requests are batched instead of serialized on the event loop. Batching is
exact: ``encode_language`` pads to a fixed ``language_max_length`` and images
are always padded to ``num_views``, so every request produces the same
sequence length and ``forward_vlm`` never has to pad across the batch.

A benchmark subclasses :class:`BatchedPolicyServer` and implements
``parse_request``; the other hooks (wire codec, replies, post-processing)
default to the msgpack-numpy protocol of ``evaluation/policy_client.py``.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import msgpack
import numpy as np
import torch
import websockets

from models.action_hub import build_action_space
from models.configuration_simvla2 import SimVLA2Config
from models.modeling_simvla2 import SimVLA2
from models.processing_simvla2 import SimVLA2Processor

logger = logging.getLogger("simvla2.serving")

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


# -----------------------------------------------------------------------------
# Command line, model loading
# -----------------------------------------------------------------------------

def add_policy_args(parser: argparse.ArgumentParser, *, bench: str, default_dtype: str,
                    default_flow_steps: int) -> None:
    """The flags every server takes; eval_common.sh passes the first five."""
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to SimVLA2 checkpoint directory (contains model.safetensors + config.json)")
    parser.add_argument("--norm_stats", type=str, default=None,
                        help=f"Norm stats JSON, e.g. norm_stats/{bench}.json "
                             "(default: the checkpoint's own norm_stats.json)")
    parser.add_argument("--dtype", type=str, default=default_dtype, choices=sorted(DTYPES),
                        help=f"Weight dtype at inference (default: {default_dtype}; fp32 on CPU)")
    parser.add_argument("--flow_steps", type=int, default=default_flow_steps,
                        help=f"Flow-matching Euler steps (default: {default_flow_steps})")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="bind address; use 0.0.0.0 to accept remote clients")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Override tokenizer (default: read from model config)")
    parser.add_argument("--siglip_model_path", type=str, default=None,
                        help="Build the vision tower from this path instead of the one in the "
                             "checkpoint config. The checkpoint overwrites these weights, so "
                             "this only changes where the throwaway read comes from.")
    parser.add_argument("--image_size", type=int, default=None,
                        help="Override image size (default: read from model config)")
    parser.add_argument("--num_views", type=int, default=None,
                        help="Override num_views (default: read from model config)")


def add_batching_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max_batch", type=int, default=8,
                        help="Most requests coalesced into one forward")
    parser.add_argument("--window_ms", type=float, default=8.0,
                        help="How long to wait for more requests once one has arrived")
    parser.add_argument("--no_fast_path", action="store_true",
                        help="Always wait out the window, even with an empty queue")


def load_policy(
    args: argparse.Namespace,
    *,
    expected_mode: str,
    use_quantile_norm: Optional[bool] = None,
) -> Tuple[SimVLA2, SimVLA2Processor, torch.device]:
    """Load the checkpoint, its norm stats and a matching processor.

    ``--norm_stats`` replaces the action space with one built from that file;
    ``use_quantile_norm`` defaults to what the checkpoint config recorded.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = DTYPES[args.dtype]
    if device.type == "cpu" and dtype != torch.float32:
        logger.warning("CPU doesn't support low precision, falling back to fp32")
        dtype = torch.float32

    logger.info(f"Loading SimVLA2 from {args.checkpoint}...")
    config = SimVLA2Config.from_pretrained(args.checkpoint)
    if args.siglip_model_path:
        # Loading the full pretrained vision tower only to have the checkpoint
        # overwrite its weights wastes a large read, so build against a
        # node-local mirror of the weights when one has been staged.
        logger.info(f"Vision tower build path: {config.siglip_model_path} -> {args.siglip_model_path}")
        config.siglip_model_path = args.siglip_model_path
    model = SimVLA2.from_pretrained(args.checkpoint, config=config, torch_dtype=dtype)
    if model.action_mode != expected_mode:
        logger.warning(f"Checkpoint action_mode is '{model.action_mode}', not '{expected_mode}'.")

    if args.norm_stats:
        norm_stats = os.path.expanduser(args.norm_stats)
        if not os.path.exists(norm_stats):
            raise FileNotFoundError(f"Norm stats not found: {norm_stats}")
        if use_quantile_norm is None:
            use_quantile_norm = bool(model.config.use_quantile_norm)
        logger.info(f"Loading norm stats from {norm_stats} "
                    f"({'quantile' if use_quantile_norm else 'z-score'})")
        model.action_space = build_action_space(
            model.action_mode, norm_stats_path=norm_stats,
            use_quantile_norm=use_quantile_norm,
        )
    elif model.action_space.action_norm_stats is not None:
        logger.info("Using the norm stats saved in the checkpoint")
    else:
        logger.warning("No norm_stats loaded! Inference will likely be garbage.")

    model = model.to(device=device, dtype=dtype).eval()
    model.action_space.to(device)

    processor = SimVLA2Processor(
        tokenizer_name=args.tokenizer_name or model.config.tokenizer_name,
        image_size=args.image_size or model.config.image_size,
        num_views=args.num_views or model.config.num_views,
    )
    logger.info(
        f"Model loaded. device={device}, dtype={dtype}, "
        f"image_size={processor.image_size}, num_views={processor.num_views}, "
        f"action_mode={model.action_mode}, num_actions={model.num_actions}, "
        f"flow_steps={args.flow_steps}"
    )
    return model, processor, device


def fit_proprio(proprio, dim: int) -> np.ndarray:
    """Truncate or zero-pad a 1-D float32 proprio to ``dim``."""
    proprio = np.asarray(proprio, dtype=np.float32).reshape(-1)
    if proprio.size < dim:
        proprio = np.pad(proprio, (0, dim - proprio.size))
    return proprio[:dim]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# -----------------------------------------------------------------------------
# Websocket server
# -----------------------------------------------------------------------------

@dataclass
class BatchItem:
    """One request, already preprocessed into CPU tensors with a batch dim of 1.

    Preprocessing (image resize/normalize plus tokenization) happens in a worker
    thread per request, so it overlaps with other requests and with the GPU
    instead of running inside the event loop. The batch worker then only has to
    concatenate these and call the model.
    """
    input_ids: torch.Tensor    # [1, L]
    image_input: torch.Tensor  # [1, V, C, H, W]
    image_mask: torch.Tensor   # [1, V]
    proprio: torch.Tensor      # [dim_proprio]
    future: asyncio.Future
    attention_mask: Optional[torch.Tensor] = None  # [1, L], if the processor emits one


class BatchedPolicyServer:
    def __init__(self, model, processor, device, args: argparse.Namespace) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.flow_steps = int(args.flow_steps)
        self.max_batch = args.max_batch
        self.window_s = args.window_ms / 1000.0
        self.fast_path_solo = not args.no_fast_path

        self.action_dim = int(model.action_space.dim_action)
        self.state_dim = int(getattr(model.action_space, "dim_proprio", self.action_dim))
        self.action_horizon = int(model.num_actions)
        self.gripper_idx = [int(i) for i in model.action_space.gripper_idx]

        self._queue: Optional[asyncio.Queue] = None
        self._stats = {"batches": 0, "requests": 0, "size_hist": {}}

    # -------------------------------------------------------------------------
    # Hooks
    # -------------------------------------------------------------------------

    def parse_request(self, request: Dict[str, Any]) -> Tuple[Sequence[np.ndarray], np.ndarray, str]:
        """Return ``(images, proprio, prompt)``; raise on a bad request.

        ``proprio`` is a 1-D array, cast straight to the weight dtype.
        """
        raise NotImplementedError

    def postprocess(self, actions: np.ndarray, items: List[BatchItem]) -> np.ndarray:
        return actions

    def metadata(self) -> Dict[str, Any]:
        """First frame on every connection."""
        return {
            "model": "SimVLA2",
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "image_size": self.processor.image_size,
            "state_dim": self.state_dim,
            # The clients cannot import models.action_hub (it pulls in
            # transformers), so the gripper channels are advertised here.
            "gripper_idx": self.gripper_idx,
        }

    def pack_metadata(self, metadata: Dict[str, Any]) -> bytes:
        import msgpack_numpy
        return msgpack_numpy.packb(metadata, use_bin_type=True)

    def unpack(self, message: bytes) -> Dict[str, Any]:
        import msgpack_numpy
        return msgpack_numpy.unpackb(message, raw=False)

    def is_control(self, request: Dict[str, Any]) -> bool:
        """Messages that get no reply."""
        return False

    def on_inference_ok(self) -> None:
        pass

    def on_inference_error(self, error: Exception) -> np.ndarray:
        # Re-raising sends pack_error() instead of a reply, so the client
        # stops instead of executing actions the policy never produced.
        raise error

    def pack_reply(self, actions: np.ndarray):
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()
        return msgpack.packb({"actions": actions}, use_bin_type=True)

    def pack_error(self, error: Exception):
        return f"Error: {error}"

    # -------------------------------------------------------------------------
    # Batching
    # -------------------------------------------------------------------------

    def _build_item(self, request: Dict[str, Any], future: asyncio.Future) -> BatchItem:
        images, proprio, prompt = self.parse_request(request)
        inputs = self.processor(images=list(images), language_instruction=[prompt])
        return BatchItem(
            input_ids=inputs["input_ids"],
            image_input=inputs["image_input"],
            image_mask=inputs["image_mask"],
            proprio=torch.from_numpy(proprio),
            future=future,
            attention_mask=inputs.get("attention_mask"),
        )

    def _run_batch(self, items: List[BatchItem]) -> np.ndarray:
        """Concatenate items along the batch dim and run one forward.

        Every item has the same sequence length and view count, so this is a
        plain concat with no padding.
        """
        model_dtype = next(self.model.parameters()).dtype
        input_ids = torch.cat([it.input_ids for it in items], dim=0).to(self.device)
        image_input = torch.cat([it.image_input for it in items], dim=0).to(
            device=self.device, dtype=model_dtype)
        image_mask = torch.cat([it.image_mask for it in items], dim=0).to(self.device)
        proprio = torch.stack([it.proprio for it in items], dim=0).to(
            device=self.device, dtype=model_dtype)
        attention_mask = None
        if items[0].attention_mask is not None:
            attention_mask = torch.cat([it.attention_mask for it in items], dim=0).to(self.device)

        with torch.no_grad():
            actions = self.model.generate_actions(
                input_ids=input_ids,
                image_input=image_input,
                image_mask=image_mask,
                proprio=proprio,
                steps=self.flow_steps,
                attention_mask=attention_mask,
            )
        return self.postprocess(actions.float().cpu().numpy(), items)

    async def _batch_worker(self) -> None:
        """Drain the queue into batches and hand each batch to the model."""
        logger.info(f"batch_worker started: max_batch={self.max_batch} "
                    f"window_ms={self.window_s * 1000:.1f} fast_path_solo={self.fast_path_solo}")
        queue = self._queue
        while True:
            try:
                items: List[BatchItem] = [await queue.get()]
            except asyncio.CancelledError:
                return

            # With nothing else queued, dispatch at once instead of waiting out
            # the batching window.
            if not (self.fast_path_solo and queue.empty()):
                deadline = time.perf_counter() + self.window_s
                while len(items) < self.max_batch:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    try:
                        items.append(await asyncio.wait_for(queue.get(), timeout=remaining))
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        break

            while len(items) < self.max_batch:
                try:
                    items.append(queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            try:
                actions = await asyncio.to_thread(self._run_batch, items)
            except Exception as error:
                logger.error(f"Inference error: {error}")
                traceback.print_exc()
                for item in items:
                    if not item.future.done():
                        item.future.set_exception(error)
                continue

            stats = self._stats
            stats["batches"] += 1
            stats["requests"] += len(items)
            stats["size_hist"][len(items)] = stats["size_hist"].get(len(items), 0) + 1
            for index, item in enumerate(items):
                if not item.future.done():
                    item.future.set_result(actions[index])

    async def _stats_reporter(self, period: float = 60.0) -> None:
        while True:
            await asyncio.sleep(period)
            s = self._stats
            if s["batches"]:
                hist = " ".join(f"{k}:{v}" for k, v in sorted(s["size_hist"].items()))
                logger.info(f"batching: {s['requests']} reqs in {s['batches']} batches "
                            f"(avg {s['requests'] / s['batches']:.2f}) sizes[{hist}]")

    # -------------------------------------------------------------------------
    # Connections
    # -------------------------------------------------------------------------

    async def _handle_connection(self, websocket, path=None) -> None:
        logger.info(f"Connection from {websocket.remote_address} opened")
        loop = asyncio.get_running_loop()
        try:
            await websocket.send(self.pack_metadata(self.metadata()))
            async for message in websocket:
                try:
                    request = self.unpack(message)
                    if self.is_control(request):
                        continue
                    item = await asyncio.to_thread(self._build_item, request, loop.create_future())
                    await self._queue.put(item)
                    try:
                        actions = await item.future
                    except Exception as error:
                        actions = self.on_inference_error(error)
                    else:
                        self.on_inference_ok()
                    await websocket.send(self.pack_reply(actions))
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    traceback.print_exc()
                    await websocket.send(self.pack_error(e))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            logger.info(f"Connection from {websocket.remote_address} closed")

    async def _serve(self, host: str, port: int) -> None:
        self._queue = asyncio.Queue()
        logger.info(f"Creating SimVLA2 server (host: {host}, port: {port})")
        # ping_interval=None: with several simulators queued behind one GPU, a
        # client can wait longer than the default 20 s keepalive timeout, and
        # websockets then closes the connection out from under it (1011
        # keepalive ping timeout) mid-episode. Both ends run on this host, so
        # liveness detection buys nothing here.
        async with websockets.serve(self._handle_connection, host, port, max_size=None,
                                    compression=None, ping_interval=None, ping_timeout=None):
            logger.info(f"SimVLA2 server listening on {host}:{port}")
            worker = asyncio.create_task(self._batch_worker())
            reporter = asyncio.create_task(self._stats_reporter())
            try:
                await asyncio.Future()
            finally:
                worker.cancel()
                reporter.cancel()

    def serve_forever(self, host: str, port: int) -> None:
        logger.info(f"  Action horizon: {self.action_horizon}, action_dim={self.action_dim}, "
                    f"state_dim={self.state_dim}")
        logger.info(f"  Batching      : max_batch={self.max_batch} "
                    f"window_ms={self.window_s * 1000:.1f} fast_path_solo={self.fast_path_solo}")
        asyncio.run(self._serve(host, port))
