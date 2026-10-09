#!/usr/bin/env python3
"""SimVLA2 Bridge policy server (WidowX in SimplerEnv, WebSocket).

    python evaluation/bridge/serve_simvla2_bridge.py \
        --checkpoint runs/bridge/ckpt-30000 --port 8000

Request (msgpack-numpy): "image", "proprio" and "language_instruction"; reply
{"actions": [T, 7]}. Concurrent clients are batched, see
evaluation/serving.py. The number of flow-matching steps is a server
setting (--flow_steps).

State format (7D): [x, y, z, rx, ry, rz, gripper]
Action format (7D): absolute [x, y, z, rx, ry, rz, gripper]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluation.serving import (
    BatchedPolicyServer,
    add_batching_args,
    add_policy_args,
    fit_proprio,
    load_policy,
    setup_logging,
)


class BridgeServer(BatchedPolicyServer):
    def parse_request(self, request):
        image = np.asarray(request["image"])
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        proprio = fit_proprio(request["proprio"], 7)
        return [Image.fromarray(image)], proprio, request["language_instruction"]

    def postprocess(self, actions, items):
        actions = actions[..., :7].copy()
        actions[..., 6] = np.clip(actions[..., 6], 0.0, 1.0)
        return actions


def main():
    parser = argparse.ArgumentParser(description="SimVLA2 Bridge Server (WebSocket)")
    add_policy_args(parser, bench="bridge", default_dtype="bf16", default_flow_steps=30)
    add_batching_args(parser)
    args = parser.parse_args()

    setup_logging()
    model, processor, device = load_policy(args, expected_mode="bridge")
    BridgeServer(model, processor, device, args).serve_forever(args.host, args.port)


if __name__ == "__main__":
    main()
