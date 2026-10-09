#!/usr/bin/env python3
"""SimVLA2 Fractal / RT-1 policy server (SimplerEnv Google robot, WebSocket).

    python evaluation/fractal/serve_simvla2_fractal.py \
        --checkpoint runs/fractal/ckpt-30000 --port 8000

Request (msgpack-numpy): "image", "proprio" and "language_instruction"; reply
{"actions": [T, 7]}. Concurrent clients are batched, see
evaluation/serving.py. Proprio is 7-D (position + Euler + gripper); the
action head produces the matching 7-D delta-position + Euler + gripper chunk
that ``fractal_client.py`` plays into SimplerEnv. The number of flow-matching
steps is a server setting (--flow_steps).
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


class FractalServer(BatchedPolicyServer):
    def parse_request(self, request):
        image = np.asarray(request["image"])
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        proprio = fit_proprio(request["proprio"], 7)
        return [Image.fromarray(image)], proprio, request["language_instruction"]

    def postprocess(self, actions, items):
        # [T, 7] = delta_pos[3] + euler_xyz[3] + gripper[1]
        actions = actions[..., :7].copy()
        actions[..., -1] = np.clip(actions[..., -1], -1.0, 1.0)
        return actions


def main():
    parser = argparse.ArgumentParser(description="SimVLA2 Fractal Server (WebSocket)")
    add_policy_args(parser, bench="fractal", default_dtype="bf16", default_flow_steps=30)
    add_batching_args(parser)
    args = parser.parse_args()

    setup_logging()
    model, processor, device = load_policy(args, expected_mode="fractal")
    FractalServer(model, processor, device, args).serve_forever(args.host, args.port)


if __name__ == "__main__":
    main()
