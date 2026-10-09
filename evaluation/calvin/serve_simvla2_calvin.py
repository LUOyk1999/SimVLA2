#!/usr/bin/env python3
"""SimVLA2 CALVIN policy server (WebSocket).

    python evaluation/calvin/serve_simvla2_calvin.py \
        --checkpoint runs/calvin/ckpt-200000 --port 8000

Serves calvin_client.py. Request (msgpack-numpy): "image0" / "image1" (static
and gripper camera), "proprio" and "language_instruction"; reply
{"actions": [T, 20]}. Concurrent clients are batched, see
evaluation/serving.py. The number of flow-matching steps is a server
setting (--flow_steps).
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
    load_policy,
    setup_logging,
)


class CalvinServer(BatchedPolicyServer):
    def parse_request(self, request):
        images = [Image.fromarray(np.asarray(request[key]))
                  for key in ("image0", "image1", "image2") if key in request]
        if not images:
            raise ValueError("no images in the request")
        # Not routed through float32: the client sends float64, cast straight
        # to the weight dtype.
        proprio = np.array(request["proprio"])
        return images, proprio, request["language_instruction"]


def main():
    parser = argparse.ArgumentParser(description="SimVLA2 CALVIN Server (WebSocket)")
    add_policy_args(parser, bench="calvin", default_dtype="bf16", default_flow_steps=20)
    add_batching_args(parser)
    args = parser.parse_args()

    setup_logging()
    model, processor, device = load_policy(args, expected_mode="calvin")
    CalvinServer(model, processor, device, args).serve_forever(args.host, args.port)


if __name__ == "__main__":
    main()
