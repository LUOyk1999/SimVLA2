#!/usr/bin/env python3
"""SimVLA2 LIBERO policy server (websocket, request-level dynamic batching).

    python evaluation/libero/serve_simvla2_libero.py \
        --checkpoint runs/libero/ckpt-30000 --port 8000

A single eval job can run several LIBERO suites as concurrent clients against
one server; see evaluation/serving.py for how requests are batched.
--max_batch / --window_ms / --no_fast_path tune the batching; their defaults
are the setting the reported numbers used.

Request (msgpack, arrays via msgpack-numpy): "observation/image" (agentview),
"observation/wrist_image", "observation/state" [8], "prompt".
Response: {"actions": [[7 floats] x horizon]}.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluation.serving import (
    BatchedPolicyServer,
    add_batching_args,
    add_policy_args,
    fit_proprio,
    load_policy,
    setup_logging,
)


class LiberoServer(BatchedPolicyServer):
    def parse_request(self, observation):
        image0 = observation.get("observation/image")
        image1 = observation.get("observation/wrist_image")
        state = observation.get("observation/state", np.zeros(self.state_dim))
        if not isinstance(image0, np.ndarray):
            image0 = np.array(image0, dtype=np.uint8)
        if not isinstance(image1, np.ndarray):
            image1 = np.array(image1, dtype=np.uint8)
        return [image0, image1], fit_proprio(state, self.state_dim), observation.get("prompt", "")


def main():
    parser = argparse.ArgumentParser(description="SimVLA2 LIBERO Server (WebSocket)")
    add_policy_args(parser, bench="libero", default_dtype="bf16", default_flow_steps=10)
    add_batching_args(parser)
    args = parser.parse_args()

    setup_logging()
    model, processor, device = load_policy(args, expected_mode="libero")
    LiberoServer(model, processor, device, args).serve_forever(args.host, args.port)


if __name__ == "__main__":
    main()
