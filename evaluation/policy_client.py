"""Websocket client for the SimVLA2 policy servers (evaluation/serving.py).

It runs in the simulators' envs and needs only numpy, msgpack, msgpack-numpy
and websockets. Framing: msgpack frames with arrays via msgpack-numpy, a
metadata frame on connect, and a text frame instead of a reply when the server
fails.

A request blocks until the server answers. There is no fallback action: a
server error, a dropped connection or no reply within ``reply_timeout`` raises,
so the rollout stops rather than going on with a substituted action.
"""
from __future__ import annotations

import inspect
from typing import Any, Dict

import msgpack_numpy
from websockets.sync.client import connect


class PolicyClient:
    def __init__(self, host: str, port: int, reply_timeout: float = 600.0) -> None:
        # Never proxied: websockets>=15 honors http_proxy and cannot parse CIDR
        # no_proxy entries such as 127.0.0.0/8. No keepalive pings: with several
        # simulators queued behind one GPU a reply can take longer than the
        # 20 s ping timeout. Older websockets (13.x is the last on Python 3.8)
        # have neither option and hand unknown keywords to the socket, so only
        # the supported ones are passed.
        supported = inspect.signature(connect).parameters
        optional = {k: None for k in ("proxy", "ping_interval", "ping_timeout") if k in supported}
        self._conn = connect(f"ws://{host}:{port}", max_size=None, compression=None, **optional)
        self.reply_timeout = reply_timeout
        self.metadata = msgpack_numpy.unpackb(self._conn.recv(timeout=reply_timeout), raw=False)

    def infer(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self._conn.send(msgpack_numpy.packb(request, use_bin_type=True))
            reply = self._conn.recv(timeout=self.reply_timeout)
        except BaseException:
            # A late reply would otherwise be read as the answer to the next
            # request; closing makes every later call raise instead.
            self._conn.close()
            raise
        if isinstance(reply, str):
            raise RuntimeError(f"policy server: {reply}")
        return msgpack_numpy.unpackb(reply, raw=False)

    def close(self) -> None:
        self._conn.close()
