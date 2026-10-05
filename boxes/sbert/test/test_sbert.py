#!/usr/bin/env python3
"""Test script for the SBERT gRPC service (shared envelope interface).

Connects to a running textEmbedding box and:
  1. builds an Envelope with sample sentences in ``data.texts``
  2. calls ``stub.Process(request)``
  3. loads and prints the shape of ``embeddings`` / ``similarities``
     and the sentence-similarity matrix

Mirrors the style of images/clip/test/test_clip.py.

Run (from the repo or image root, server already up):
    python images/textEmbedding/test/test_sbert.py
    BOX_HOST=10.0.0.5:8061 python images/textEmbedding/test/test_sbert.py
"""

import json
import os
import sys

# Make the protos/ folder importable (same pattern as the clip test).
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_TEST_DIR, "..", "protos"))

import grpc  # noqa: E402
import pipeline_pb2  # noqa: E402
import pipeline_pb2_grpc  # noqa: E402
import aux  # noqa: E402


def bytes_to_tensor(b: bytes):
    import torch
    import io as bio
    return torch.load(bio.BytesIO(b), weights_only=False)


def main():
    target = os.getenv("BOX_HOST", "localhost:8061")
    print(f"Target: {target}")

    texts = [
        "o manel comeu um gelado.",
        "o carro anda rapido",
        "a bea bebeu agua muito fria",
    ]

    config = {
        "sbert": {
            "command": "encode",
            "parameters": {},
        }
    }

    request = pipeline_pb2.Envelope(
        config_json=json.dumps(config),
        data={"texts": aux.wrap_value(texts)},
    )

    channel = grpc.insecure_channel(
        target,
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ],
    )
    stub = pipeline_pb2_grpc.PipelineServiceStub(channel)

    response = stub.Process(request)

    cfg = json.loads(response.config_json or "{}")
    print(f"config: {cfg}")

    if "sbert" in cfg and cfg["sbert"].get("status") == "error":
        print(f"  ERROR: {cfg['sbert'].get('error')}")
        return 1

    tensors = {}
    for key in ("embeddings", "similarities"):
        if key not in response.data:
            print(f"missing field: {key}")
            continue
        val = aux.unwrap_value(response.data[key])
        if not isinstance(val, (bytes, bytearray)):
            print(f"{key}: (not bytes) {type(val)}")
            continue
        t = bytes_to_tensor(val)
        tensors[key] = t
        print(f"{key}: shape={tuple(t.shape)} dtype={t.dtype}")

    if "similarities" in tensors:
        sim = tensors["similarities"].detach().cpu().numpy()
        print("\nsentence-similarity matrix:")
        for i, lab in enumerate(texts):
            row = sim[i]
            top3 = sorted(range(len(texts)), key=lambda j: -row[j])[:3]
            top3_str = ", ".join(f"#{j}={row[j]:.4f}" for j in top3)
            print(f"  {i} ({lab[:30]:<30s}): {top3_str}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
