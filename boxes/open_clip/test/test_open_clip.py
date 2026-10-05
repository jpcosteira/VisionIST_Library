#!/usr/bin/env python3
"""Live test for the open_clip box (shared envelope interface).

Connects to a running box and:
  1. lists what the box offers (command "models")
  2. encodes two images and a few prompts with the default model
  3. checks shapes, that embeddings are unit length and that the similarity is
     their cosine
  4. prints the zero-shot ranking, so a real checkpoint should say
     dog.jpg -> "a dog" and car.jpg -> "a race car"

With the default checkpoint the ranking is meaningful; against a box running
random weights (the in-process smoke test's loader) only the structure is.

    python test/test_open_clip.py
    BOX_HOST=10.0.0.5:8061 python test/test_open_clip.py
    OPEN_CLIP_TEST_MODEL=ViT-L-14 OPEN_CLIP_TEST_PRETRAINED=datacomp_xl_s13b_b90k \\
        python test/test_open_clip.py

dog.jpg is a declared asset: `python tools/fetch_assets.py open_clip` from the
repo root downloads it. Without it the test uses car.jpg alone.
"""

import io
import json
import os
import sys

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_TEST_DIR, "..", "protos"))

import grpc  # noqa: E402
import numpy as np  # noqa: E402
try:
    import pipeline_pb2  # noqa: E402
    import pipeline_pb2_grpc  # noqa: E402
    import aux  # noqa: E402
except ImportError:
    # protos/ ships the .proto only; visionist-client carries compiled stubs
    # and the same wrap/unwrap helpers.
    from visionist_client._pb import pipeline_pb2, pipeline_pb2_grpc, aux  # noqa: E402

BOX = "open_clip"
TEXTS = ["a dog", "a cat", "a race car", "a diagram", "grass"]


def call(stub, section, data=None):
    req = pipeline_pb2.Envelope(config_json=json.dumps({BOX: section}),
                                data=data or {})
    resp = stub.Process(req)
    cfg = json.loads(resp.config_json or "{}").get(BOX, {})
    return cfg, resp


def field(resp, name):
    blob = aux.unwrap_value(resp.data[name])
    return np.load(io.BytesIO(bytes(blob)))


def main():
    target = os.getenv("BOX_HOST", "localhost:8061")
    print(f"Target: {target}")
    channel = grpc.insecure_channel(target, options=[
        ("grpc.max_send_message_length", -1),
        ("grpc.max_receive_message_length", -1)])
    stub = pipeline_pb2_grpc.PipelineServiceStub(channel)

    cfg, _ = call(stub, {"command": "models"})
    if cfg.get("status") != "done":
        print(f"  ERROR: {cfg}")
        return 1
    print(f"default: {cfg['default']}   {cfg['num_models']} models   "
          f"loaded: {cfg['loaded']}")

    names = [n for n in ("dog.jpg", "car.jpg")
             if os.path.isfile(os.path.join(_TEST_DIR, n))]
    if not names:
        print("no test images found next to this script")
        return 1
    images = []
    for n in names:
        with open(os.path.join(_TEST_DIR, n), "rb") as fh:
            images.append(fh.read())
    print(f"images: {names}")

    params = {}
    if os.getenv("OPEN_CLIP_TEST_MODEL"):
        params["model"] = os.environ["OPEN_CLIP_TEST_MODEL"]
        params["pretrained"] = os.getenv("OPEN_CLIP_TEST_PRETRAINED", "")
    cfg, resp = call(stub, {"command": "encode", "parameters": params},
                     {"images": aux.wrap_value(images), "texts": aux.wrap_value(TEXTS)})
    print("config:", {k: v for k, v in cfg.items() if k != "encoding"})
    if cfg.get("status") != "done":
        print(f"  ERROR: {cfg.get('error')}")
        return 1

    ie, te, sim, probs = (field(resp, f)
                          for f in ("image_emb", "text_emb", "similarity", "probs"))
    for name, arr in (("image_emb", ie), ("text_emb", te),
                      ("similarity", sim), ("probs", probs)):
        print(f"{name}: shape={arr.shape} dtype={arr.dtype}")

    ok = True
    ok &= ie.shape[0] == len(images) and te.shape[0] == len(TEXTS)
    ok &= bool(np.allclose(np.linalg.norm(ie, axis=1), 1, atol=1e-3))
    ok &= bool(np.allclose(sim, ie @ te.T, atol=1e-3))
    ok &= bool(np.allclose(probs.sum(1), 1, atol=1e-4))
    print("structure checks:", "ok" if ok else "FAILED")

    print("\nzero-shot ranking (softmax over the prompts):")
    for i, n in enumerate(names):
        top = np.argsort(-probs[i])[:3]
        print(f"  {n:>8s}: " + ", ".join(f"{TEXTS[j]}={probs[i][j]:.3f}" for j in top))

    print("\nDone.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
