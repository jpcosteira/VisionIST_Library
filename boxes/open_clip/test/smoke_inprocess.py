"""In-process smoke test for the open_clip service: no box build, no weights.

It swaps the service's model loader for a randomly initialised ViT-B-32 (the
architecture is real, the weights are not), so everything the box does around
the model - validation, batching, normalisation, the cache, the envelope - runs
for real while the numbers themselves mean nothing. Meaningful embeddings need
the live test (test_open_clip.py) against a box that has downloaded weights.

    cd boxes/open_clip && python test/smoke_inprocess.py

Needs torch, open_clip_torch, Pillow, numpy and grpcio-tools locally.
"""
import io
import json
import os
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "protos")
try:
    import pipeline_pb2  # noqa: F401
except ImportError:
    # protos/ holds the .proto, not generated code (the Dockerfile compiles it):
    # compile into a temp dir for this run only.
    import subprocess
    import tempfile
    _gen = tempfile.mkdtemp(prefix="open_clip_pb_")
    subprocess.run([sys.executable, "-m", "grpc_tools.protoc", "-Iprotos",
                    f"--python_out={_gen}", f"--grpc_python_out={_gen}",
                    "protos/pipeline.proto"], check=True)
    sys.path.insert(0, _gen)
os.environ["OPEN_CLIP_IDLE_SECONDS"] = "0"        # no watchdog thread in a test

import numpy as np
import open_clip
from PIL import Image

import open_clip_service as svc
import pipeline_pb2
from aux import wrap_value, unwrap_value

LOADS = []


def fake_load(model, pretrained):
    """Same shape as svc.load_model, but never reads a checkpoint."""
    LOADS.append((model, pretrained))
    net, _t, pre = open_clip.create_model_and_transforms(model, pretrained=None,
                                                         device="cpu")
    net.eval()
    return svc._Loaded(net, pre, open_clip.get_tokenizer(model))


svc.load_model = fake_load


def jpeg(seed, size=(96, 80)):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def call(srv, section=None, images=None, texts=None, raw_config=None):
    data = {}
    if images is not None:
        data["images"] = wrap_value(images)
    if texts is not None:
        data["texts"] = wrap_value(texts)
    cfg = raw_config if raw_config is not None else (
        json.dumps({"open_clip": section}) if section is not None else "")
    return srv.Process(pipeline_pb2.Envelope(config_json=cfg, data=data), None)


def sec(resp):
    return json.loads(resp.config_json or "{}").get("open_clip", {})


def arr(resp, field):
    return np.load(io.BytesIO(bytes(unwrap_value(resp.data[field]))))


failures = []


def check(name, cond, detail=""):
    print(f"  {'ok ' if cond else 'FAIL'} {name} {detail}")
    if not cond:
        failures.append(name)


srv = svc.PipelineService()
A, B = jpeg(1), jpeg(2)
TX = ["a dog", "a cat", "a race car"]

print("== 1. images + texts, defaults ==")
r = call(srv, {"command": "encode"}, [A, B], TX)
s = sec(r)
check("status done", s.get("status") == "done", str(s.get("error", "")))
check("defaults used", (s.get("model"), s.get("pretrained")) ==
      ("ViT-B-32", "laion2b_s34b_b79k"), str((s.get("model"), s.get("pretrained"))))
ie, te, sim, pr = (arr(r, f) for f in ("image_emb", "text_emb", "similarity", "probs"))
check("image_emb (2,512)", ie.shape == (2, 512) and ie.dtype == np.float32, str(ie.shape))
check("text_emb (3,512)", te.shape == (3, 512), str(te.shape))
check("similarity (2,3)", sim.shape == (2, 3), str(sim.shape))
check("probs rows sum to 1", np.allclose(pr.sum(1), 1, atol=1e-5), str(pr.sum(1)))
check("embeddings are unit length", np.allclose(np.linalg.norm(ie, axis=1), 1, atol=1e-5))
check("similarity is cosine", np.allclose(sim, ie @ te.T, atol=1e-3))  # matmul may use reduced-precision kernels
check("declared encoding", s["encoding"] == {f: "numpy" for f in
      ("image_emb", "text_emb", "similarity", "probs")}, str(s.get("encoding")))
check("counts reported", (s["num_images"], s["num_texts"], s["embedding_dim"]) == (2, 3, 512))

print("== 2. images only / texts only ==")
r = call(srv, {"command": "encode"}, [A], None)
check("images only: image_emb, nothing else", set(r.data) == {"image_emb"}, str(set(r.data)))
r = call(srv, {"command": "encode"}, None, TX)
check("texts only: text_emb, nothing else", set(r.data) == {"text_emb"}, str(set(r.data)))

print("== 3. normalize=false, batching ==")
r = call(srv, {"parameters": {"normalize": False, "batch_size": 1}}, [A, B, A], TX)
raw = arr(r, "image_emb")
check("raw embeddings keep their norm", not np.allclose(np.linalg.norm(raw, axis=1), 1, atol=1e-3))
check("batch_size=1 gives the same rows as one batch",
      np.allclose(raw[0], raw[2], atol=1e-5) and raw.shape == (3, 512))
check("similarity is still cosine", np.allclose(
    arr(r, "similarity"), (raw / np.linalg.norm(raw, axis=1, keepdims=True)) @
    (arr(r, "text_emb") / np.linalg.norm(arr(r, "text_emb"), axis=1, keepdims=True)).T, atol=1e-3))

print("== 4. a second model, the cache, reset ==")
LOADS.clear()
call(srv, {"parameters": {"model": "RN50", "pretrained": "openai"}}, [A], TX)
check("RN50/openai loaded", LOADS == [("RN50", "openai")], str(LOADS))
call(srv, {"parameters": {"model": "RN50", "pretrained": "openai"}}, [A], TX)
check("second call hits the cache", len(LOADS) == 1, str(LOADS))
check("cache holds one model (OPEN_CLIP_CACHE_SIZE=1)", len(srv._cache) == 1, str(list(srv._cache)))
call(srv, {"parameters": {"model": "ViT-B/32"}}, [A], TX)       # OpenAI's spelling
check("'ViT-B/32' is accepted and normalised", LOADS[-1][0] == "ViT-B-32", str(LOADS))
r = call(srv, {"command": "reset"})
check("reset unloads", sec(r).get("status") == "done" and len(srv._cache) == 0
      and sec(r).get("unloaded") == 1, str(sec(r)))

print("== 5. models command ==")
r = call(srv, {"command": "models"})
s = sec(r)
check("lists models", s.get("status") == "done" and s["num_models"] > 20, str(s.get("num_models")))
check("reports the default", s["default"]["model"] == "ViT-B-32")
r = call(srv, {"command": "models", "parameters": {"model": "ViT-B-32"}})
check("per-model tag list", "laion2b_s34b_b79k" in sec(r)["models"]["ViT-B-32"])

print("== 6. errors are errors, empty is empty ==")
check("no config -> error", sec(call(srv, raw_config="")).get("status") == "error")
check("no data -> empty_request", sec(call(srv, {"command": "encode"})).get("status") == "empty_request")
check("bad json section -> error", sec(call(srv, raw_config='{"x":1}')).get("status") == "error")
for name, section, imgs, txts, frag in [
    ("unknown command", {"command": "dance"}, [A], TX, "unknown command"),
    ("unknown model", {"parameters": {"model": "ViT-Z-99", "pretrained": "x"}}, [A], TX, "unknown model"),
    ("unknown pretrained lists the real ones",
     {"parameters": {"pretrained": "nope"}}, [A], TX, "laion2b_s34b_b79k"),
    ("non-default model without a tag", {"parameters": {"model": "RN50"}}, [A], TX, "needs a 'pretrained'"),
    ("bad device", {"parameters": {"device": "tpu"}}, [A], TX, "device"),
    ("cuda without a GPU", {"parameters": {"device": "cuda"}}, [A], TX, "no GPU"),
    ("bad batch_size", {"parameters": {"batch_size": 0}}, [A], TX, "batch_size"),
    ("bad normalize", {"parameters": {"normalize": "yes"}}, [A], TX, "normalize"),
    ("undecodable image", {"command": "encode"}, [b"not an image"], TX, "decode image #1"),
]:
    s = sec(call(srv, section, imgs, txts))
    if name == "cuda without a GPU":
        import torch
        if torch.cuda.is_available():
            continue
    check(name, s.get("status") == "error" and frag in s.get("error", ""), str(s.get("error")))
check("an hf-hub model needs no tag", svc.resolve("hf-hub:org/repo", None) == ("hf-hub:org/repo", None))

print()
print("FAILED: " + ", ".join(failures) if failures else "PASS -- all open_clip smoke cases.")
sys.exit(1 if failures else 0)
