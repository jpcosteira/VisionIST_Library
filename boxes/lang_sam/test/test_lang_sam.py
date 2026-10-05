#!/usr/bin/env python3
"""Test script for the LangSAM (lang_segm) gRPC service (shared envelope).

Connects to a running lang_sam box and:
  1. builds an Envelope with the two photos bundled in the clip box's
     test folder (``dog.jpg`` - the dog in the grass, ``car.jpg`` - the car)
     plus matching text prompts
  2. calls stub.Process(request)
  3. decodes data["results"] (zstd + pickle) and prints the per-image
     masks / bboxes / scores

Mirrors the style of images/clip/test/test_clip.py.

Run (server already up on :8061):
    python images/lang_segm/test/test_lang_sam.py
    BOX_HOST=10.0.0.5:8061 python images/lang_segm/test/test_lang_sam.py
"""

import io
import json
import os
import sys

# Make the protos/ folder importable (same pattern as the clip test).
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_TEST_DIR, "..", "protos"))

import grpc  # noqa: E402
import numpy as np  # noqa: E402
import pipeline_pb2  # noqa: E402
import pipeline_pb2_grpc  # noqa: E402
import aux  # noqa: E402


_CLIP_TEST_DIR = os.path.normpath(os.path.join(_TEST_DIR, "..", "..", "clip", "test"))


def load_test_image(name: str) -> bytes:
    """Load a bundled test photo (from the clip box test folder).

    Falls back to a generated image if the file is not present.
    """
    path = os.path.join(_CLIP_TEST_DIR, name)
    if os.path.isfile(path):
        with open(path, "rb") as f:
            data = f.read()
        print(f"loaded {path}: {len(data) / (1024*1024):.2f} MB")
        return data

    print(f"warning: {path} not found, generating a placeholder image")
    from PIL import Image
    img = Image.new("RGB", (512, 512), (200, 200, 120))
    px = img.load()
    for y in range(112, 400):
        for x in range(112, 400):
            px[x, y] = (40, 120, 200)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def visualize(out, img_bytes, out_path) -> list:
    """Blend each mask (distinct color) over the original image, draw a
    legend (swatch + label + score) for each mask, and save the result."""
    from PIL import Image as PILImage, ImageDraw, ImageFont

    base = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = base.size
    canvas = np.array(base, dtype=np.float32)
    rng = np.random.default_rng(1234 + w * h)
    alpha = 0.45

    try:
        scores = [float(v) for v in np.asarray(out.get("mask_scores", np.array([]))).ravel()]
    except (TypeError, ValueError):
        scores = []
    # GDINO labels align 1:1 with the boxes, and SAM returns one mask per
    # box in order -> label[i] names mask[i].
    try:
        labels = [str(v) for v in list(out.get("labels", []))]
    except TypeError:
        labels = []
        
    def name_for(j: int) -> str:
        if j < len(labels) and labels[j]:
            return labels[j].strip().rstrip(".")
        return f"mask {j}"

    legend = []
    for j, m in enumerate(out.get("masks", [])):
        a = np.asarray(m)
        while a.ndim > 2:
            a = a[0]
        mask = a.astype(bool)
        if mask.shape != (h, w):
            mask = np.asarray(
                PILImage.fromarray(a.astype(np.uint8)).resize((w, h), PILImage.NEAREST) > 0)
        color = tuple(int(c) for c in rng.integers(40, 255, 3))
        canvas[mask] = (1 - alpha) * canvas[mask] + alpha * np.array(color)
        legend.append((j, color, scores[j] if j < len(scores) else None, name_for(j)))

    img = PILImage.fromarray(canvas.astype(np.uint8))

    # Legend: one row per mask, top-left, swatch + label/score.
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:  # older Pillow without size argument
        font = ImageFont.load_default()

    row_h, sw, pad = 26, 18, 8
    for i, (j, color, score, name) in enumerate(legend):
        y = pad + i * row_h
        # semi-opaque backing strip for readability
        txt = name
        if score is not None:
            txt += f"  (score {score:.3f})"
        # measure text (approximate: truetype gives getbbox, default bitmap too)
        tw = draw.textlength(txt, font=font) if hasattr(draw, "textlength") else 8 * len(txt)
        draw.rectangle([pad, y, pad + sw + 8 + int(tw) + 4, y + int(row_h * 0.8)],
                       fill=(0, 0, 0))
        draw.rectangle([pad + 2, y + 4, pad + 2 + sw - 4, y + int(row_h * 0.8) - 4],
                       fill=color, outline=(255, 255, 255))
        draw.text((pad + sw + 6, y + 2), txt, fill=(255, 255, 255), font=font)

    img.save(out_path)
    return legend


def main():
    target = os.getenv("BOX_HOST", "localhost:8061")
    print(f"Target: {target}")

    # The same two photos the clip box uses for its test (dog in the
    # grass, race car); the prompts below include matching subjects.
    image_files = ["dog.jpg", "car.jpg"]
    image_bytes_list = [load_test_image(n) for n in image_files]
    texts = ["a dog", "grass", "garden", "a race car", "a rectangle"]

    # Envelope config: section namespaced under the box key, like every
    # other box (tapnext / clip / sbert).
    config = {
        "lang_sam": {
            "command": "segment",
            "parameters": {},           # e.g. {"device": "cuda:0"}
            "text_prompt": texts,
        }
    }

    request = pipeline_pb2.Envelope(
        config_json=json.dumps(config),
        data={"images": aux.wrap_value(image_bytes_list)},
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

    section = next((v for v in cfg.values() if isinstance(v, dict)), {})
    if section.get("status") == "error":
        print(f"  ERROR: {section.get('error')}")
        return 1

    status = section.get("status")
    if status != "done":
        print(f"  unexpected status: {status}")
        return 1

    if "results" not in response.data:
        print("missing field: results")
        return 1

    blob = aux.unwrap_value(response.data["results"])
    if not isinstance(blob, (bytes, bytearray)):
        print(f"results: (not bytes) {type(blob)}")
        return 1

    import pickle
    import zstandard as zstd
    out_list = pickle.loads(zstd.ZstdDecompressor().decompress(bytes(blob)))

    print(f"\nper-image results ({len(out_list)} image(s), "
          f"prompt: '{' '.join(texts)}'.')")
    for i, out in enumerate(out_list):
        masks = out.get("masks", [])
        others = {k: (v.shape if hasattr(v, "shape") else v)
                  for k, v in out.items() if k not in ("masks", "mask_scores")}
        print(f"  image {i} ({image_files[i]}): {len(masks)} mask(s)")
        for j, m in enumerate(masks):
            frac = float(m.astype(bool).mean()) if hasattr(m, "mean") else None
            frac_str = f", area={frac:.1%}" if frac is not None else ""
            print(f"    mask {j}: shape={tuple(m.shape)}{frac_str}")
        if out.get("mask_scores") is not None:
            scores = [round(float(s), 4) for s in np.asarray(out["mask_scores"]).ravel()[:8]]
            print(f"    mask_scores: {scores}")
        if others:
            print(f"    other fields: {others}")

        # Paint each mask area in its own color and save the annotated image.
        out_path = os.path.join(
            _TEST_DIR, f"output_{os.path.splitext(image_files[i])[0]}_langsam.png")
        legend = visualize(out, image_bytes_list[i], out_path)
        print(f"  painted {len(legend)} mask area(s) -> {os.path.abspath(out_path)}")
        for j, color, score, name in legend:
            score_str = f" score={score:.3f}" if score is not None else ""
            print(f"    {name}: RGB{color}{score_str}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
