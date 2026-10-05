"""Fetch the OpenD4RT checkpoint at image build time.

Kept as a file rather than inlined in the Dockerfile so the build fails with a
readable message rather than a shell quoting accident, and so the skip path
(--build-arg DOWNLOAD_WEIGHTS=false) is obvious.
"""

import os
import shutil
import sys

VARIANT = os.environ.get("HF_VARIANT", "OpenD4RT_48CLIP_9Mix_NoCropAUG")
REPO = os.environ.get("HF_REPO_ID", "Lijiaxin0111/OpenD4RT")
WANT = os.environ.get("WANT_WEIGHTS", "true").strip().lower()
FILES = ("model.yaml", "opend4rt.ckpt")
DEST = f"/weights/{VARIANT}"

if WANT not in ("true", "1", "yes", "on"):
    print(f"DOWNLOAD_WEIGHTS={WANT!r}: not fetching. Mount the checkpoint at "
          f"{DEST} (model.yaml + opend4rt.ckpt) when you run the box.")
    sys.exit(0)

from huggingface_hub import hf_hub_download  # noqa: E402

os.makedirs(DEST, exist_ok=True)
for name in FILES:
    remote = f"checkpoints/{VARIANT}/{name}"
    print(f"fetching {REPO}:{remote}", flush=True)
    try:
        cached = hf_hub_download(repo_id=REPO, filename=remote)
    except Exception as e:                                  # noqa: BLE001
        sys.exit(f"could not fetch {remote} from {REPO}: {e}\n"
                 f"If this host cannot reach huggingface.co, build with "
                 f"--build-arg DOWNLOAD_WEIGHTS=false and mount {DEST}.")
    shutil.copyfile(cached, os.path.join(DEST, name))
    size = os.path.getsize(os.path.join(DEST, name)) / 1e9
    print(f"  -> {DEST}/{name}  ({size:.2f} GB)", flush=True)

print("checkpoint ready")
