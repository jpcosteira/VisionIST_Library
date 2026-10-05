#!/usr/bin/env python3
"""Download the test fixtures that are too big to commit.

A box declares anything over the size cap under `assets:` in its manifest
instead of committing it. This fetches them into the paths the tests expect,
verifies the sha256, and skips files already present and correct.

    python3 tools/fetch_assets.py                 # every box
    python3 tools/fetch_assets.py unimatch clip   # just these
    python3 tools/fetch_assets.py --list          # what would be fetched

Why this exists: a git repository keeps every byte of every version forever,
and everyone who clones pays for it. In the 11-box repo this registry came
from, two PNG fixtures in one box were 11 MB of a 16 MB tree, and the same
388 KB dog.jpg was committed three times under three names. At a hundred
boxes that pattern is the difference between a clone that takes seconds and
one that takes minutes.
"""

import argparse
import hashlib
import sys
import urllib.error
import urllib.request

from _registry import box_dirs, load_manifest, repo_root


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def human(n) -> str:
    if n is None:
        return "?"
    return f"{n/1024:.0f} KB" if n < 1024 * 1024 else f"{n/1024/1024:.1f} MB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("boxes", nargs="*", help="box names (default: all)")
    ap.add_argument("--list", action="store_true", help="show, do not download")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    root = repo_root()
    wanted = set(args.boxes)
    fetched = skipped = failed = 0

    for box in box_dirs(root):
        if wanted and box.name not in wanted:
            continue
        assets = load_manifest(box).get("assets") or []
        for asset in assets:
            dest = box / asset["path"]
            label = f"boxes/{box.name}/{asset['path']}"

            if args.list:
                print(f"  {label}  {human(asset.get('bytes'))}  {asset['url']}")
                continue

            if dest.is_file() and not args.force:
                if "sha256" not in asset or sha256(dest) == asset["sha256"]:
                    skipped += 1
                    continue
                print(f"  {label}: present but the checksum differs, re-fetching")

            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                with urllib.request.urlopen(asset["url"], timeout=120) as fh:
                    data = fh.read()
            except (urllib.error.URLError, TimeoutError) as e:
                print(f"  FAIL {label}: {e}", file=sys.stderr)
                failed += 1
                continue

            got = hashlib.sha256(data).hexdigest()
            if "sha256" in asset and got != asset["sha256"]:
                print(f"  FAIL {label}: sha256 mismatch\n"
                      f"       expected {asset['sha256']}\n"
                      f"       got      {got}", file=sys.stderr)
                failed += 1
                continue

            dest.write_bytes(data)
            print(f"  {label}  {human(len(data))}")
            fetched += 1

    if args.list:
        return 0
    print(f"\n{fetched} fetched, {skipped} already present"
          + (f", {failed} FAILED" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
