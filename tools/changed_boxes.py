#!/usr/bin/env python3
"""Which boxes did this change touch? Prints a CI matrix for just those.

A pull request that edits one box must not rebuild a hundred images - and one
of these boxes downloads 5 GB of weights at build time. This maps changed
paths to box directories and emits the same matrix shape as
`build_index.py --matrix`, so the build job is identical whether it runs on a
PR (changed boxes) or on a release (all of them).

A change to contract/, tools/ or registry/schema.json affects every box, so
those select everything.

    python3 tools/changed_boxes.py --base origin/main      # matrix JSON
    python3 tools/changed_boxes.py --base origin/main --names
"""

import argparse
import json
import subprocess
import sys

from _registry import load_all, repo_root

#: Paths that invalidate every box rather than one.
GLOBAL_PATHS = ("contract/", "tools/", "registry/schema.json",
                ".github/workflows/")


def changed_paths(base: str, head: str) -> list[str]:
    try:
        out = subprocess.run(["git", "diff", "--name-only", f"{base}...{head}"],
                             capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError:
        # No merge base (a shallow clone, or the first commit): fall back to a
        # plain two-dot diff rather than silently building nothing.
        out = subprocess.run(["git", "diff", "--name-only", base, head],
                             capture_output=True, text=True, check=True).stdout
    return [p for p in out.splitlines() if p.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--names", action="store_true",
                    help="print names one per line instead of a matrix")
    args = ap.parse_args()

    root = repo_root()
    manifests = {m["name"]: m for m in load_all(root)}
    paths = changed_paths(args.base, args.head)

    if any(p.startswith(GLOBAL_PATHS) for p in paths):
        selected = sorted(manifests)
        reason = "a shared path changed"
    else:
        selected = sorted({p.split("/")[1] for p in paths
                           if p.startswith("boxes/") and len(p.split("/")) > 2
                           and p.split("/")[1] in manifests})
        reason = f"{len(paths)} changed path(s)"

    print(f"{reason} -> {len(selected)} box(es)", file=sys.stderr)

    if args.names:
        for name in selected:
            print(name)
        return 0

    matrix = [{"name": n,
               "repository": manifests[n]["image"]["repository"],
               "version": manifests[n]["version"],
               "registries": ",".join(manifests[n]["image"].get(
                   "registries", ["ghcr", "dockerhub"])),
               "context": f"boxes/{n}",
               "key": manifests[n]["key"]} for n in selected]
    print(json.dumps({"box": matrix}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
