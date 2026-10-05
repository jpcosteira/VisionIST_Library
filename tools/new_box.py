#!/usr/bin/env python3
"""Scaffold a new box from template_box/.

Copies the template, renames the service file to <key>_service.py, substitutes
the name and key throughout, and syncs the contract into it. What you get
builds and answers correctly before you have written a line of your own code -
so the first thing you debug is your model, not the envelope.

    python3 tools/new_box.py depth_anything \
        --summary "Monocular depth with Depth Anything v2" \
        --runtime gpu --tags depth,monocular --github your-handle

    cd boxes/depth_anything
    # write src/depth_anything_service.py, fill in box.yaml
    python3 ../../tools/validate_boxes.py depth_anything
    python3 ../../tools/build_index.py
"""

import argparse
import pathlib
import shutil
import subprocess
import sys

from _registry import repo_root


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("name", help="box directory name: lowercase, underscores")
    ap.add_argument("--key", default=None,
                    help="config section name (default: same as the box name)")
    ap.add_argument("--summary", default="One line describing what this box does")
    ap.add_argument("--runtime", default="cpu",
                    choices=["cpu", "gpu-or-cpu", "gpu", "cuda-only"])
    ap.add_argument("--tags", default="", help="comma-separated")
    ap.add_argument("--github", default="your-github-handle")
    args = ap.parse_args()

    name = args.name
    key = args.key or name
    if not name.replace("_", "").isalnum() or not name[0].isalpha() \
            or name != name.lower():
        sys.exit(f"{name!r} must be lowercase letters, digits and underscores, "
                 f"starting with a letter")

    root = repo_root()
    dest = root / "boxes" / name
    if dest.exists():
        sys.exit(f"boxes/{name} already exists")
    template = root / "template_box"
    if not template.is_dir():
        sys.exit("template_box/ is missing")

    shutil.copytree(template, dest)

    # The Dockerfile renames <SERVICE_NAME>_service.py to service.py inside
    # the image, so the file has to be named after the config key.
    (dest / "src" / "template_service.py").rename(dest / "src" / f"{key}_service.py")

    tags = [t.strip() for t in args.tags.split(",") if t.strip()] or [name]
    manifest = (dest / "box.yaml").read_text()
    manifest = (manifest
                .replace("# template - VisionIST box manifest",
                         f"# {name} - VisionIST box manifest")
                .replace("name: template  ", f"name: {name}  ")
                .replace("key: template   ", f"key: {key}   ")
                .replace("summary: One line describing what this box does, "
                         "for the registry table", f"summary: {args.summary}")
                .replace("runtime: cpu  ", f"runtime: {args.runtime}  ")
                .replace("tags: [template]", f"tags: [{', '.join(tags)}]")
                .replace("repository: sipgisr/visionist-template",
                         f"repository: sipgisr/visionist-{name.replace('_', '-')}")
                .replace("github: your-github-handle", f"github: {args.github}"))
    (dest / "box.yaml").write_text(manifest)

    service = dest / "src" / f"{key}_service.py"
    service.write_text(service.read_text().replace('BOX_KEY = "template"',
                                                   f'BOX_KEY = "{key}"'))

    dockerfile = dest / "docker" / "Dockerfile"
    dockerfile.write_text(dockerfile.read_text()
                          .replace("ARG SERVICE_NAME=template",
                                   f"ARG SERVICE_NAME={key}"))

    readme = dest / "README.md"
    if readme.is_file():
        readme.write_text(readme.read_text().replace("template", name))

    subprocess.run([sys.executable, str(root / "tools" / "sync_contract.py")],
                   check=False, capture_output=True)

    print(f"boxes/{name}/ created (section {key!r})\n")
    print("next:")
    print(f"  1. write boxes/{name}/src/{key}_service.py")
    print(f"  2. fill in boxes/{name}/box.yaml and README.md")
    print(f"  3. add a test fixture under 256 KB (bigger ones go in assets:)")
    print(f"  4. python3 tools/validate_boxes.py {name}")
    print(f"  5. python3 tools/build_index.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
