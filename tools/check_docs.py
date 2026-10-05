#!/usr/bin/env python3
"""Check that every box's written instructions match the registry's protocol.

A box is built and run one way (CONTRIBUTING.md, and the CI that enforces it):

    cd boxes/<name>
    docker build --tag sipgisr/visionist-<name> -f docker/Dockerfile \\
                 --build-arg SERVICE_NAME=<key> .
    docker run --rm -p 8061:8061 -e PORT=8061 sipgisr/visionist-<name>
    python test/test_<name>.py

The context is the box's own directory, the Dockerfile is docker/Dockerfile,
SERVICE_NAME is the config key, and the image is named after the manifest's
image.repository. This repo was split out of one with an `images/` directory
and different box names; instructions copied across carried those along, and a
README that says `cd images/lightglue_box` fails for everyone who trusts it.

What is checked, per box (README.md and any other *.md in the box, the test
scripts' docstrings, and the Dockerfile's own header comments):

  1. no leftover pre-split paths or names (images/, lightglue_box, ...)
  2. every `docker build`: runs from `cd boxes/<name>` in the same code block,
     uses -f docker/Dockerfile and context `.`, SERVICE_NAME (if given) equals
     the key, and the tag is the manifest's image repository
  3. every `docker run`: names that image (or its published reference at the
     manifest's version), publishes a host port onto 8061, and passes
     --gpus all when the box is gpu or cuda-only
  4. every `python ... test/...` command points at a file that exists, from a
     directory that is the box (or boxes/<name>/...)
  5. the Dockerfile builds from the box directory: every COPY source exists,
     and its default SERVICE_NAME is the key

    python3 tools/check_docs.py              # all boxes (and template_box)
    python3 tools/check_docs.py yolo clip    # just these
"""

import argparse
import glob
import pathlib
import re
import shlex
import sys

from _registry import box_dirs, load_manifest, repo_root

#: Names and paths from before the split. A hit is always a stale instruction.
STALE = [
    (r"\bimages/\w+/", "the old images/ directory (boxes live under boxes/)"),
    (r"\b(lightglue|opencv|features|moge)_box\b", "an old *_box name"),
    (r"\btapnext_tracker\b", "the old directory name tapnext_tracker"),
    (r"\btextEmbedding\b", "the old directory name textEmbedding"),
    (r"\blang_segm\b", "the old directory name lang_segm"),
]
#: Lines that mention an old name on purpose (compatibility aliases, history).
STALE_OK = re.compile(r"alias|also answers|also accepts|accepted|old callers|"
                      r"differ|NOT `|previously|was called|used to|lived in|"
                      r"formerly|legacy|renamed|pre-contract|section name|"
                      r"section names|_BOX_KEYS|aispgradio", re.I)


def fences(text):
    """Yield (first_line_no, [lines]) for each ``` code block."""
    inside, start, buf = False, 0, []
    for no, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith("```"):
            if inside:
                yield start, buf
                buf = []
            inside, start = not inside, no + 1
        elif inside:
            buf.append(line)


def commands(lines):
    """Join backslash continuations; strip prompts and trailing comments."""
    out, cur = [], ""
    for raw in lines:
        line = raw.rstrip()
        cur += (" " if cur else "") + line.rstrip("\\").strip()
        if line.endswith("\\"):
            continue
        s = re.sub(r"^\s*(\$|#\s{1,4})\s*", "", cur)       # "$ cmd" / "#   cmd"
        s = re.sub(r"\s+#\s.*$", "", s)                    # trailing comment
        out.append(s.strip())
        cur = ""
    return out


def split_shell(cmd):
    try:
        return shlex.split(cmd, comments=False)
    except ValueError:
        return cmd.split()


def check_commands(name, key, repo, runtime, version, where, lines, problems):
    """Checks 2-4 on one block of shell lines."""
    cmds = commands(lines)
    cds = [c for c in cmds if c.startswith("cd ")]
    want_cd = f"boxes/{name}"
    image_short = repo                                    # sipgisr/visionist-x
    published = f"docker.io/{repo}:{version}"
    built = set()

    for c in cmds:
        c = re.sub(r"^cd\s+\S+\s*&&\s*", "", c)           # "cd x && docker ..."
        words = split_shell(c)
        if not words:
            continue

        if words[:2] == ["docker", "build"] or words[:3] == ["docker", "buildx", "build"]:
            tag = None
            for i, w in enumerate(words):
                if w in ("--tag", "-t") and i + 1 < len(words):
                    tag = words[i + 1]
                if w.startswith("--tag="):
                    tag = w.split("=", 1)[1]
            if tag:
                built.add(tag)
            dockerfile = next((words[i + 1] for i, w in enumerate(words)
                               if w in ("-f", "--file") and i + 1 < len(words)), None)
            if dockerfile != "docker/Dockerfile":
                problems.append(f"{where}: docker build must use -f docker/Dockerfile "
                                f"(got {dockerfile!r})")
            if words[-1] != ".":
                problems.append(f"{where}: docker build context must be '.', the box "
                                f"directory (got {words[-1]!r})")
            svc = [w.split("=", 1)[1] for w in words if w.startswith("SERVICE_NAME=")]  # value of --build-arg SERVICE_NAME=...
            if not svc:
                problems.append(f"{where}: docker build must pass "
                                f"--build-arg SERVICE_NAME={key}")
            elif svc[0] != key:
                problems.append(f"{where}: SERVICE_NAME={svc[0]} but the key is {key!r}")
            if tag and tag.split(":")[0] != image_short:
                problems.append(f"{where}: tag {tag!r} should be {image_short!r}")
            if not tag:
                problems.append(f"{where}: docker build has no --tag")
            if not any(x.rstrip("/") == want_cd for x in (cd.split()[1] for cd in cds)):
                problems.append(f"{where}: a docker build must be preceded by "
                                f"'cd {want_cd}' in the same code block")

        elif words[:2] == ["docker", "run"]:
            image = None
            skip = False
            opts_with_arg = {"-p", "-e", "-v", "--name", "--gpus", "--ipc", "--shm-size",
                             "--network", "--env", "--volume", "--publish", "-w", "--entrypoint"}
            for w in words[2:]:
                if skip:
                    skip = False
                    continue
                if w in opts_with_arg:
                    skip = True
                    continue
                if w.startswith("-"):
                    continue
                image = w
                break
            ok_images = {image_short, published, f"{image_short}:{version}"} | built
            if image is None or image.split("@")[0] not in ok_images \
                    and image not in ok_images:
                problems.append(f"{where}: docker run image {image!r} is not "
                                f"{image_short!r} or {published!r}")
            if not re.search(r"(-p|--publish)\s+\d+:8061\b", c):
                problems.append(f"{where}: docker run must publish a host port onto "
                                f"the container's 8061")
            if runtime in ("gpu", "cuda-only") and "--gpus" not in c:
                problems.append(f"{where}: {runtime} box, but docker run has no --gpus all")

        elif re.match(r"^python3?\s", c) or c.startswith("BOX_HOST="):
            m = re.search(r"python3?\s+(?:-m\s+pytest\s+)?(\S+\.py)", c)
            if not m:
                continue
            path = m.group(1)
            if "tools/" in path or "contract/" in path or path.startswith("src/"):
                continue
            if not re.match(rf"^(test/|boxes/{name}/test/|\./test/)", path):
                problems.append(f"{where}: {path!r} - tests are run as test/<file> from "
                                f"boxes/{name}")


def check_box(box, problems):
    name = box.name
    m = load_manifest(box)
    key, repo = m["key"], m["image"]["repository"]
    runtime, version = m["runtime"], m["version"]
    tag = lambda f: f"{name}/{f}"                                     # noqa: E731

    # 1. stale names, in prose and code alike
    docs = sorted(box.rglob("*.md")) + sorted((box / "test").glob("*.py")) \
        + [box / "docker" / "Dockerfile"]
    for f in docs:
        for no, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            if STALE_OK.search(line):
                continue
            for pat, what in STALE:
                if re.search(pat, line):
                    problems.append(f"{tag(f.relative_to(box).as_posix())}:{no}: "
                                    f"{what}: {line.strip()[:90]}")

    # 2-4. commands in markdown code blocks
    for f in sorted(box.rglob("*.md")):
        for start, lines in fences(f.read_text(errors="replace")):
            if any(l.strip().startswith(("docker build", "docker run", "$ docker",
                                          "python", "cd ", "BOX_HOST=")) for l in lines):
                check_commands(name, key, repo, runtime, version,
                               f"{tag(f.relative_to(box).as_posix())}:{start}",
                               lines, problems)

    # ... in the Dockerfile's header comments and the tests' docstrings
    df = (box / "docker" / "Dockerfile").read_text()
    head = [l for l in df.splitlines() if l.startswith("#")]
    check_commands(name, key, repo, runtime, version, tag("docker/Dockerfile"),
                   [l for l in head if re.search(r"docker (build|run)|^#\s+cd ", l)], problems)
    for f in sorted((box / "test").glob("*.py")):
        text = f.read_text(errors="replace")
        doc = re.match(r'\s*(?:#[^\n]*\n)*\s*[ruRU]?("""|\'\'\')(.*?)\1', text, re.S)
        if doc:
            lines = [l for l in doc.group(2).splitlines() if re.search(
                r"^\s*(\$ )?(cd |python3? |BOX_HOST=|docker )", l)]
            check_commands(name, key, repo, runtime, version,
                           tag(f"test/{f.name}"), lines, problems)

    # 5. the Dockerfile builds from this directory
    defaults = dict(re.findall(r"^ARG (\w+)=(\S+)", df, re.M))
    svc = re.findall(r"^ARG SERVICE_NAME=(\S+)", df, re.M)
    if svc and svc[0] != key:
        problems.append(f"{tag('docker/Dockerfile')}: default SERVICE_NAME={svc[0]} "
                        f"but the key is {key!r}")
    for full in re.findall(r"^COPY\s+(.+)$", df, re.M):
        if "--from" in full:                  # copies out of an earlier stage
            continue
        parts = [w for w in full.split() if not w.startswith("--")]
        line = " ".join(parts)
        if len(parts) < 2:
            continue
        for src in parts[:-1]:
            src = re.sub(r"\$\{(\w+)\}", lambda mm: defaults.get(mm.group(1), mm.group(0)), src)
            if "${" in src:
                continue
            if not glob.glob(str(box / src.rstrip("/"))):
                problems.append(f"{tag('docker/Dockerfile')}: COPY source {src!r} does not "
                                f"exist in the build context boxes/{name}/")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("boxes", nargs="*")
    args = ap.parse_args()
    root = repo_root()
    dirs = [d for d in box_dirs(root) if not args.boxes or d.name in args.boxes]
    if not args.boxes:
        dirs.append(root / "template_box")
    failed = 0
    for box in dirs:
        problems = []
        if box.name == "template_box":
            # the template is scaffolded with `template` substituted by the real name
            continue
        check_box(box, problems)
        problems = list(dict.fromkeys(problems))           # one line per finding
        if problems:
            failed += 1
            print(f"FAIL {box.name}")
            for p in problems:
                print(f"       {p}")
        else:
            print(f"ok   {box.name}")
    print()
    print(f"{failed} of {len(dirs) - (0 if args.boxes else 1)} box(es) have wrong instructions"
          if failed else "all instructions follow the registry protocol")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
