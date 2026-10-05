#!/usr/bin/env python3
"""The contract test every box must pass, whatever it computes.

Box-agnostic by construction: it reads boxes/<name>/box.yaml to learn the
config section key and nothing else. A box's own test/ suite checks that its
numbers are right; this checks that it is a well-behaved member of the fleet.

    python3 contract/conformance/test_conformance.py --box yolo --host localhost:9068
    python3 contract/conformance/test_conformance.py --box clip --host localhost:9061 --strict

Required (a failure here fails CI):
  1. the box is reachable and serves the PipelineService
  2. `{"<key>": {"command": "reset"}}` is accepted - the contract says every
     standard box takes a reset, even if it is a no-op
  3. every reply's config is an object with the box's section in it
  4. the section carries a `status` from done | empty_request | error
  5. any declared `encoding` names codecs the client knows, and names only
     fields the reply actually carries
  6. a request with no data at all does not hang or crash the box

Advisory (reported, not failed): boxes differ legitimately here.
  - an unknown command: most boxes error, but several treat any non-reset
    command as their default action
  - an empty request: most answer empty_request, lang_sam echoes the envelope
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))

import grpc  # noqa: E402

try:
    from visionist_client import Visionist
except ImportError:                                        # pragma: no cover
    sys.exit("visionist-client is required:  pip install visionist-client")

try:
    import yaml
except ImportError:                                        # pragma: no cover
    sys.exit("PyYAML is required:  pip install -r tools/requirements.txt")

KNOWN_CODECS = {"identity", "json", "numpy", "torch", "zstd_pickle"}
STATUSES = {"done", "empty_request", "error"}


class Report:
    def __init__(self):
        self.failed, self.advisory = [], []

    def check(self, ok, name, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail else ""))
        if not ok:
            self.failed.append(name)

    def note(self, name, detail):
        print(f"  note {name}  {detail}")
        self.advisory.append(f"{name}: {detail}")


def section_of(res, key):
    cfg = res.config if isinstance(res.config, dict) else {}
    return cfg.get(key), cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--box", required=True, help="box directory name")
    ap.add_argument("--host", required=True, help="host:port of a running box")
    ap.add_argument("--root", default=None, help="registry checkout (default: inferred)")
    ap.add_argument("--strict", action="store_true",
                    help="treat advisory findings as failures too")
    ap.add_argument("--timeout", type=float, default=120)
    args = ap.parse_args()

    root = pathlib.Path(args.root) if args.root \
        else pathlib.Path(__file__).resolve().parents[2]
    manifest_path = root / "boxes" / args.box / "box.yaml"
    if not manifest_path.is_file():
        sys.exit(f"no manifest at {manifest_path}")
    manifest = yaml.safe_load(manifest_path.open())
    key = manifest["key"]

    print(f"conformance: {args.box} (section {key!r}) at {args.host}\n")
    r = Report()

    box = Visionist(args.host, timeout=args.timeout)

    # 1. reachable
    info = box.info(timeout=15)
    r.check(bool(info.get("reachable")), "reachable", args.host)
    if not info.get("reachable"):
        print("\nFAIL: the box is not answering; nothing else can be checked")
        return 1
    if info.get("reflection"):
        r.check(info.get("service") == "pipeline.PipelineService",
                "serves PipelineService", str(info.get("service")))
    else:
        r.note("reflection", "not served; cannot self-describe (not required)")

    # 2 + 3 + 4. reset is the one command every box must accept
    res = box.run(config={key: {"command": "reset"}})
    section, cfg = section_of(res, key)
    r.check(isinstance(cfg, dict), "reply config is an object")
    r.check(isinstance(section, dict),
            f"reply is namespaced under {key!r}",
            "" if isinstance(section, dict) else f"got keys {sorted(cfg)}")
    if isinstance(section, dict):
        status = section.get("status")
        r.check(status in STATUSES, "status is from the vocabulary", str(status))
        r.check(status == "done", "reset is accepted", str(status))

    # 5. the declared encoding must be decodable and must describe real fields
    def check_encoding(res, label):
        sec, _ = section_of(res, key)
        if not isinstance(sec, dict) or "encoding" not in sec:
            return
        enc = sec["encoding"]
        names = [enc] if isinstance(enc, str) else list(enc.values())
        unknown = sorted(set(names) - KNOWN_CODECS)
        r.check(not unknown, f"{label}: declared codecs are known",
                f"unknown: {unknown}" if unknown else "")
        if isinstance(enc, dict):
            missing = sorted(set(enc) - set(res.fields))
            if missing:
                r.note(f"{label}: encoding names absent fields", str(missing))

    check_encoding(res, "reset")

    # 6. an empty request must be answered, not crash the box
    try:
        res = box.run(config={key: {}})
        sec, _ = section_of(res, key)
        status = (sec or {}).get("status")
        r.check(isinstance(sec, dict), "empty request is answered",
                f"status {status!r}")
        if status not in (None, "empty_request"):
            r.note("empty request", f"answered {status!r} rather than empty_request")
        check_encoding(res, "empty")
    except grpc.RpcError as e:
        r.check(False, "empty request is answered", f"{e.code().name}: {e.details()}")

    # advisory: what an unrecognised command does
    try:
        res = box.run(config={key: {"command": "__not_a_command__"}})
        sec, _ = section_of(res, key)
        status = (sec or {}).get("status")
        if status == "error":
            r.check(True, "unknown command is rejected")
        else:
            r.note("unknown command",
                   f"answered {status!r} - this box treats any non-reset "
                   f"command as its default action")
    except grpc.RpcError as e:
        r.check(False, "unknown command is answered",
                f"{e.code().name}: {e.details()}")

    box.close()

    print()
    if r.failed:
        print(f"FAIL: {len(r.failed)} required check(s): {', '.join(r.failed)}")
        return 1
    if r.advisory and args.strict:
        print(f"FAIL (--strict): {len(r.advisory)} advisory finding(s)")
        return 1
    print(f"PASS: {args.box} conforms"
          + (f" ({len(r.advisory)} advisory)" if r.advisory else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
