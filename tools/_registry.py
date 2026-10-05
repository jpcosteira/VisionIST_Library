"""Shared helpers: find the repo, load box manifests, resolve image references.

Every other tool builds on this. Nothing here knows about any particular box -
the knowledge is in boxes/<name>/box.yaml.
"""

from __future__ import annotations

import pathlib
import sys

try:
    import yaml
except ImportError:                                        # pragma: no cover
    sys.exit("PyYAML is required:  pip install -r tools/requirements.txt")

#: Registry host per short name. A box names a repository; the caller picks
#: where to pull it from, so the same fleet definition works off either.
REGISTRY_HOSTS = {
    "ghcr": "ghcr.io",
    "dockerhub": "docker.io",
}

#: Committed test fixtures above this are rejected: the repo is cloned by
#: everyone, forever, including the history. Bigger files belong in `assets:`.
MAX_FIXTURE_BYTES = 256 * 1024


def repo_root(start: pathlib.Path | None = None) -> pathlib.Path:
    """The VisionIST_Library checkout containing this file."""
    here = (start or pathlib.Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "boxes").is_dir() and (parent / "contract").is_dir():
            return parent
    raise SystemExit("not inside a VisionIST_Library checkout "
                     "(no boxes/ + contract/ above this file)")


def box_dirs(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p for p in (root / "boxes").iterdir()
                  if p.is_dir() and (p / "box.yaml").is_file())


def load_manifest(box_dir: pathlib.Path) -> dict:
    with (box_dir / "box.yaml").open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{box_dir.name}/box.yaml is not a mapping")
    data["_dir"] = box_dir.name
    return data


def load_all(root: pathlib.Path) -> list[dict]:
    return [load_manifest(d) for d in box_dirs(root)]


def image_ref(manifest: dict, registry: str = "dockerhub",
              version: str | None = None) -> str:
    """Full pullable reference for a box on one registry.

    ``ghcr`` keeps the repository path as given; ``dockerhub`` flattens it,
    because Docker Hub has exactly one level of namespace - sipgisr/visionist-clip
    becomes sipgisr/visionist-clip.
    """
    if registry not in REGISTRY_HOSTS:
        raise ValueError(f"unknown registry {registry!r} "
                         f"(known: {', '.join(sorted(REGISTRY_HOSTS))})")
    repo = manifest["image"]["repository"]
    tag = version or manifest["version"]
    if registry == "dockerhub":
        org, _, name = repo.rpartition("/")
        repo = f"{org.replace('-', '').replace('/', '')}/{name}" if org else name
    return f"{REGISTRY_HOSTS[registry]}/{repo}:{tag}"


def registries_of(manifest: dict) -> list[str]:
    return manifest.get("image", {}).get("registries") or ["dockerhub"]
