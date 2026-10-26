"""Package the frozen development data for an isolated Colab fold-1 run."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile


PROJECT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("FLOWER_DATA_ROOT", "flower_data"))
OUTPUT = PROJECT / "flower_colab_fold1_bundle.zip"
FOLDS = PROJECT / "artifacts/data/folds.jsonl"
MANIFEST = PROJECT / "artifacts/data/manifest.jsonl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    rows = [json.loads(line) for line in FOLDS.read_text(encoding="utf8").splitlines()]
    paths = sorted({row["path"] for row in rows})
    missing = [path for path in paths if not (DATA / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} frozen-fold images: {missing[:3]}")
    files = {
        "project/train.py": PROJECT / "train.py",
        "project/class_names.json": PROJECT / "class_names.json",
        "project/artifacts/data/folds.jsonl": FOLDS,
        "project/artifacts/data/manifest.jsonl": MANIFEST,
    }
    with ZipFile(OUTPUT, "w", compression=ZIP_STORED, allowZip64=True) as archive:
        for archive_name, source in files.items():
            archive.write(source, archive_name)
        for relative in paths:
            archive.write(DATA / relative, "flower_data/" + relative.replace("\\", "/"))
    print(json.dumps({
        "bundle": str(OUTPUT), "images": len(paths), "bytes": OUTPUT.stat().st_size,
        "sha256": sha256(OUTPUT), "folds_sha256": sha256(FOLDS),
        "manifest_sha256": sha256(MANIFEST),
    }, indent=2))


if __name__ == "__main__":
    main()
