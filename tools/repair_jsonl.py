"""Copy valid, unique JSONL records from a damaged append-only dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    valid: list[str] = []
    seen: set[str] = set()
    invalid = duplicates = 0
    for line in args.source.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            fen = str(record["fen"])
        except (json.JSONDecodeError, KeyError, TypeError):
            invalid += 1
            continue
        if fen in seen:
            duplicates += 1
            continue
        seen.add(fen)
        valid.append(json.dumps(record, separators=(",", ":"), sort_keys=True))

    payload = "\n".join(valid) + "\n"
    args.destination.write_text(payload, encoding="utf-8")
    manifest = {
        "format": 1,
        "generator": "tools/repair_jsonl.py",
        "source": args.source.name,
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "valid": len(valid),
        "invalid": invalid,
        "duplicates": duplicates,
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }
    args.destination.with_suffix(args.destination.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"valid={len(valid)} invalid={invalid} duplicates={duplicates}")


if __name__ == "__main__":
    main()
