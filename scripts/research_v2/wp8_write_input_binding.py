"""WP8 Phase C provenance materializer: aggregate EDGAR input binding.

One-shot use: compute the aggregate content binding for the SEC EDGAR
fundamentals shards and CIK mapping, then atomically write the tracked
``provenance/wp8/input_binding.json``.  This reads EDGAR/CIK input bytes for
SHA-256 hashing only; it never reads locked-holdout target labels, returns,
predictions or outcome rows.

Run:

    PYTHONPATH=. /opt/venv/bin/python scripts/research_v2/wp8_write_input_binding.py

Determinism: the binding contains no timestamps or wall-clock values, so
identical input files always produce byte-identical output.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.data.edgar_binding import build_edgar_input_binding  # noqa: E402
from src.research.ids import canonical_json  # noqa: E402

OUTPUT_REL = Path("provenance") / "wp8" / "input_binding.json"


def write_input_binding(root=None):
    """Compute and atomically persist the EDGAR input binding."""
    root = Path(root or ROOT)
    binding = build_edgar_input_binding(root)
    output = root / OUTPUT_REL
    output.parent.mkdir(parents=True, exist_ok=True)
    blob = canonical_json(binding) + "\n"

    handle_fd, tmp_path = tempfile.mkstemp(
        dir=str(output.parent), prefix=".tmp-input-binding-", suffix=".json"
    )
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            handle.write(blob)
        os.replace(tmp_path, output)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return output, binding


def main(argv=None):
    output, binding = write_input_binding()
    fundamentals = binding["edgar_fundamentals"]
    shards = fundamentals["shards"]
    print(
        "INPUT_BINDING_WRITTEN path=%s shard_count=%d bound_fundamental_rows=%d "
        "manifest_sha256=%s cik_sha256=%s"
        % (
            output,
            len(shards),
            int(sum(shard["row_count"] for shard in shards)),
            fundamentals["shard_manifest_sha256"],
            binding["edgar_cik_mapping"]["sha256"],
        )
    )
    return binding


if __name__ == "__main__":
    main(sys.argv[1:])
