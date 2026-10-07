"""WP8 EDGAR input-binding tests (synthetic fixtures only).

These tests verify the aggregate SHA-256 binding used to make the one-shot
holdout features deterministic.  Fixtures are created entirely in temporary
directories and never touch real research data, locked-holdout labels, returns,
predictions or 2022+ outcomes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.research.data.edgar_binding import (
    EdgarBindingError,
    build_edgar_input_binding,
    load_and_verify_cik_by_ticker,
    load_and_verify_edgar_fundamentals,
    verify_edgar_input_binding,
)

COLUMNS = [
    "cik",
    "field",
    "value",
    "fiscal_period_start",
    "fiscal_period_end",
    "accession",
    "available_at",
    "form",
]


def _fundamentals_base(root):
    return root / "data" / "research_v2" / "silver" / "edgar_fundamentals"


def _cik_path(root):
    return root / "artifacts" / "research" / "wp4" / "edgar_cik_mapping.json"


def _write_shard(root, version, rows):
    shard_dir = _fundamentals_base(root) / version
    shard_dir.mkdir(parents=True)
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame.to_parquet(shard_dir / "data.parquet", index=False)
    provenance = {
        "columns": COLUMNS,
        "row_count": int(len(frame)),
        "version": version,
    }
    (shard_dir / "provenance.json").write_text(
        json.dumps(provenance, sort_keys=True), encoding="utf-8"
    )
    return frame


def _write_cik(root, payload=None):
    path = _cik_path(root)
    path.parent.mkdir(parents=True)
    payload = payload if payload is not None else {
        "mapping": {
            "AAPL": {"cik": 1, "match": "EXACT"},
            "BRK": {"cik": 2, "match": "NOT_EXACT"},
        },
        "unmapped": 0,
        "ambiguous": 0,
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _base_fixture(root):
    _write_shard(
        root,
        "aaaaaaaaaaaaaaaa",
        [
            {
                "cik": "0000000001",
                "field": "eps",
                "value": 1.5,
                "fiscal_period_start": "2020-01-01",
                "fiscal_period_end": "2020-12-31",
                "accession": "acc-shared",
                "available_at": "2021-01-01",
                "form": "10-K",
            },
        ],
    )
    _write_shard(
        root,
        "bbbbbbbbbbbbbbbb",
        [
            {
                "cik": "0000000002",
                "field": "revenue",
                "value": 100.0,
                "fiscal_period_start": "2020-01-01",
                "fiscal_period_end": "2020-12-31",
                "accession": "acc-2",
                "available_at": "2021-01-01",
                "form": "10-K",
            },
            # Duplicate of the first shard on the deduplication key.
            {
                "cik": "0000000001",
                "field": "eps",
                "value": 1.5,
                "fiscal_period_start": "2020-01-01",
                "fiscal_period_end": "2020-12-31",
                "accession": "acc-shared",
                "available_at": "2021-01-02",
                "form": "10-K",
            },
        ],
    )
    _write_cik(root)


def test_build_verify_and_load_edgar_binding(tmp_path):
    _base_fixture(tmp_path)
    binding = build_edgar_input_binding(tmp_path)

    assert binding["schema_version"] == "edgar_input_binding_v1"
    assert binding["edgar_fundamentals"]["shard_manifest_sha256"] == hashlib.sha256(
        json.dumps(
            binding["edgar_fundamentals"]["shards"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    assert binding["edgar_cik_mapping"]["path"] == "artifacts/research/wp4/edgar_cik_mapping.json"
    assert "generated_at" not in binding
    assert verify_edgar_input_binding(tmp_path, binding) is True

    frame = load_and_verify_edgar_fundamentals(tmp_path, binding)
    assert len(frame) == 2
    assert set(frame["cik"]) == {"0000000001", "0000000002"}

    by_ticker, payload = load_and_verify_cik_by_ticker(tmp_path, binding)
    assert by_ticker == {"AAPL": "0000000001"}
    assert payload["unmapped"] == 0


def test_build_edgar_binding_is_deterministic(tmp_path):
    _base_fixture(tmp_path)
    first = build_edgar_input_binding(tmp_path)
    second = build_edgar_input_binding(tmp_path)
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_corrupt_fundamentals_parquet_raises(tmp_path):
    _base_fixture(tmp_path)
    binding = build_edgar_input_binding(tmp_path)

    shard = binding["edgar_fundamentals"]["shards"][0]
    data_path = _fundamentals_base(tmp_path) / shard["version"] / "data.parquet"
    data_path.write_bytes(b"corrupt" + data_path.read_bytes())

    with pytest.raises(EdgarBindingError) as recorded:
        verify_edgar_input_binding(tmp_path, binding)
    assert "data.parquet" in str(recorded.value)
    assert "sha256 mismatch" in str(recorded.value)


def test_missing_shard_raises(tmp_path):
    _base_fixture(tmp_path)
    binding = build_edgar_input_binding(tmp_path)
    shard = binding["edgar_fundamentals"]["shards"][0]
    (_fundamentals_base(tmp_path) / shard["version"] / "data.parquet").unlink()

    with pytest.raises(EdgarBindingError) as recorded:
        verify_edgar_input_binding(tmp_path, binding)
    assert "missing shard" in str(recorded.value)


def test_extra_shard_raises(tmp_path):
    _base_fixture(tmp_path)
    binding = build_edgar_input_binding(tmp_path)
    _write_shard(
        tmp_path,
        "cccccccccccccccc",
        [{
            "cik": "0000000003",
            "field": "eps",
            "value": 2.0,
            "fiscal_period_start": "2020-01-01",
            "fiscal_period_end": "2020-12-31",
            "accession": "acc-3",
            "available_at": "2021-01-01",
            "form": "10-K",
        }],
    )

    with pytest.raises(EdgarBindingError) as recorded:
        verify_edgar_input_binding(tmp_path, binding)
    assert "extra shard" in str(recorded.value)


def test_corrupt_cik_mapping_raises(tmp_path):
    _base_fixture(tmp_path)
    binding = build_edgar_input_binding(tmp_path)
    _cik_path(tmp_path).write_text(
        json.dumps({"mapping": {}, "unmapped": 1, "ambiguous": 0}), encoding="utf-8"
    )

    with pytest.raises(EdgarBindingError) as recorded:
        verify_edgar_input_binding(tmp_path, binding)
    assert "CIK mapping sha256 mismatch" in str(recorded.value)


def test_missing_provenance_sidecar_raises(tmp_path):
    _base_fixture(tmp_path)
    (tmp_path / "data" / "research_v2" / "silver" / "edgar_fundamentals" /
     "aaaaaaaaaaaaaaaa" / "provenance.json").unlink()

    with pytest.raises(EdgarBindingError) as recorded:
        build_edgar_input_binding(tmp_path)
    assert "no provenance.json sidecar" in str(recorded.value)
