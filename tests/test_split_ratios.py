"""Regression tests for dataset partition ratios."""

import pytest

from adapterforge.data.format import split_records, split_records_by_doc


@pytest.mark.parametrize("splitter", [split_records, split_records_by_doc])
@pytest.mark.parametrize("ratios", [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
def test_split_respects_disabled_partitions(splitter, ratios):
    records = [{"id": i} for i in range(10)]
    splits = splitter(records, ratios=ratios)
    for name, ratio in zip(("train", "val", "test"), ratios, strict=True):
        assert len(splits[name]) == (10 if ratio else 0)


@pytest.mark.parametrize("splitter", [split_records, split_records_by_doc])
@pytest.mark.parametrize(
    "ratios",
    [(-0.1, 0.5, 0.6), (float("nan"), 0.05, 0.05), (0.9, 0.1), (0.9, 0.05, 0.05, 0.0)],
)
def test_split_rejects_invalid_ratios(splitter, ratios):
    with pytest.raises(ValueError, match="ratios"):
        splitter([{"id": i} for i in range(10)], ratios=ratios)


@pytest.mark.parametrize("splitter", [split_records, split_records_by_doc])
@pytest.mark.parametrize("count", [0, 1, 2, 3, 7])
@pytest.mark.parametrize("ratios", [(0.9, 0.05, 0.05), (0.0, 0.5, 0.5), (0.1, 0.8, 0.1)])
def test_small_partitions_preserve_every_record_once(splitter, count, ratios):
    records = [{"id": i, "meta": {"doc_id": f"doc-{i}"}} for i in range(count)]
    splits = splitter(records, ratios=ratios)
    assert sorted(r["id"] for part in splits.values() for r in part) == list(range(count))
    for name, ratio in zip(("train", "val", "test"), ratios, strict=True):
        if ratio == 0:
            assert splits[name] == []
