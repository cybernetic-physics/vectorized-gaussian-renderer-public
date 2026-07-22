from scripts.verify_publication_code_freeze import publication_path_allowed


def test_publication_code_freeze_allows_only_declared_prose_and_results() -> None:
    assert publication_path_allowed("post.md")
    assert publication_path_allowed("publication/claim-ledger.json")
    assert publication_path_allowed(
        "experiments/flashgs_matched/results/final-table.md"
    )


def test_publication_code_freeze_rejects_load_bearing_changes() -> None:
    assert not publication_path_allowed("benchmarks/run_flashgs_matched.py")
    assert not publication_path_allowed("src/renderer.cu")
    assert not publication_path_allowed(
        "experiments/flashgs_matched/BENCHMARK_CONTRACT.md"
    )
