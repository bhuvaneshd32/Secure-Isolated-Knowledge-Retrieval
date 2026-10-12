"""The full-evaluation script runs end to end and its CSVs contain the expected facts."""

import csv
import shutil
import sys
from pathlib import Path

import pytest

from evaluation import run_all

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def public_kb(tmp_path):
    """A temp copy of the public KB, so the embedding cache is not written into data/."""
    dst = tmp_path / "public_kb.jsonl"
    shutil.copy(ROOT / "data" / "public_kb.jsonl", dst)
    return str(dst)


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_quick_run_writes_consistent_csvs(tmp_path, monkeypatch, public_kb):
    out = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_all", "--skip-llm", "--skip-perf", "--public-kb", public_kb, "--out", str(out)],
    )
    run_all.main()

    for name in ("leakage", "qa", "retrieval", "key_exposure", "tamper", "summary"):
        assert (out / f"{name}.csv").exists()
    assert (out / "run_info.json").exists()

    summary = {r["system"]: r for r in _rows(out / "summary.csv")}
    assert int(summary["naive_shared"]["foreign_secrets_in_prompt"]) > 0  # the attack is real
    for system in ("plaintext", "method_a", "method_b", "method_b_improved"):
        assert int(summary[system]["foreign_secrets_in_prompt"]) == 0
        assert float(summary[system]["score_parity_with_plaintext"]) == 1.0

    # blast radius of one leaked key: A everything, B only part of the chain
    assert float(summary["method_a"]["key_leak_mean_fraction_readable"]) == 1.0
    assert float(summary["method_b_improved"]["key_leak_mean_fraction_readable"]) < 0.6

    outcomes = {(r["system"], r["attack"]): r["outcome"] for r in _rows(out / "tamper.csv")}
    assert outcomes[("method_a", "flip_iv")] == "detected"
    assert outcomes[("method_b_improved", "flip_iv")] == "detected"
    assert outcomes[("method_b", "flip_iv")] == "silent_modification"
    assert outcomes[("plaintext", "flip_ciphertext")] == "silent_modification"


def test_resume_does_not_duplicate_rows(tmp_path, monkeypatch, public_kb):
    out = tmp_path / "run"
    argv = ["run_all", "--skip-llm", "--skip-perf", "--public-kb", public_kb, "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    run_all.main()
    n_leak = len(_rows(out / "leakage.csv"))
    n_qa = len(_rows(out / "qa.csv"))
    monkeypatch.setattr(sys, "argv", [*argv[:-2], "--resume", str(out)])
    run_all.main()
    assert len(_rows(out / "leakage.csv")) == n_leak
    assert len(_rows(out / "qa.csv")) == n_qa


class _FlakyLLM:
    """Fails `fail_times` times with an LLMError, then answers."""

    def __init__(self, fail_times):
        self.fail_times, self.n = fail_times, 0

    def generate(self, prompt):
        self.n += 1
        if self.n <= self.fail_times:
            raise run_all.LLMError("simulated ReadTimeout")
        return "ok"


def test_llm_call_is_retried_then_succeeds():
    caller = run_all.LlmCaller(_FlakyLLM(fail_times=2), reuse=False, retries=2)
    ans, _, _, err = caller.ask("p")
    assert (ans, err) == ("ok", "") and caller.errors == 0


def test_llm_call_that_keeps_failing_is_recorded_not_fatal():
    caller = run_all.LlmCaller(_FlakyLLM(fail_times=99), reuse=False, retries=1)
    ans, _, _, err = caller.ask("p")
    assert ans == "" and "ReadTimeout" in err and caller.errors == 1
