import pytest

from transcription_metrics import (
    ComparisonMode,
    TranscriptionMetrics,
)


def test_gt_labels_keep_cer_and_wer():
    labels = TranscriptionMetrics.get_display_labels(ComparisonMode.GROUND_TRUTH)

    assert labels.char_rate == "CER"
    assert labels.word_rate == "WER"
    assert "standard HTR metric" in labels.micro_char_rate_note


def test_engine_comparison_labels_use_disagreement_terminology():
    labels = TranscriptionMetrics.get_display_labels(ComparisonMode.ENGINE_COMPARISON)

    assert labels.char_rate == "Char disagreement"
    assert labels.word_rate == "Word disagreement"
    assert "GT-free" in labels.micro_char_rate_note


def test_engine_comparison_display_metrics_are_symmetric_and_bounded():
    left = TranscriptionMetrics.compare_lines("alpha beta", "alpha gamma delta")
    right = TranscriptionMetrics.compare_lines("alpha gamma delta", "alpha beta")

    left_display = TranscriptionMetrics.get_display_metrics(left, ComparisonMode.ENGINE_COMPARISON)
    right_display = TranscriptionMetrics.get_display_metrics(right, ComparisonMode.ENGINE_COMPARISON)

    assert left_display.char_rate == pytest.approx(right_display.char_rate)
    assert left_display.word_rate == pytest.approx(right_display.word_rate)
    assert 0.0 <= left_display.char_rate <= 100.0
    assert 0.0 <= left_display.word_rate <= 100.0


def test_engine_comparison_summary_is_symmetric():
    refs = ["abc", "alpha beta"]
    hyps = ["abcd", "alpha gamma delta"]

    forward = TranscriptionMetrics.calculate_summary_metrics(
        refs,
        hyps,
        ComparisonMode.ENGINE_COMPARISON,
    )
    backward = TranscriptionMetrics.calculate_summary_metrics(
        hyps,
        refs,
        ComparisonMode.ENGINE_COMPARISON,
    )

    assert forward.micro_char_rate == pytest.approx(backward.micro_char_rate)
    assert forward.macro_word_rate == pytest.approx(backward.macro_word_rate)
