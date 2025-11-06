"""
Transcription Quality Metrics for HTR Evaluation

Provides CER, WER, and character-level diff operations for comparing
transcriptions. Used by the comparison widget for engine evaluation.

Author: Claude Code
Date: 2025-11-05
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import Levenshtein


@dataclass
class DiffOperation:
    """
    Single character-level diff operation.

    Used for visualizing differences between reference and hypothesis.
    """
    operation: str  # 'equal', 'replace', 'insert', 'delete'
    ref_char: str   # Character from reference (empty for insertions)
    hyp_char: str   # Character from hypothesis (empty for deletions)
    ref_pos: int    # Position in reference string
    hyp_pos: int    # Position in hypothesis string


@dataclass
class LineMetrics:
    """
    Complete metrics for comparing a single line.

    Attributes:
        reference: Ground truth or baseline text
        hypothesis: Predicted text to compare against reference
        cer: Character Error Rate (0-100)
        wer: Word Error Rate (0-100)
        match_percent: Percentage of matching characters (0-100)
        edit_distance: Levenshtein edit distance
        diff_ops: List of character-level diff operations
    """
    reference: str
    hypothesis: str
    cer: float
    wer: float
    match_percent: float
    edit_distance: int
    diff_ops: List[DiffOperation]


class TranscriptionMetrics:
    """
    Calculate HTR quality metrics.

    Uses python-Levenshtein for fast edit distance calculation.
    All methods are static and can be called without instantiation.

    Example:
        >>> cer = TranscriptionMetrics.calculate_cer("hello", "helo")
        >>> print(f"CER: {cer:.2f}%")
        CER: 20.00%
    """

    @staticmethod
    def calculate_cer(reference: str, hypothesis: str) -> float:
        """
        Calculate Character Error Rate using Levenshtein distance.

        CER = (insertions + deletions + substitutions) / total_characters

        Args:
            reference: Ground truth text
            hypothesis: Predicted text

        Returns:
            CER as percentage (0.0-100.0)

        Examples:
            >>> TranscriptionMetrics.calculate_cer("test", "test")
            0.0
            >>> TranscriptionMetrics.calculate_cer("test", "text")
            25.0
        """
        # Handle empty strings
        if not reference:
            return 100.0 if hypothesis else 0.0

        distance = Levenshtein.distance(reference, hypothesis)
        return (distance / len(reference)) * 100.0

    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate using Levenshtein distance.

        WER = (insertions + deletions + substitutions) / total_words

        Words are split by whitespace. Typically 3-4x higher than CER
        for natural language text.

        Args:
            reference: Ground truth text
            hypothesis: Predicted text

        Returns:
            WER as percentage (0.0-100.0)

        Examples:
            >>> TranscriptionMetrics.calculate_wer("hello world", "hello earth")
            50.0
        """
        # Split into words
        ref_words = reference.split()
        hyp_words = hypothesis.split()

        # Handle empty word lists
        if not ref_words:
            return 100.0 if hyp_words else 0.0

        # Calculate edit distance between word sequences
        distance = Levenshtein.distance(ref_words, hyp_words)
        return (distance / len(ref_words)) * 100.0

    @staticmethod
    def calculate_match_percent(reference: str, hypothesis: str) -> float:
        """
        Calculate match percentage (inverse of normalized edit distance).

        This is more intuitive than CER for users: higher = better.

        Match% = (max_length - edit_distance) / max_length * 100

        Args:
            reference: Ground truth text
            hypothesis: Predicted text

        Returns:
            Match percentage (0.0-100.0)

        Examples:
            >>> TranscriptionMetrics.calculate_match_percent("test", "test")
            100.0
            >>> TranscriptionMetrics.calculate_match_percent("test", "text")
            75.0
        """
        max_len = max(len(reference), len(hypothesis))

        # Both empty = perfect match
        if max_len == 0:
            return 100.0

        distance = Levenshtein.distance(reference, hypothesis)
        return ((max_len - distance) / max_len) * 100.0

    @staticmethod
    def get_diff_operations(reference: str, hypothesis: str) -> List[DiffOperation]:
        """
        Get character-level diff operations for visualization.

        Uses Levenshtein edit operations to create a list of differences
        between reference and hypothesis. This is used for color-coded
        diff display in the GUI.

        Operation types:
        - 'equal': Characters match
        - 'replace': Character substitution
        - 'insert': Character added in hypothesis
        - 'delete': Character removed from hypothesis

        Args:
            reference: Ground truth text
            hypothesis: Predicted text

        Returns:
            List of DiffOperation objects

        Examples:
            >>> ops = TranscriptionMetrics.get_diff_operations("cat", "cut")
            >>> ops[1].operation
            'replace'
            >>> ops[1].ref_char
            'a'
            >>> ops[1].hyp_char
            'u'
        """
        ops = []

        # Get edit operations from Levenshtein
        # Returns list of (operation, ref_pos, hyp_pos)
        editops = Levenshtein.editops(reference, hypothesis)

        # Track positions in both strings
        ref_idx = 0
        hyp_idx = 0

        for op_type, ref_pos, hyp_pos in editops:
            # Add any matching characters before this operation
            while ref_idx < ref_pos and hyp_idx < hyp_pos:
                ops.append(DiffOperation(
                    operation='equal',
                    ref_char=reference[ref_idx],
                    hyp_char=hypothesis[hyp_idx],
                    ref_pos=ref_idx,
                    hyp_pos=hyp_idx
                ))
                ref_idx += 1
                hyp_idx += 1

            # Add the edit operation
            if op_type == 'replace':
                ops.append(DiffOperation(
                    operation='replace',
                    ref_char=reference[ref_pos],
                    hyp_char=hypothesis[hyp_pos],
                    ref_pos=ref_pos,
                    hyp_pos=hyp_pos
                ))
                ref_idx = ref_pos + 1
                hyp_idx = hyp_pos + 1

            elif op_type == 'delete':
                ops.append(DiffOperation(
                    operation='delete',
                    ref_char=reference[ref_pos],
                    hyp_char='',
                    ref_pos=ref_pos,
                    hyp_pos=hyp_pos
                ))
                ref_idx = ref_pos + 1
                # hyp_idx stays the same

            elif op_type == 'insert':
                ops.append(DiffOperation(
                    operation='insert',
                    ref_char='',
                    hyp_char=hypothesis[hyp_pos],
                    ref_pos=ref_pos,
                    hyp_pos=hyp_pos
                ))
                hyp_idx = hyp_pos + 1
                # ref_idx stays the same

        # Add any remaining matching characters
        while ref_idx < len(reference) and hyp_idx < len(hypothesis):
            ops.append(DiffOperation(
                operation='equal',
                ref_char=reference[ref_idx],
                hyp_char=hypothesis[hyp_idx],
                ref_pos=ref_idx,
                hyp_pos=hyp_idx
            ))
            ref_idx += 1
            hyp_idx += 1

        return ops

    @staticmethod
    def compare_lines(reference: str, hypothesis: str) -> LineMetrics:
        """
        Perform complete comparison of two lines.

        This is the main entry point for line comparison. It calculates
        all metrics and generates diff operations in a single call.

        Args:
            reference: Ground truth or baseline transcription
            hypothesis: Engine output to compare

        Returns:
            LineMetrics object with all metrics and diff operations

        Examples:
            >>> metrics = TranscriptionMetrics.compare_lines("hello", "helo")
            >>> print(f"CER: {metrics.cer:.2f}%, WER: {metrics.wer:.2f}%")
            CER: 20.00%, WER: 0.00%
        """
        # Calculate all metrics
        cer = TranscriptionMetrics.calculate_cer(reference, hypothesis)
        wer = TranscriptionMetrics.calculate_wer(reference, hypothesis)
        match = TranscriptionMetrics.calculate_match_percent(reference, hypothesis)
        distance = Levenshtein.distance(reference, hypothesis)
        diff_ops = TranscriptionMetrics.get_diff_operations(reference, hypothesis)

        return LineMetrics(
            reference=reference,
            hypothesis=hypothesis,
            cer=cer,
            wer=wer,
            match_percent=match,
            edit_distance=distance,
            diff_ops=diff_ops
        )

    @staticmethod
    def calculate_overall_metrics(
        references: List[str],
        hypotheses: List[str]
    ) -> Tuple[float, float, float]:
        """
        Calculate overall metrics for multiple lines.

        Args:
            references: List of ground truth texts
            hypotheses: List of predicted texts (same length as references)

        Returns:
            Tuple of (average_cer, average_wer, average_match)

        Raises:
            ValueError: If lengths don't match

        Examples:
            >>> refs = ["hello", "world"]
            >>> hyps = ["helo", "world"]
            >>> cer, wer, match = TranscriptionMetrics.calculate_overall_metrics(refs, hyps)
            >>> print(f"Overall CER: {cer:.2f}%")
            Overall CER: 10.00%
        """
        if len(references) != len(hypotheses):
            raise ValueError(
                f"Reference and hypothesis lists must have same length "
                f"(got {len(references)} vs {len(hypotheses)})"
            )

        if not references:
            return 0.0, 0.0, 100.0

        total_cer = 0.0
        total_wer = 0.0
        total_match = 0.0

        for ref, hyp in zip(references, hypotheses):
            total_cer += TranscriptionMetrics.calculate_cer(ref, hyp)
            total_wer += TranscriptionMetrics.calculate_wer(ref, hyp)
            total_match += TranscriptionMetrics.calculate_match_percent(ref, hyp)

        n = len(references)
        return (total_cer / n, total_wer / n, total_match / n)


# Example usage
if __name__ == "__main__":
    # Test with some examples
    print("=" * 70)
    print("TRANSCRIPTION METRICS - Examples")
    print("=" * 70)
    print()

    # Example 1: Exact match
    ref1 = "hello world"
    hyp1 = "hello world"
    metrics1 = TranscriptionMetrics.compare_lines(ref1, hyp1)
    print(f"Example 1: Exact match")
    print(f"  Reference:  '{ref1}'")
    print(f"  Hypothesis: '{hyp1}'")
    print(f"  CER: {metrics1.cer:.2f}%")
    print(f"  WER: {metrics1.wer:.2f}%")
    print(f"  Match: {metrics1.match_percent:.2f}%")
    print()

    # Example 2: Single character error
    ref2 = "test"
    hyp2 = "text"
    metrics2 = TranscriptionMetrics.compare_lines(ref2, hyp2)
    print(f"Example 2: Single substitution")
    print(f"  Reference:  '{ref2}'")
    print(f"  Hypothesis: '{hyp2}'")
    print(f"  CER: {metrics2.cer:.2f}%")
    print(f"  WER: {metrics2.wer:.2f}%")
    print(f"  Match: {metrics2.match_percent:.2f}%")
    print(f"  Diff operations:")
    for op in metrics2.diff_ops:
        if op.operation != 'equal':
            print(f"    {op.operation}: '{op.ref_char}' -> '{op.hyp_char}' "
                  f"at ref_pos={op.ref_pos}, hyp_pos={op.hyp_pos}")
    print()

    # Example 3: Cyrillic text (Church Slavonic)
    ref3 = "и идѣше поутемь"
    hyp3 = "и идѣше поутемь"
    metrics3 = TranscriptionMetrics.compare_lines(ref3, hyp3)
    print(f"Example 3: Cyrillic text (exact match)")
    print(f"  Reference:  '{ref3}'")
    print(f"  Hypothesis: '{hyp3}'")
    print(f"  CER: {metrics3.cer:.2f}%")
    print(f"  WER: {metrics3.wer:.2f}%")
    print()

    # Example 4: Cyrillic with error
    ref4 = "гредоущоу же ѥмоу"
    hyp4 = "гредоущом же ѥмоу"
    metrics4 = TranscriptionMetrics.compare_lines(ref4, hyp4)
    print(f"Example 4: Cyrillic text (one character error)")
    print(f"  Reference:  '{ref4}'")
    print(f"  Hypothesis: '{hyp4}'")
    print(f"  CER: {metrics4.cer:.2f}%")
    print(f"  WER: {metrics4.wer:.2f}%")
    print(f"  Match: {metrics4.match_percent:.2f}%")
    print()

    # Example 5: Overall metrics
    refs = [ref1, ref2, ref3, ref4]
    hyps = [hyp1, hyp2, hyp3, hyp4]
    avg_cer, avg_wer, avg_match = TranscriptionMetrics.calculate_overall_metrics(refs, hyps)
    print(f"Example 5: Overall metrics for {len(refs)} lines")
    print(f"  Average CER: {avg_cer:.2f}%")
    print(f"  Average WER: {avg_wer:.2f}%")
    print(f"  Average Match: {avg_match:.2f}%")
    print()

    print("=" * 70)
