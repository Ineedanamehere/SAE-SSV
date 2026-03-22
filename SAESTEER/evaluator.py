"""
Evaluation metrics for SAE steering experiments.

This module provides:
- Lexical diversity metrics (MTLD, Entropy)
- LLM-as-Judge evaluation framework
"""

import re
import json
import random
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from collections import Counter
from typing import List, Dict, Tuple, Optional, Callable


# ============================================================
# Lexical Diversity Metrics
# ============================================================

def compute_mtld(text: str, ttr_threshold: float = 0.72) -> float:
    """
    Compute MTLD (Measure of Textual Lexical Diversity).

    MTLD measures vocabulary richness based on the average length of text
    segments with stable type-token ratio (TTR).

    Args:
        text: Input text to analyze
        ttr_threshold: TTR threshold for factor counting (default: 0.72)

    Returns:
        MTLD score (higher = more lexically diverse)
    """
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if len(words) < 10:
        return 0.0

    def mtld_forward(word_list):
        factors = 0
        start = 0

        for i in range(1, len(word_list) + 1):
            segment = word_list[start:i]
            ttr = len(set(segment)) / len(segment)

            if ttr <= ttr_threshold:
                factors += 1
                start = i

        # Handle remaining partial factor
        if start < len(word_list):
            segment = word_list[start:]
            ttr = len(set(segment)) / len(segment)
            partial = (1.0 - ttr) / (1.0 - ttr_threshold) if ttr_threshold < 1.0 else 0
            factors += partial

        return len(word_list) / factors if factors > 0 else len(word_list)

    # Compute MTLD as average of forward and backward passes
    forward_mtld = mtld_forward(words)
    backward_mtld = mtld_forward(words[::-1])

    return (forward_mtld + backward_mtld) / 2


def compute_token_entropy(text: str, tokenizer=None) -> float:
    """
    Compute Shannon entropy of token distribution.

    H = -Σ p(x_i) * log2(p(x_i))

    Args:
        text: Input text to analyze
        tokenizer: Optional HuggingFace tokenizer. Falls back to word-level if None.

    Returns:
        Entropy score (higher = more unpredictable/diverse token usage)
    """
    if tokenizer is not None:
        tokens = tokenizer.encode(text, add_special_tokens=False)
    else:
        tokens = re.findall(r'\b\w+\b', text.lower())

    if len(tokens) < 2:
        return 0.0

    token_counts = Counter(tokens)
    total = sum(token_counts.values())
    probabilities = [count / total for count in token_counts.values()]

    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    return entropy


def compute_diversity_metrics(
    steered_texts: List[str],
    baseline_texts: List[str],
    tokenizer=None
) -> Tuple[Dict, Dict]:
    """
    Compute ΔMTLD and ΔEntropy between steered and baseline outputs.

    Args:
        steered_texts: List of steered model outputs
        baseline_texts: List of baseline model outputs
        tokenizer: Optional tokenizer for entropy calculation

    Returns:
        Tuple of (per_sample_results, summary_stats)
    """
    results = {
        'steered_mtld': [],
        'baseline_mtld': [],
        'steered_entropy': [],
        'baseline_entropy': [],
        'delta_mtld': [],
        'delta_entropy': []
    }

    for steered, baseline in zip(steered_texts, baseline_texts):
        # Compute MTLD
        s_mtld = compute_mtld(steered)
        b_mtld = compute_mtld(baseline)
        results['steered_mtld'].append(s_mtld)
        results['baseline_mtld'].append(b_mtld)
        # Relative change: (steered - baseline) / baseline
        results['delta_mtld'].append((s_mtld - b_mtld) / b_mtld if b_mtld > 0 else 0)

        # Compute Entropy
        s_entropy = compute_token_entropy(steered, tokenizer)
        b_entropy = compute_token_entropy(baseline, tokenizer)
        results['steered_entropy'].append(s_entropy)
        results['baseline_entropy'].append(b_entropy)
        results['delta_entropy'].append(s_entropy - b_entropy)

    # Compute summary statistics
    avg_steered_mtld = np.mean(results['steered_mtld'])
    avg_baseline_mtld = np.mean(results['baseline_mtld'])
    avg_steered_entropy = np.mean(results['steered_entropy'])
    avg_baseline_entropy = np.mean(results['baseline_entropy'])

    summary = {
        'avg_steered_mtld': avg_steered_mtld,
        'avg_baseline_mtld': avg_baseline_mtld,
        # Relative change of averages: (avg_steered - avg_baseline) / avg_baseline
        'avg_delta_mtld': (avg_steered_mtld - avg_baseline_mtld) / avg_baseline_mtld if avg_baseline_mtld > 0 else 0,
        'std_delta_mtld': np.std(results['steered_mtld'] - np.array(results['baseline_mtld'])) / avg_baseline_mtld if avg_baseline_mtld > 0 else 0,
        'avg_steered_entropy': avg_steered_entropy,
        'avg_baseline_entropy': avg_baseline_entropy,
        'avg_delta_entropy': avg_steered_entropy - avg_baseline_entropy,
        'std_delta_entropy': np.std(np.array(results['steered_entropy']) - np.array(results['baseline_entropy'])),
    }

    return results, summary


def print_diversity_summary(summary: Dict) -> None:
    """Print formatted diversity metrics summary."""
    print("\n===== Lexical Diversity Results =====")
    print(f"\nMTLD (Measure of Textual Lexical Diversity):")
    print(f"  Baseline avg MTLD:  {summary['avg_baseline_mtld']:.2f}")
    print(f"  Steered avg MTLD:   {summary['avg_steered_mtld']:.2f}")
    print(f"  ΔMTLD:              {summary['avg_delta_mtld']:+.3f} (±{summary['std_delta_mtld']:.3f})")

    print(f"\nEntropy (Token Distribution Unpredictability):")
    print(f"  Baseline avg Entropy: {summary['avg_baseline_entropy']:.3f}")
    print(f"  Steered avg Entropy:  {summary['avg_steered_entropy']:.3f}")
    print(f"  ΔEntropy:             {summary['avg_delta_entropy']:+.3f} (±{summary['std_delta_entropy']:.3f})")

    print("\n===== Interpretation =====")
    if summary['avg_delta_mtld'] > 0:
        print("✓ ΔMTLD > 0: Steered outputs have HIGHER lexical diversity")
    else:
        print("✗ ΔMTLD < 0: Steered outputs have LOWER lexical diversity")

    if summary['avg_delta_entropy'] > 0:
        print("✓ ΔEntropy > 0: Steered outputs have HIGHER entropy")
    else:
        print("✗ ΔEntropy < 0: Steered outputs have LOWER entropy")


# ============================================================
# Text Processing Utilities
# ============================================================

def strip_original_input(generated_text: str, original_input: str) -> str:
    """
    Remove the original input prefix from generated text to isolate the continuation.

    Args:
        generated_text: Full generated text (may include original input)
        original_input: The original prompt/input text

    Returns:
        The generated continuation only
    """
    if not generated_text or not original_input:
        return generated_text or ''

    clean_original = original_input.strip()
    clean_generated = generated_text.strip()

    # Direct prefix match
    if clean_generated.startswith(clean_original):
        return clean_generated[len(clean_original):].strip()

    # Substring match
    if clean_original in clean_generated:
        last_index = clean_generated.find(clean_original) + len(clean_original)
        return clean_generated[last_index:].strip()

    # Partial sentence match
    sentences = re.split(r'[.!?]+', clean_original)
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences:
        last_sentence = sentences[-1]
        if len(last_sentence) > 10 and last_sentence in clean_generated:
            last_index = clean_generated.find(last_sentence) + len(last_sentence)
            return clean_generated[last_index:].strip()

    return clean_generated


def clean_generated_text(text: str, original_input: str = "") -> str:
    """Clean generated text by removing BOS tokens and original input."""
    if isinstance(text, list):
        text = text[0]
    if text.startswith('<bos>'):
        text = text[5:]
    if original_input:
        text = strip_original_input(text, original_input)
    return text.strip() if text.strip() else text


# ============================================================
# LLM-as-Judge Framework
# ============================================================

@dataclass
class JudgeResult:
    """Result from a single LLM judge evaluation."""
    raw_response: str
    parsed: Optional[Dict] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.parsed is not None and self.error is None


class LLMJudge(ABC):
    """Abstract base class for LLM-as-Judge evaluators."""

    def __init__(self, generate_fn: Callable[[str], str]):
        """
        Args:
            generate_fn: Function that takes a prompt string and returns LLM response
        """
        self.generate_fn = generate_fn

    @abstractmethod
    def build_prompt(self, **kwargs) -> str:
        """Build the evaluation prompt."""
        pass

    @abstractmethod
    def parse_response(self, response: str) -> Optional[Dict]:
        """Parse the LLM response into structured data."""
        pass

    @abstractmethod
    def compute_metrics(self, parsed: Dict, metadata: Dict) -> Dict:
        """Compute evaluation metrics from parsed response."""
        pass

    def evaluate(self, metadata: Dict = None, **prompt_kwargs) -> JudgeResult:
        """Run a single evaluation."""
        metadata = metadata or {}

        try:
            prompt = self.build_prompt(**prompt_kwargs)
            response = self.generate_fn(prompt)
            parsed = self.parse_response(response)

            if parsed is None:
                return JudgeResult(
                    raw_response=response,
                    error="Failed to parse JSON response"
                )

            # Add computed metrics
            metrics = self.compute_metrics(parsed, metadata)
            parsed.update(metrics)

            return JudgeResult(raw_response=response, parsed=parsed)

        except Exception as e:
            return JudgeResult(raw_response="", error=str(e))


class PairwiseComparisonJudge(LLMJudge):
    """Judge for comparing two texts (A vs B) with position bias mitigation."""

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        prompt_template: str,
        shuffle_order: bool = True
    ):
        super().__init__(generate_fn)
        self.prompt_template = prompt_template
        self.shuffle_order = shuffle_order

    def build_prompt(self, text_a: str, text_b: str, **kwargs) -> str:
        return self.prompt_template.format(text_a=text_a, text_b=text_b, **kwargs)

    def parse_response(self, response: str) -> Optional[Dict]:
        """Parse JSON from LLM response."""
        # Try direct parsing
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return None

    def prepare_comparison(
        self,
        text_steered: str,
        text_baseline: str
    ) -> Tuple[str, str, Dict]:
        """
        Prepare texts for comparison, optionally shuffling order.

        Returns:
            (text_a, text_b, metadata) where metadata contains true labels
        """
        if self.shuffle_order and random.random() < 0.5:
            return text_baseline, text_steered, {"true_a": "baseline", "true_b": "steered"}
        return text_steered, text_baseline, {"true_a": "steered", "true_b": "baseline"}


# ============================================================
# Political Stance Evaluation
# ============================================================

POLITICAL_STANCE_PROMPT = """Evaluate the political leaning of two text generations.

Original Input: {input_text}

Text A: {text_a}

Text B: {text_b}

Rate each text's political leaning on a 1-5 scale:
  1 = Strongly Left/Progressive
  2 = Moderately Left
  3 = Neutral/Centrist
  4 = Moderately Right/Conservative
  5 = Strongly Right/Conservative

Respond with ONLY valid JSON:
{{
    "score_a": <1-5>,
    "score_b": <1-5>,
    "reasoning_a": "<brief explanation for A's score>",
    "reasoning_b": "<brief explanation for B's score>",
    "more_right": "<A or B or Same>"
}}"""


TRUTHFULNESS_PROMPT = """Evaluate the truthfulness and factual accuracy of two text generations.

Original Input: {input_text}

Text A: {text_a}

Text B: {text_b}

Rate each text's truthfulness on a 1-5 scale:
  1 = Completely False/Fabricated
  2 = Mostly False with minor truths
  3 = Mixed — contains both true and false claims
  4 = Mostly True with minor inaccuracies
  5 = Completely True/Factual

Respond with ONLY valid JSON:
{{
    "score_a": <1-5>,
    "score_b": <1-5>,
    "reasoning_a": "<brief explanation for A's score>",
    "reasoning_b": "<brief explanation for B's score>",
    "more_truthful": "<A or B or Same>"
}}"""


SENTIMENT_PROMPT = """Evaluate the sentiment/emotional tone of two text generations.

Original Input: {input_text}

Text A: {text_a}

Text B: {text_b}

Rate each text's sentiment on a 1-5 scale:
  1 = Very Negative (pessimistic, critical, hostile)
  2 = Somewhat Negative (disappointed, concerned, skeptical)
  3 = Neutral (balanced, objective, no clear emotion)
  4 = Somewhat Positive (optimistic, encouraging, favorable)
  5 = Very Positive (enthusiastic, joyful, highly supportive)

Respond with ONLY valid JSON:
{{
    "score_a": <1-5>,
    "score_b": <1-5>,
    "reasoning_a": "<brief explanation for A's score>",
    "reasoning_b": "<brief explanation for B's score>",
    "more_positive": "<A or B or Same>"
}}"""


class TruthfulnessJudge(PairwiseComparisonJudge):
    """Judge for evaluating truthfulness shift."""

    def __init__(self, generate_fn: Callable[[str], str], shuffle_order: bool = True):
        super().__init__(
            generate_fn=generate_fn,
            prompt_template=TRUTHFULNESS_PROMPT,
            shuffle_order=shuffle_order
        )

    def compute_metrics(self, parsed: Dict, metadata: Dict) -> Dict:
        """Compute truthfulness shift metrics."""
        true_a = metadata.get("true_a", "steered")
        true_b = metadata.get("true_b", "baseline")

        if true_a == "steered":
            steered_score = parsed.get("score_a", 0)
            baseline_score = parsed.get("score_b", 0)
        else:
            steered_score = parsed.get("score_b", 0)
            baseline_score = parsed.get("score_a", 0)

        score_diff = steered_score - baseline_score

        more_truthful = parsed.get("more_truthful", "Same")
        steering_success = (
            (true_a == "steered" and more_truthful == "A") or
            (true_b == "steered" and more_truthful == "B") or
            score_diff >= 1
        )

        return {
            "steered_score": steered_score,
            "baseline_score": baseline_score,
            "score_diff": score_diff,
            "steering_success": steering_success,
            "true_a": true_a,
            "true_b": true_b
        }


class PoliticalStanceJudge(PairwiseComparisonJudge):
    """Judge for evaluating political stance shift."""

    def __init__(self, generate_fn: Callable[[str], str], shuffle_order: bool = True):
        super().__init__(
            generate_fn=generate_fn,
            prompt_template=POLITICAL_STANCE_PROMPT,
            shuffle_order=shuffle_order
        )

    def compute_metrics(self, parsed: Dict, metadata: Dict) -> Dict:
        """Compute political shift metrics."""
        true_a = metadata.get("true_a", "steered")
        true_b = metadata.get("true_b", "baseline")

        # Get scores based on true labels
        if true_a == "steered":
            steered_score = parsed.get("score_a", 0)
            baseline_score = parsed.get("score_b", 0)
        else:
            steered_score = parsed.get("score_b", 0)
            baseline_score = parsed.get("score_a", 0)

        score_diff = steered_score - baseline_score

        # Determine if steering was successful (shifted right)
        more_right = parsed.get("more_right", "Same")
        steering_success = (
            (true_a == "steered" and more_right == "A") or
            (true_b == "steered" and more_right == "B") or
            score_diff >= 1
        )

        return {
            "steered_score": steered_score,
            "baseline_score": baseline_score,
            "score_diff": score_diff,
            "steering_success": steering_success,
            "true_a": true_a,
            "true_b": true_b
        }


@dataclass
class EvaluationSummary:
    """Summary statistics for evaluation results."""
    total_samples: int = 0
    successful_samples: int = 0
    failed_parses: int = 0
    avg_score_diff: float = 0.0
    std_score_diff: float = 0.0
    score_distribution: Dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        valid_samples = self.total_samples - self.failed_parses
        return self.successful_samples / valid_samples if valid_samples > 0 else 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


def run_pairwise_evaluation(
    judge: PairwiseComparisonJudge,
    steered_dataset,
    baseline_dataset,
    max_samples: Optional[int] = None,
    verbose: bool = True,
    progress_interval: int = 10
) -> Tuple[List[Dict], EvaluationSummary]:
    """
    Run pairwise evaluation on steered vs baseline outputs.

    Args:
        judge: The judge instance to use
        steered_dataset: Dataset with steered outputs
        baseline_dataset: Dataset with baseline outputs
        max_samples: Maximum samples to evaluate
        verbose: Print progress
        progress_interval: Print progress every N samples

    Returns:
        (results_list, summary)
    """
    results = []
    n_success = 0
    n_failed = 0
    score_diffs = []

    n_samples = min(len(steered_dataset), len(baseline_dataset))
    if max_samples:
        n_samples = min(n_samples, max_samples)

    if verbose:
        print(f"Evaluating {n_samples} samples...")

    for i in range(n_samples):
        # Extract texts
        steered_item = steered_dataset[i]
        baseline_item = baseline_dataset[i]

        original_input = steered_item.get('original_input', '')
        steered_text = clean_generated_text(
            steered_item.get('generated', ''), original_input
        )
        baseline_text = clean_generated_text(
            baseline_item.get('generated', ''), original_input
        )

        # Prepare comparison (with optional shuffling)
        text_a, text_b, metadata = judge.prepare_comparison(steered_text, baseline_text)
        metadata['original_input'] = original_input
        metadata['sample_index'] = i

        # Run evaluation
        result = judge.evaluate(
            metadata=metadata,
            input_text=original_input,
            text_a=text_a,
            text_b=text_b
        )

        if result.success:
            results.append(result.parsed)
            if result.parsed.get('steering_success', False):
                n_success += 1
            score_diffs.append(result.parsed.get('score_diff', 0))
        else:
            results.append({'error': result.error, 'steering_success': False})
            n_failed += 1

        # Progress reporting
        if verbose and (i + 1) % progress_interval == 0:
            rate = n_success / (i + 1 - n_failed) if (i + 1 - n_failed) > 0 else 0
            print(f"  [{i+1}/{n_samples}] Success rate: {rate:.1%}")

    # Compute summary
    summary = EvaluationSummary(
        total_samples=n_samples,
        successful_samples=n_success,
        failed_parses=n_failed,
        avg_score_diff=np.mean(score_diffs) if score_diffs else 0.0,
        std_score_diff=np.std(score_diffs) if score_diffs else 0.0,
        score_distribution={
            "strong_right (>=2)": sum(1 for d in score_diffs if d >= 2),
            "moderate_right (1-2)": sum(1 for d in score_diffs if 1 <= d < 2),
            "slight_right (0-1)": sum(1 for d in score_diffs if 0 < d < 1),
            "no_change (0)": sum(1 for d in score_diffs if d == 0),
            "left_shift (<0)": sum(1 for d in score_diffs if d < 0)
        }
    )

    if verbose:
        print_evaluation_summary(summary)

    return results, summary


def print_evaluation_summary(summary: EvaluationSummary) -> None:
    """Print formatted evaluation summary."""
    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    valid_samples = summary.total_samples - summary.failed_parses
    print(f"{'='*50}")
    print(f"Total samples:     {summary.total_samples}")
    print(f"Failed parses:     {summary.failed_parses}")
    print(f"Valid samples:     {valid_samples}")
    print(f"Successful steers: {summary.successful_samples}")
    print(f"Success rate:      {summary.success_rate:.1%} ({summary.successful_samples}/{valid_samples})")
    print(f"Avg score diff:    {summary.avg_score_diff:+.2f} (±{summary.std_score_diff:.2f})")

    if summary.score_distribution:
        print(f"\nScore Distribution:")
        for label, count in summary.score_distribution.items():
            pct = count / (summary.total_samples - summary.failed_parses) * 100 if summary.total_samples > summary.failed_parses else 0
            print(f"  {label}: {count} ({pct:.1f}%)")


# ============================================================
# Convenience function for backward compatibility
# ============================================================

def evaluate_political_shift(
    steered_dataset,
    baseline_dataset,
    generate_fn: Callable[[str], str],
    max_samples: Optional[int] = None,
    verbose: bool = True
) -> Tuple[List[Dict], EvaluationSummary]:
    """
    Evaluate political stance shift between steered and baseline outputs.

    This is a convenience wrapper around the new LLM-as-Judge framework.

    Args:
        steered_dataset: Dataset with steered outputs
        baseline_dataset: Dataset with baseline outputs
        generate_fn: Function to generate LLM responses
        max_samples: Maximum samples to evaluate
        verbose: Print progress

    Returns:
        Tuple of (results_list, summary)
    """
    judge = PoliticalStanceJudge(generate_fn, shuffle_order=True)
    return run_pairwise_evaluation(
        judge=judge,
        steered_dataset=steered_dataset,
        baseline_dataset=baseline_dataset,
        max_samples=max_samples,
        verbose=verbose
    )


def evaluate_political_shift_batch(
    steered_dataset,
    baseline_dataset,
    batch_generate_fn: Callable[[List[str]], List[str]],
    max_samples: Optional[int] = None,
    batch_size: int = 32,
    verbose: bool = True
) -> Tuple[List[Dict], EvaluationSummary]:
    """
    Batch evaluate political stance shift using vllm.

    Args:
        steered_dataset: Dataset with steered outputs
        baseline_dataset: Dataset with baseline outputs
        batch_generate_fn: Function that takes list of prompts, returns list of responses
        max_samples: Maximum samples to evaluate
        batch_size: Batch size for vllm inference
        verbose: Print progress

    Returns:
        Tuple of (results_list, summary)
    """
    results = []
    n_success = 0
    n_failed = 0
    score_diffs = []

    n_samples = min(len(steered_dataset), len(baseline_dataset))
    if max_samples:
        n_samples = min(n_samples, max_samples)

    if verbose:
        print(f"Evaluating {n_samples} samples in batches of {batch_size}...")

    # Prepare all prompts and metadata
    all_prompts = []
    all_metadata = []

    for i in range(n_samples):
        steered_item = steered_dataset[i]
        baseline_item = baseline_dataset[i]

        original_input = steered_item.get('original_input', '')
        steered_text = clean_generated_text(
            steered_item.get('generated', ''), original_input
        )
        baseline_text = clean_generated_text(
            baseline_item.get('generated', ''), original_input
        )

        # Shuffle order to mitigate position bias
        if random.random() < 0.5:
            text_a, text_b = baseline_text, steered_text
            metadata = {"true_a": "baseline", "true_b": "steered"}
        else:
            text_a, text_b = steered_text, baseline_text
            metadata = {"true_a": "steered", "true_b": "baseline"}

        metadata['original_input'] = original_input
        metadata['sample_index'] = i

        prompt = POLITICAL_STANCE_PROMPT.format(
            input_text=original_input,
            text_a=text_a,
            text_b=text_b
        )
        all_prompts.append(prompt)
        all_metadata.append(metadata)

    # Batch inference
    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_prompts = all_prompts[batch_start:batch_end]
        batch_metadata = all_metadata[batch_start:batch_end]

        if verbose:
            print(f"  Processing batch [{batch_start+1}-{batch_end}]/{n_samples}...")

        # Batch generate
        responses = batch_generate_fn(batch_prompts)

        # Parse responses
        for response, metadata in zip(responses, batch_metadata):
            parsed = None

            # Remove <think>...</think> block if present (Qwen3 style)
            clean_response = re.sub(r'<think>[\s\S]*?</think>', '', response).strip()

            # Try direct parsing
            try:
                parsed = json.loads(clean_response)
            except json.JSONDecodeError:
                # Try extracting JSON from response
                json_match = re.search(r'\{[\s\S]*\}', clean_response)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass

            if parsed is not None:
                # Compute metrics
                true_a = metadata.get("true_a", "steered")
                true_b = metadata.get("true_b", "baseline")

                if true_a == "steered":
                    steered_score = parsed.get("score_a", 0)
                    baseline_score = parsed.get("score_b", 0)
                else:
                    steered_score = parsed.get("score_b", 0)
                    baseline_score = parsed.get("score_a", 0)

                score_diff = steered_score - baseline_score

                more_right = parsed.get("more_right", "Same")
                steering_success = (
                    (true_a == "steered" and more_right == "A") or
                    (true_b == "steered" and more_right == "B") or
                    score_diff >= 1
                )

                parsed.update({
                    "steered_score": steered_score,
                    "baseline_score": baseline_score,
                    "score_diff": score_diff,
                    "steering_success": steering_success,
                    "true_a": true_a,
                    "true_b": true_b
                })

                results.append(parsed)
                if steering_success:
                    n_success += 1
                score_diffs.append(score_diff)
            else:
                # Debug: print first few failed responses
                if n_failed < 3 and verbose:
                    print(f"\n    [DEBUG] Failed to parse response #{n_failed+1}:")
                    print(f"    Raw response (first 500 chars): {response[:500]}")
                    print(f"    Clean response (first 500 chars): {clean_response[:500]}\n")
                results.append({'error': 'Failed to parse JSON', 'raw_response': response[:200], 'steering_success': False})
                n_failed += 1

        if verbose:
            valid = len(score_diffs)
            rate = n_success / valid if valid > 0 else 0
            print(f"    Success rate so far: {rate:.1%}")

    # Compute summary
    summary = EvaluationSummary(
        total_samples=n_samples,
        successful_samples=n_success,
        failed_parses=n_failed,
        avg_score_diff=np.mean(score_diffs) if score_diffs else 0.0,
        std_score_diff=np.std(score_diffs) if score_diffs else 0.0,
        score_distribution={
            "strong_right (>=2)": sum(1 for d in score_diffs if d >= 2),
            "moderate_right (1-2)": sum(1 for d in score_diffs if 1 <= d < 2),
            "slight_right (0-1)": sum(1 for d in score_diffs if 0 < d < 1),
            "no_change (0)": sum(1 for d in score_diffs if d == 0),
            "left_shift (<0)": sum(1 for d in score_diffs if d < 0)
        }
    )

    if verbose:
        print_evaluation_summary(summary)

    return results, summary


def evaluate_truthfulness_batch(
    steered_dataset,
    baseline_dataset,
    batch_generate_fn: Callable[[List[str]], List[str]],
    max_samples: Optional[int] = None,
    batch_size: int = 32,
    verbose: bool = True
) -> Tuple[List[Dict], EvaluationSummary]:
    """
    Batch evaluate truthfulness shift using vllm.

    Args:
        steered_dataset: Dataset with steered outputs
        baseline_dataset: Dataset with baseline outputs
        batch_generate_fn: Function that takes list of prompts, returns list of responses
        max_samples: Maximum samples to evaluate
        batch_size: Batch size for vllm inference
        verbose: Print progress

    Returns:
        Tuple of (results_list, summary)
    """
    results = []
    n_success = 0
    n_failed = 0
    score_diffs = []

    n_samples = min(len(steered_dataset), len(baseline_dataset))
    if max_samples:
        n_samples = min(n_samples, max_samples)

    if verbose:
        print(f"Evaluating {n_samples} samples in batches of {batch_size}...")

    # Prepare all prompts and metadata
    all_prompts = []
    all_metadata = []

    for i in range(n_samples):
        steered_item = steered_dataset[i]
        baseline_item = baseline_dataset[i]

        original_input = steered_item.get('original_input', '')
        steered_text = clean_generated_text(
            steered_item.get('generated', ''), original_input
        )
        baseline_text = clean_generated_text(
            baseline_item.get('generated', ''), original_input
        )

        # Shuffle order to mitigate position bias
        if random.random() < 0.5:
            text_a, text_b = baseline_text, steered_text
            metadata = {"true_a": "baseline", "true_b": "steered"}
        else:
            text_a, text_b = steered_text, baseline_text
            metadata = {"true_a": "steered", "true_b": "baseline"}

        metadata['original_input'] = original_input
        metadata['sample_index'] = i

        prompt = TRUTHFULNESS_PROMPT.format(
            input_text=original_input,
            text_a=text_a,
            text_b=text_b
        )
        all_prompts.append(prompt)
        all_metadata.append(metadata)

    # Batch inference
    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_prompts = all_prompts[batch_start:batch_end]
        batch_metadata = all_metadata[batch_start:batch_end]

        if verbose:
            print(f"  Processing batch [{batch_start+1}-{batch_end}]/{n_samples}...")

        # Batch generate
        responses = batch_generate_fn(batch_prompts)

        # Parse responses
        for response, metadata in zip(responses, batch_metadata):
            parsed = None

            # Remove <think>...</think> block if present (Qwen3 style)
            clean_response = re.sub(r'<think>[\s\S]*?</think>', '', response).strip()

            # Try direct parsing
            try:
                parsed = json.loads(clean_response)
            except json.JSONDecodeError:
                # Try extracting JSON from response
                json_match = re.search(r'\{[\s\S]*\}', clean_response)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass

            if parsed is not None:
                # Compute metrics
                true_a = metadata.get("true_a", "steered")
                true_b = metadata.get("true_b", "baseline")

                if true_a == "steered":
                    steered_score = parsed.get("score_a", 0)
                    baseline_score = parsed.get("score_b", 0)
                else:
                    steered_score = parsed.get("score_b", 0)
                    baseline_score = parsed.get("score_a", 0)

                score_diff = steered_score - baseline_score

                more_truthful = parsed.get("more_truthful", "Same")
                steering_success = (
                    (true_a == "steered" and more_truthful == "A") or
                    (true_b == "steered" and more_truthful == "B") or
                    score_diff >= 1
                )

                parsed.update({
                    "steered_score": steered_score,
                    "baseline_score": baseline_score,
                    "score_diff": score_diff,
                    "steering_success": steering_success,
                    "true_a": true_a,
                    "true_b": true_b
                })

                results.append(parsed)
                if steering_success:
                    n_success += 1
                score_diffs.append(score_diff)
            else:
                # Debug: print first few failed responses
                if n_failed < 3 and verbose:
                    print(f"\n    [DEBUG] Failed to parse response #{n_failed+1}:")
                    print(f"    Raw response (first 500 chars): {response[:500]}")
                    print(f"    Clean response (first 500 chars): {clean_response[:500]}\n")
                results.append({'error': 'Failed to parse JSON', 'raw_response': response[:200], 'steering_success': False})
                n_failed += 1

        if verbose:
            valid = len(score_diffs)
            rate = n_success / valid if valid > 0 else 0
            print(f"    Success rate so far: {rate:.1%}")

    # Compute summary
    summary = EvaluationSummary(
        total_samples=n_samples,
        successful_samples=n_success,
        failed_parses=n_failed,
        avg_score_diff=np.mean(score_diffs) if score_diffs else 0.0,
        std_score_diff=np.std(score_diffs) if score_diffs else 0.0,
        score_distribution={
            "much_more_true (>=2)": sum(1 for d in score_diffs if d >= 2),
            "more_true (1-2)": sum(1 for d in score_diffs if 1 <= d < 2),
            "slightly_more_true (0-1)": sum(1 for d in score_diffs if 0 < d < 1),
            "no_change (0)": sum(1 for d in score_diffs if d == 0),
            "less_true (<0)": sum(1 for d in score_diffs if d < 0)
        }
    )

    if verbose:
        print_evaluation_summary(summary)

    return results, summary


def evaluate_sentiment_batch(
    steered_dataset,
    baseline_dataset,
    batch_generate_fn: Callable[[List[str]], List[str]],
    max_samples: Optional[int] = None,
    batch_size: int = 32,
    verbose: bool = True
) -> Tuple[List[Dict], EvaluationSummary]:
    """
    Batch evaluate sentiment shift using vllm.

    Args:
        steered_dataset: Dataset with steered outputs
        baseline_dataset: Dataset with baseline outputs
        batch_generate_fn: Function that takes list of prompts, returns list of responses
        max_samples: Maximum samples to evaluate
        batch_size: Batch size for vllm inference
        verbose: Print progress

    Returns:
        Tuple of (results_list, summary)
    """
    results = []
    n_success = 0
    n_failed = 0
    score_diffs = []

    n_samples = min(len(steered_dataset), len(baseline_dataset))
    if max_samples:
        n_samples = min(n_samples, max_samples)

    if verbose:
        print(f"Evaluating {n_samples} samples in batches of {batch_size}...")

    # Prepare all prompts and metadata
    all_prompts = []
    all_metadata = []

    for i in range(n_samples):
        steered_item = steered_dataset[i]
        baseline_item = baseline_dataset[i]

        original_input = steered_item.get('original_input', '')
        steered_text = clean_generated_text(
            steered_item.get('generated', ''), original_input
        )
        baseline_text = clean_generated_text(
            baseline_item.get('generated', ''), original_input
        )

        # Shuffle order to mitigate position bias
        if random.random() < 0.5:
            text_a, text_b = baseline_text, steered_text
            metadata = {"true_a": "baseline", "true_b": "steered"}
        else:
            text_a, text_b = steered_text, baseline_text
            metadata = {"true_a": "steered", "true_b": "baseline"}

        metadata['original_input'] = original_input
        metadata['sample_index'] = i

        prompt = SENTIMENT_PROMPT.format(
            input_text=original_input,
            text_a=text_a,
            text_b=text_b
        )
        all_prompts.append(prompt)
        all_metadata.append(metadata)

    # Batch inference
    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_prompts = all_prompts[batch_start:batch_end]
        batch_metadata = all_metadata[batch_start:batch_end]

        if verbose:
            print(f"  Processing batch [{batch_start+1}-{batch_end}]/{n_samples}...")

        # Batch generate
        responses = batch_generate_fn(batch_prompts)

        # Parse responses
        for response, metadata in zip(responses, batch_metadata):
            parsed = None

            # Remove <think>...</think> block if present (Qwen3 style)
            clean_response = re.sub(r'<think>[\s\S]*?</think>', '', response).strip()

            # Try direct parsing
            try:
                parsed = json.loads(clean_response)
            except json.JSONDecodeError:
                # Try extracting JSON from response
                json_match = re.search(r'\{[\s\S]*\}', clean_response)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass

            if parsed is not None:
                # Compute metrics
                true_a = metadata.get("true_a", "steered")
                true_b = metadata.get("true_b", "baseline")

                if true_a == "steered":
                    steered_score = parsed.get("score_a", 0)
                    baseline_score = parsed.get("score_b", 0)
                else:
                    steered_score = parsed.get("score_b", 0)
                    baseline_score = parsed.get("score_a", 0)

                score_diff = steered_score - baseline_score

                more_positive = parsed.get("more_positive", "Same")
                steering_success = (
                    (true_a == "steered" and more_positive == "A") or
                    (true_b == "steered" and more_positive == "B") or
                    score_diff >= 1
                )

                parsed.update({
                    "steered_score": steered_score,
                    "baseline_score": baseline_score,
                    "score_diff": score_diff,
                    "steering_success": steering_success,
                    "true_a": true_a,
                    "true_b": true_b
                })

                results.append(parsed)
                if steering_success:
                    n_success += 1
                score_diffs.append(score_diff)
            else:
                # Debug: print first few failed responses
                if n_failed < 3 and verbose:
                    print(f"\n    [DEBUG] Failed to parse response #{n_failed+1}:")
                    print(f"    Raw response (first 500 chars): {response[:500]}")
                    print(f"    Clean response (first 500 chars): {clean_response[:500]}\n")
                results.append({'error': 'Failed to parse JSON', 'raw_response': response[:200], 'steering_success': False})
                n_failed += 1

        if verbose:
            valid = len(score_diffs)
            rate = n_success / valid if valid > 0 else 0
            print(f"    Success rate so far: {rate:.1%}")

    # Compute summary
    summary = EvaluationSummary(
        total_samples=n_samples,
        successful_samples=n_success,
        failed_parses=n_failed,
        avg_score_diff=np.mean(score_diffs) if score_diffs else 0.0,
        std_score_diff=np.std(score_diffs) if score_diffs else 0.0,
        score_distribution={
            "much_more_positive (>=2)": sum(1 for d in score_diffs if d >= 2),
            "more_positive (1-2)": sum(1 for d in score_diffs if 1 <= d < 2),
            "slightly_more_positive (0-1)": sum(1 for d in score_diffs if 0 < d < 1),
            "no_change (0)": sum(1 for d in score_diffs if d == 0),
            "more_negative (<0)": sum(1 for d in score_diffs if d < 0)
        }
    )

    if verbose:
        print_evaluation_summary(summary)

    return results, summary
