"""
Sliding-window perplexity evaluation for RBVTQuant models.

Outside the RBVT assignment method itself, this file is kept intentionally close
to NCCQuant's baseline evaluator so RTN and RBVT are compared on the same
evaluation backbone.
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _hf_device_map(device: str):
    return {"": device} if device != "auto" else "auto"


class RBVTSlidingWindowEvaluator:
    def __init__(
        self,
        device: str = "cuda",
        seed: int = 42,
        stride: int = 512,
        max_length: int = 2048,
        cache_dir: str = "./dataset_cache",
        hf_token: str | None = None,
    ):
        self.device = device
        self.seed = seed
        self.stride = stride
        self.max_length = max_length
        self.results = {}
        self.cache_dir = Path(cache_dir)
        self.hf_token = hf_token
        self.cache_dir.mkdir(exist_ok=True)

        print("=" * 80)
        print("RBVT SLIDING WINDOW PERPLEXITY EVALUATION")
        print("=" * 80)
        print(f"Device: {device}")
        print(f"Stride: {stride}")
        print(f"Max Seq Length: {max_length}")
        print(f"Cache Dir: {cache_dir}")
        print("=" * 80)

    def load_wikitext2_test(self, n_samples=None):
        print("\n[1/3] Loading WikiText-2 test...")

        cache_file = self.cache_dir / f"wikitext2_test_seed{self.seed}.pkl"
        if cache_file.exists():
            print(f"  Loading from cache: {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        dataset = load_dataset(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            split="test",
            token=self.hf_token,
        )
        full_text = "\n".join([x for x in dataset["text"] if x])
        print(f"  Loaded continuous stream ({len(full_text)} chars)")

        result = [full_text]
        print(f"  Saving to cache: {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        return result

    def load_c4_validation(self, n_samples: int = 500):
        print("\n[2/3] Loading C4 validation...")

        cache_file = self.cache_dir / f"c4_validation_n{n_samples}_seed{self.seed}.pkl"
        if cache_file.exists():
            print(f"  Loading from cache: {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        dataset = load_dataset(
            "allenai/c4",
            "en",
            split="validation",
            streaming=True,
            token=self.hf_token,
        )
        texts = []
        for item in tqdm(dataset, total=n_samples, desc="  Collecting C4"):
            if len(texts) >= n_samples:
                break
            if len(item["text"].strip()) > 500:
                texts.append(item["text"])

        full_text = "\n\n".join(texts)
        print(f"  Loaded continuous stream ({len(full_text)} chars, {len(texts)} documents)")

        result = [full_text]
        print(f"  Saving to cache: {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        return result

    @torch.no_grad()
    def evaluate_sliding_window(self, model, tokenizer, texts):
        model.eval()
        nlls = []
        total_tokens = 0

        for text in texts:
            encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            input_ids = encodings.input_ids

            if tokenizer.bos_token_id is not None:
                if input_ids.shape[1] == 0 or input_ids[0, 0].item() != tokenizer.bos_token_id:
                    bos_tensor = torch.tensor([[tokenizer.bos_token_id]], device=input_ids.device)
                    input_ids = torch.cat([bos_tensor, input_ids], dim=1)

            if input_ids.size(1) > self.max_length * 200:
                input_ids = input_ids[:, : self.max_length * 200]

            input_ids = input_ids.to(self.device)
            seq_len = input_ids.size(1)
            if seq_len < 2:
                continue

            window_range = list(range(0, seq_len, self.stride))
            num_windows = len(window_range)
            print(f"  Processing {seq_len:,} tokens in {num_windows} windows...")

            prev_end_loc = 0
            pbar = tqdm(window_range, desc="  Windows", unit="win", leave=False)

            for begin_loc in pbar:
                end_loc = min(begin_loc + self.max_length, seq_len)
                trg_len = end_loc - prev_end_loc

                input_chunk = input_ids[:, begin_loc:end_loc]
                target_chunk = input_chunk.clone()
                if begin_loc > 0:
                    target_chunk[:, :-trg_len] = -100

                if target_chunk.size(1) == 0:
                    break

                outputs = model(input_chunk, labels=target_chunk)
                neg_log_likelihood = outputs.loss * trg_len

                nlls.append(neg_log_likelihood)
                prev_end_loc = end_loc

                current_nll = torch.stack(nlls).sum()
                current_ppl = torch.exp(current_nll / (total_tokens + prev_end_loc)).item()
                pbar.set_postfix({"PPL": f"{current_ppl:.4f}", "tokens": f"{total_tokens + prev_end_loc:,}"})

                if end_loc == seq_len:
                    break

            total_tokens += seq_len

        if not nlls:
            return None

        total_nll = torch.stack(nlls).sum()
        perplexity = torch.exp(total_nll / total_tokens).item()
        return {"perplexity": perplexity, "total_tokens": total_tokens}

    @staticmethod
    def load_tokenizer(model_path: str):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True,
            )
        return tokenizer

    def evaluate_model_on_dataset(self, model_path: str, model_name: str, texts, dataset_name: str):
        print(f"\n  Evaluating {model_name} on {dataset_name}...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True,
                token=self.hf_token,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=_hf_device_map(self.device),
                trust_remote_code=True,
                token=self.hf_token,
            )

            results = self.evaluate_sliding_window(model, tokenizer, texts)
            if results:
                print(f"  Perplexity: {results['perplexity']:.4f}")
            else:
                print("  Evaluation failed (no results)")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return results

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_evaluation(self, rbvt_path: str, baseline_path: str | None = None, n_samples: int = 2000):
        print("\n" + "=" * 80)
        print("LOADING DATASETS")
        print("=" * 80)

        datasets = {
            "WikiText-2": self.load_wikitext2_test(n_samples),
            "C4": self.load_c4_validation(n_samples),
        }

        print("\n" + "=" * 80)
        print("EVALUATING MODELS")
        print("=" * 80)

        models = {"RBVTQuant": rbvt_path}
        if baseline_path:
            models["Baseline"] = baseline_path

        for dataset_name, texts in datasets.items():
            print(f"\n{'=' * 80}")
            print(f"Dataset: {dataset_name}")
            print(f"{'=' * 80}")
            for model_name, model_path in models.items():
                result = self.evaluate_model_on_dataset(model_path, model_name, texts, dataset_name)
                if result:
                    self.results.setdefault(dataset_name, {})[model_name] = result

        return self.results

    def generate_report(self):
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)

        has_comparison = any(len(models) == 2 for models in self.results.values())
        if not has_comparison:
            print(f"\n{'Dataset':<15} {'Model':<15} {'Perplexity':<15} {'Total Tokens':<15}")
            print("-" * 64)
            rows = []
            for dataset_name, models_data in self.results.items():
                for model_name, data in models_data.items():
                    ppl = data["perplexity"]
                    tokens = data["total_tokens"]
                    print(f"{dataset_name:<15} {model_name:<15} {ppl:<15.4f} {tokens:<15,}")
                    rows.append({
                        "dataset": dataset_name,
                        "model": model_name,
                        "perplexity": ppl,
                        "total_tokens": tokens,
                    })
            return rows

        print(f"\n{'Dataset':<15} {'RBVTQuant':<15} {'Baseline':<15} {'Delta %':<12} {'Winner':<10}")
        print("-" * 76)
        rows = []
        for dataset_name, models_data in self.results.items():
            if "RBVTQuant" not in models_data or "Baseline" not in models_data:
                continue
            rbvt_ppl = models_data["RBVTQuant"]["perplexity"]
            baseline_ppl = models_data["Baseline"]["perplexity"]
            delta_pct = ((rbvt_ppl - baseline_ppl) / baseline_ppl) * 100.0
            winner = "RBVTQuant" if delta_pct < -0.05 else ("Baseline" if delta_pct > 0.05 else "Tie")
            print(f"{dataset_name:<15} {rbvt_ppl:<15.4f} {baseline_ppl:<15.4f} {delta_pct:+11.3f}%  {winner:<10}")
            rows.append({
                "dataset": dataset_name,
                "rbvt_ppl": rbvt_ppl,
                "baseline_ppl": baseline_ppl,
                "delta_pct": delta_pct,
                "winner": winner,
            })
        return rows

    def summarize(self, rows):
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        if not rows or "rbvt_ppl" not in rows[0]:
            print("Single-model perplexity evaluation complete.")
            return {"mode": "single_model"}

        rbvt_wins = sum(1 for row in rows if row["winner"] == "RBVTQuant")
        baseline_wins = sum(1 for row in rows if row["winner"] == "Baseline")
        ties = sum(1 for row in rows if row["winner"] == "Tie")
        avg_rbvt = np.mean([row["rbvt_ppl"] for row in rows])
        avg_baseline = np.mean([row["baseline_ppl"] for row in rows])
        avg_delta_pct = ((avg_rbvt - avg_baseline) / avg_baseline) * 100.0

        print(f"RBVT wins: {rbvt_wins}/{len(rows)}")
        print(f"Baseline wins: {baseline_wins}/{len(rows)}")
        print(f"Ties: {ties}/{len(rows)}")
        print(f"Average RBVT perplexity: {avg_rbvt:.4f}")
        print(f"Average baseline perplexity: {avg_baseline:.4f}")
        print(f"Average delta: {avg_delta_pct:+.3f}%")

        if rbvt_wins > baseline_wins:
            winner = "RBVTQuant"
        elif baseline_wins > rbvt_wins:
            winner = "Baseline"
        else:
            winner = "Tie"
        print(f"Overall winner: {winner}")

        return {
            "mode": "comparison",
            "winner": winner,
            "rbvt_wins": rbvt_wins,
            "baseline_wins": baseline_wins,
            "ties": ties,
            "avg_rbvt": avg_rbvt,
            "avg_baseline": avg_baseline,
            "avg_delta_pct": avg_delta_pct,
        }


def main():
    parser = argparse.ArgumentParser(
        description="RBVT sliding-window perplexity evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rbvt-path", type=str, required=True, help="Path to the RBVT-quantized model")
    parser.add_argument("--baseline-path", type=str, default="", help="Optional baseline model path for side-by-side comparison")
    parser.add_argument("--n-samples", type=int, default=2000, help="Number of samples/documents for stream datasets")
    parser.add_argument("--stride", type=int, default=512, help="Sliding-window stride")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum window length in tokens")
    parser.add_argument("--cache-dir", type=str, default="./dataset_cache", help="Dataset cache directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device or device_map value for model loading")
    args = parser.parse_args()

    evaluator = RBVTSlidingWindowEvaluator(
        device=args.device,
        seed=args.seed,
        stride=args.stride,
        max_length=args.max_length,
        cache_dir=args.cache_dir,
    )
    evaluator.run_evaluation(
        rbvt_path=args.rbvt_path,
        baseline_path=args.baseline_path if args.baseline_path else None,
        n_samples=args.n_samples,
    )
    rows = evaluator.generate_report()
    evaluator.summarize(rows)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
