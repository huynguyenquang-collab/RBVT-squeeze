"""Verify the datasets compatibility path required by lm-eval tasks."""

from __future__ import annotations

import argparse

from lm_eval_runner import LMEvalHarnessRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-piqa", action="store_true")
    args = parser.parse_args()

    runner = LMEvalHarnessRunner(tasks=["piqa"])
    runner._patch_datasets_repo_aliases()

    from datasets import Features

    sequence_feature = {
        "choices": {
            "_type": "Sequence",
            "feature": {"_type": "Value", "dtype": "string"},
        }
    }
    features = Features.from_dict(sequence_feature)
    if features["choices"].__class__.__name__ not in {"Sequence", "LargeList"}:
        raise RuntimeError(f"Sequence compatibility failed: {features!r}")
    print("datasets Sequence compatibility: OK")

    if args.download_piqa:
        from datasets import load_dataset

        dataset = load_dataset("piqa", split="validation")
        if len(dataset) == 0:
            raise RuntimeError("PIQA validation split is empty")
        print(f"PIQA dataset load: OK ({len(dataset)} rows)")


if __name__ == "__main__":
    main()
