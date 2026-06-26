"""
Calibration data loading utilities for RBVTQuant.

This is a local copy so RBVTQuant remains self-contained and does not depend on
NCCQuant at runtime.
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset


def get_c4_calibration_data(tokenizer, n_samples=128, seqlen=2048, seed=42, return_tensors=False, cache_dir="./calibration_cache"):
    print(f"\n[C4 Calibration Data - Optimized]")
    print(f"  Samples: {n_samples}")
    print(f"  Sequence length: {seqlen} tokens")
    print(f"  Method: Random slicing with fast filtering")
    print(f"  Seed: {seed}")

    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    cache_file = cache_path / f"c4_calib_n{n_samples}_len{seqlen}_seed{seed}_tensors{return_tensors}.pkl"
    if cache_file.exists():
        print(f"\n  Loading from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"\n  No cache found, downloading from C4...")
    random.seed(seed)

    url = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz"
    traindata = load_dataset(
        "json",
        data_files={"train": url},
        split="train",
        streaming=True,
    )

    dataset = []
    skipped = 0
    char_threshold = seqlen * 3

    print(f"\n  Streaming C4 with fast filtering...")
    for data in traindata:
        text = data["text"]
        if len(text) < char_threshold:
            skipped += 1
            continue

        trainenc = tokenizer(text, return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            skipped += 1
            continue

        max_start = trainenc.input_ids.shape[1] - seqlen
        start_idx = random.randint(0, max_start)
        end_idx = start_idx + seqlen
        inp = trainenc.input_ids[:, start_idx:end_idx]

        if return_tensors:
            dataset.append(inp)
        else:
            dataset.append(tokenizer.decode(inp[0], skip_special_tokens=True))

        if len(dataset) % 32 == 0:
            print(f"    Collected {len(dataset)}/{n_samples} samples (skipped {skipped} short docs)...")
        if len(dataset) == n_samples:
            break

    print(f"\n  Collected {len(dataset)} samples from C4")
    print(f"  Skipped {skipped} documents (too short)")
    print(f"  Saving to cache: {cache_file}")
    with open(cache_file, "wb") as f:
        pickle.dump(dataset, f)
    return dataset


def get_wikitext2_calibration_data(tokenizer, n_samples=128, seqlen=2048, seed=42, split="train", cache_dir="./calibration_cache"):
    print(f"\n[WikiText-2 Calibration Data]")
    print(f"  Samples: {n_samples}")
    print(f"  Sequence length: {seqlen} tokens")
    print(f"  Split: {split}")
    print(f"  Seed: {seed}")

    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    cache_file = cache_path / f"wikitext2_calib_n{n_samples}_len{seqlen}_seed{seed}_split{split}.pkl"
    if cache_file.exists():
        print(f"\n  Loading from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"\n  No cache found, downloading from WikiText-2...")
    random.seed(seed)

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    texts = [item["text"] for item in dataset if len(item["text"].strip()) > 0]

    print(f"  Total non-empty texts: {len(texts)}")
    print(f"  Tokenizing and concatenating...")
    all_tokens = []
    for text in texts:
        tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        all_tokens.append(tokens)

    all_tokens = torch.cat(all_tokens, dim=0)
    print(f"  Total tokens: {len(all_tokens)}")

    num_chunks = len(all_tokens) // seqlen
    print(f"  Available {seqlen}-token chunks: {num_chunks}")
    if num_chunks < n_samples:
        print(f"  Warning: Only {num_chunks} chunks available, requested {n_samples}")
        n_samples = num_chunks

    chunk_indices = random.sample(range(num_chunks), n_samples)
    calibration_texts = []
    for idx in chunk_indices:
        start = idx * seqlen
        end = start + seqlen
        chunk_tokens = all_tokens[start:end]
        calibration_texts.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))

    print(f"  Collected {len(calibration_texts)} samples from WikiText-2")
    print(f"  Saving to cache: {cache_file}")
    with open(cache_file, "wb") as f:
        pickle.dump(calibration_texts, f)
    return calibration_texts


def load_calibration_data(dataset_name, tokenizer, n_samples=128, seqlen=2048, seed=42, cache_dir="./calibration_cache"):
    dataset_name = dataset_name.lower()
    if dataset_name == "c4":
        return get_c4_calibration_data(tokenizer, n_samples, seqlen, seed, cache_dir=cache_dir)
    if dataset_name in ["wikitext2", "wikitext"]:
        return get_wikitext2_calibration_data(tokenizer, n_samples, seqlen, seed, split="train", cache_dir=cache_dir)
    raise ValueError(f"Unknown dataset: {dataset_name}. Use 'c4' or 'wikitext2'")
