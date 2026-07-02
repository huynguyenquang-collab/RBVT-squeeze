"""
Calibration data loading utilities for RBVTQuant.

This is a local copy so RBVTQuant remains self-contained and does not depend on
NCCQuant at runtime.
"""

from __future__ import annotations

import os
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
    cache_path.mkdir(parents=True, exist_ok=True)

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


def _redpajama_text(example):
    for key in ("text", "raw_content", "content"):
        value = example.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def get_redpajama_calibration_data(
    tokenizer,
    n_samples=1024,
    seqlen=4096,
    seed=0,
    split="train",
    return_tensors=False,
    cache_dir="./calibration_cache",
):
    dataset_name = os.environ.get("REDPAJAMA_DATASET", "ZengXiangyu/RedPajama-Data-1T-Sample")
    dataset_config = os.environ.get("REDPAJAMA_CONFIG") or None
    streaming = os.environ.get("REDPAJAMA_STREAMING", "0") == "1"

    print(f"\n[RedPajama Calibration Data]")
    print(f"  Dataset: {dataset_name}")
    print(f"  Config: {dataset_config or '<none>'}")
    print(f"  Split: {split}")
    print(f"  Samples: {n_samples}")
    print(f"  Sequence length: {seqlen} tokens")
    print(f"  Seed: {seed}")

    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    safe_dataset = dataset_name.replace("/", "__")
    safe_config = dataset_config or "default"
    cache_file = cache_path / (
        f"redpajama_{safe_dataset}_{safe_config}_n{n_samples}_len{seqlen}_seed{seed}_split{split}_"
        f"tensors{return_tensors}.pkl"
    )
    if cache_file.exists():
        print(f"\n  Loading from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("\n  No cache found, loading RedPajama...")
    rng = random.Random(seed)
    if dataset_config is None:
        data = load_dataset(dataset_name, split=split, streaming=streaming)
    else:
        data = load_dataset(dataset_name, dataset_config, split=split, streaming=streaming)

    calibration_texts = []
    skipped = 0
    for item in data:
        text = _redpajama_text(item)
        if not text:
            skipped += 1
            continue
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        if encoded.numel() < seqlen:
            skipped += 1
            continue
        start = rng.randint(0, encoded.numel() - seqlen)
        chunk = encoded[start : start + seqlen]
        if return_tensors:
            calibration_texts.append(chunk.unsqueeze(0).to(torch.long))
        else:
            calibration_texts.append(tokenizer.decode(chunk, skip_special_tokens=True))
        if len(calibration_texts) % 32 == 0:
            print(f"    Collected {len(calibration_texts)}/{n_samples} samples (skipped {skipped})...")
        if len(calibration_texts) >= n_samples:
            break

    if len(calibration_texts) != n_samples:
        raise RuntimeError(
            f"Collected only {len(calibration_texts)} RedPajama samples; expected {n_samples}. "
            "Try REDPAJAMA_STREAMING=1 or another REDPAJAMA_DATASET/REDPAJAMA_CONFIG."
        )

    print(f"  Collected {len(calibration_texts)} samples from RedPajama")
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
    if dataset_name in ["redpajama", "red_pajama"]:
        return get_redpajama_calibration_data(tokenizer, n_samples, seqlen, seed, split="train", cache_dir=cache_dir)
    raise ValueError(f"Unknown dataset: {dataset_name}. Use 'c4', 'wikitext2', or 'redpajama'")
