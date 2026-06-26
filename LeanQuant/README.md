<h1 align="center">LeanQuant</h1>
<p align="center">
  <img src="https://raw.githubusercontent.com/LeanModels/LeanQuant/refs/heads/master/assets/LeanLlama.jpg" alt="A lean and mean llama." width="512">
</p>
<p align="center">
    <strong>ICLR 2025</strong> | <em>Accurate and Scalable LLM Quantization with Loss-error-aware Grid</em>
</p>
<p align="center">
    🚀 Quantizes a <strong>70B model</strong> on <strong>a single 24GB GPU in 4 hours</strong><br>
    ⚡ Quantizes a <strong>405B model</strong> on <strong>two 48GB GPUs in 24 hours</strong>
</p>

---

## 🔍 What is LeanQuant?

**LeanQuant** is an efficient large language model (LLM) quantization framework that minimizes quality loss while maximizing computational and memory efficiency. It introduces a **loss-error-aware quantization grid** that preserves outliers in the inverse Hessian—achieving superior model quality without extra storage or inference overhead.

📄 Read the full paper: [arXiv](https://arxiv.org/abs/2407.10032)

---

## 🚀 Why LeanQuant?

✅ **Scalable Quantization** – Handles ultra-large models with one or two GPUs

✅ **Efficient Inference** – Optimized 4-bit CUDA kernels for fast and memory-efficient execution

✅ **High Accuracy** – Compares favorably against state-of-the-art quantization methods

✅ **Versatile** – Supports non-uniform and affine quantization formats

✅ **Minimal Dependencies** – Easy installation and setup

✅ **Broad Compatibility** – Works on most CUDA GPUs, and supports multi-GPU distributed inference

---

## 🛠️ Quick Start

1. Make sure you have a Linux environment, a CUDA-enabled GPU, and Python and PyTorch installed.
2. Install our pip package.
```bash
# For CUDA 11.x
pip install leanquant[cuda11]

# For CUDA 12.x
pip install leanquant[cuda12]
```
3. Download a LeanQuant model from the Model Zoo below or from [our HuggingFace page](https://huggingface.co/LeanQuant). Each downloaded model is a `.safetensors` file. For example, download the 4-bit `Llama-3.1-8B-Instruct` from [this link](https://huggingface.co/LeanQuant/Llama-3.1-8B-Instruct-nu-4bit/resolve/main/model.safetensors) or with the command `wget https://huggingface.co/LeanQuant/Llama-3.1-8B-Instruct-nu-4bit/resolve/main/model.safetensors`.
4. The model can now be loaded for inference using the following script:
```python
from leanquant import LeanQuantModelForCausalLM

model = LeanQuantModelForCausalLM.from_pretrained(
    "<base-model-name>",
    "<path-to-model-safetensors>",
    bits=<bit-width>,
    device_map="auto"
)
```

**A Complete Example:** The following script shows how to run inference with a 4-bit `Llama-3.1-8B-Instruct` (with the model downloaded to `./model.safetensors`):
```python
import torch
from leanquant import LeanQuantModelForCausalLM
from transformers import AutoTokenizer

### Load model and tokenizer
base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = LeanQuantModelForCausalLM.from_pretrained(
    base_model_name,
    "./model.safetensors",
    bits=4,
    device_map="auto"
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

### Tokenize prompt
prompt = [
    {"role": "system", "content": "You are a helpful assistant, that responds as a pirate."},
    {"role": "user", "content": "What is quantization for deep learning models?"},
]
inputs = tokenizer.apply_chat_template(
    prompt,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

### Run generation and decode generated tokens
with torch.no_grad():
    output = model.generate(**inputs, do_sample=True, max_new_tokens=256)

generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
print(generated_text)
```

## 🦁 Model Zoo

Explore our collection of pre-quantized models for efficient deployment.

| Base Model Name                                 | Quantized Bits| Download Link |
|-------------------------------------------------|---------------|----------------------------------------------------------------|
| meta-llama/Meta-Llama-3-8B                      | 4-bit         | [Download](https://huggingface.co/LeanQuant/Meta-Llama-3-8B-nu-4bit/resolve/main/model.safetensors) |
| meta-llama/Meta-Llama-3-8B                      | 3-bit         | [Download](https://huggingface.co/LeanQuant/Meta-Llama-3-8B-nu-3bit/resolve/main/model.safetensors) |
| meta-llama/Meta-Llama-3-8B                      | 2-bit         | [Download](https://huggingface.co/LeanQuant/Meta-Llama-3-8B-nu-2bit/resolve/main/model.safetensors) |
| meta-llama/Llama-2-7b-hf                        | 4-bit         | [Download](https://huggingface.co/LeanQuant/Llama-2-7b-nu-4bit/resolve/main/model.safetensors) |
| meta-llama/Llama-2-7b-hf                        | 3-bit         | [Download](https://huggingface.co/LeanQuant/Llama-2-7b-nu-3bit/resolve/main/model.safetensors) |
| meta-llama/Llama-2-7b-hf                        | 2-bit         | [Download](https://huggingface.co/LeanQuant/Llama-2-7b-nu-2bit/resolve/main/model.safetensors) |
| meta-llama/Llama-2-13b-hf                       | 4-bit         | [Download](https://huggingface.co/LeanQuant/Llama-2-13b-nu-4bit/resolve/main/model.safetensors) |
| meta-llama/Llama-2-13b-hf                       | 3-bit         | [Download](https://huggingface.co/LeanQuant/Llama-2-13b-nu-3bit/resolve/main/model.safetensors) |
| meta-llama/Llama-2-13b-hf                       | 2-bit         | [Download](https://huggingface.co/LeanQuant/Llama-2-13b-nu-2bit/resolve/main/model.safetensors) |
| mistralai/Mistral-7B-v0.1                       | 4-bit         | [Download](https://huggingface.co/LeanQuant/Mistral-7B-v0.1-nu-4bit/resolve/main/model.safetensors) |
| mistralai/Mistral-7B-v0.1                       | 3-bit         | [Download](https://huggingface.co/LeanQuant/Mistral-7B-v0.1-nu-3bit/resolve/main/model.safetensors) |
| mistralai/Mistral-7B-v0.1                       | 2-bit         | [Download](https://huggingface.co/LeanQuant/Mistral-7B-v0.1-nu-2bit/resolve/main/model.safetensors) |
| huggyllama/llama-13b                            | 4-bit         | [Download](https://huggingface.co/LeanQuant/llama-13b-nu-4bit/resolve/main/model.safetensors) |
| huggyllama/llama-13b                            | 3-bit         | [Download](https://huggingface.co/LeanQuant/llama-13b-nu-3bit/resolve/main/model.safetensors) |
| huggyllama/llama-13b                            | 2-bit         | [Download](https://huggingface.co/LeanQuant/llama-13b-nu-2bit/resolve/main/model.safetensors) |
| meta-llama/Meta-Llama-3-8B-Instruct             | 4-bit         | [Download](https://huggingface.co/LeanQuant/Meta-Llama-3-8B-Instruct-nu-4bit/resolve/main/model.safetensors) |
| meta-llama/Llama-3.1-8B                         | 4-bit         | [Download](https://huggingface.co/LeanQuant/Llama-3.1-8B-nu-4bit/resolve/main/model.safetensors) |
| meta-llama/Llama-3.1-8B-Instruct                | 4-bit         | [Download](https://huggingface.co/LeanQuant/Llama-3.1-8B-Instruct-nu-4bit/resolve/main/model.safetensors) |
| meta-llama/Llama-3.1-70B                        | 4-bit         | [Download](https://huggingface.co/LeanQuant/Llama-3.1-70B-nu-4bit/resolve/main/model.safetensors) |
| meta-llama/Llama-3.3-70B-Instruct               | 4-bit         | [Download](https://huggingface.co/LeanQuant/Llama-3.3-70B-Instruct-nu-4bit/resolve/main/model.safetensors) |

🚀 More models coming soon!

## 📌 How to Quantize and Evaluate a Model

Follow these steps to quantize and evaluate a large language model.

### Requirements

- At least **one CUDA-enabled GPU** is required for quantization and evaluation.  
- A **Linux environment** is recommended.

### Setup

1. **Clone the Repository**  
```bash
git clone https://github.com/LeanModels/LeanQuant.git
cd LeanQuant
```

2. **[Optional] Create a Conda Environment**
```bash
conda create -n leanquant python=3.10
conda activate leanquant
```

3. **Install Dependencies**

- Install [PyTorch](https://pytorch.org/get-started/locally/).
- Install [CuPy](https://docs.cupy.dev/en/stable/install.html) based on your CUDA version.
```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x
```
4. **Install Additional Requirements**

```bash
pip install -r requirements.txt
```

### Quantizing Models

We currently support the Llama and Mistral family models (non-VLM, non-MOE). To quantize a model using **LeanQuant**, run the following command:

```bash
python llama.py <huggingface-model-name-or-path> <calibration-dataset-name> \
    --new-eval \
    --wbits 4 \
    --nsamples 128 \
    --true-sequential --act-order \
    --percdamp 0.1 \
    --exponent 4 \
    --save_path <quantized-model-name>.safetensors
```

**Parameter Explanation:**

| Parameter                     | Description |
|--------------------------------|-------------|
| `<huggingface-model-name-or-path>` | The HuggingFace model name or local model path to quantize. Example: `meta-llama/Llama-3.1-8B-Instruct`. |
| `<calibration-dataset-name>`  | Calibration dataset for quantization. Choices: `wikitext2`, `ptb`, `c4`, `c4-new` (recommended: `c4-new`). |
| `--new-eval`                  | Enables new evaluation mode for perplexity testing. |
| `--wbits`                     | Bit-width for quantization. Choices: `4`, `3`, or `2`. |
| `--nsamples`                  | Number of calibration samples. Recommended: `128`, `256`, or `512`. |
| `--true-sequential` & `--act-order` | Improves quantized model quality. Recommended to enable. |
| `--percdamp`                  | Dampening applied to the Hessian. Recommended: `0.1` or `0.01`. |
| `--exponent`                  | Strength parameter for preserving outliers in quantization (`p` in the paper). Recommended: `3` or `4`. |
| `--save_path`                 | Path and filename to save the quantized model. |

**Example:**

To quantize `meta-llama/Llama-3.1-8B-Instruct` to 4-bit precision, run:

```bash
python llama.py meta-llama/Llama-3.1-8B-Instruct c4-new \
    --new-eval \
    --wbits 4 \
    --nsamples 128 \
    --true-sequential --act-order \
    --percdamp 0.1 \
    --exponent 4 \
    --save_path Llama-3.1-8B-Instruct-4bit.safetensors
```

### Quantizing Very Large Models

LeanQuant enables efficient quantization of Llama-3.1-405B using either two 48GB GPUs or a single 80GB GPU. Use the following command to quantize the 405B model.
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;CUDA_VISIBLE_DEVICES=0,1 python llama.py meta-llama/Llama-3.1-405B-Instruct \
    c4-new --new-eval \
    --wbits 4 --nsamples 64 \
    --true-sequential --act-order \
    --percdamp 0.1 \
    --exponent 4.0 \
    --offload_threshold 53248 \
    --save_path Llama-3.1-405B-Instruct-4bit.safetensors
```
The `--offload_threshold` argument helps prevent out-of-memory errors by offloading Hessian matrix computation to the second GPU. `--offload_threshold 53248` offloads Hessian computation to the second GPU if the matrix dimension is greater than or equal to 53248. If your GPU has enough memory (80GB or above), you can disable Hessian offloading and use a single GPU with `--offload_threshold 1000000`.

### Evaluating Quantized Models  

To evaluate a quantized model using **LeanQuant**, run the following command:

```bash
python eval_quantized.py \
    --base_model_name_or_path <base-model-name-or-path> \
    --leanquant_path <path-to-quantized-model-safetensors> \
    --bits 4 \
    --tasks mmlu ai2_arc lambada hellaswag winogrande piqa \
    --eval_batch_size 4
```

**Parameter Explanation:**

| Parameter                      | Description |
|--------------------------------|-------------|
| `--base_model_name_or_path`  | Name or path of the original Hugging Face model that was quantized. |
| `--leanquant_path`           | Path to the `.safetensors` file of the quantized model. |
| `--bits`                     | Bit-width of the quantized model. Choices: `4`, `3`, `2`. |
| `--tasks`                    | Benchmark tasks from `lm-eval`. |
| `--eval_batch_size`          | Batch size for evaluation. Adjust based on available GPU memory. |

**Example:**

To evaluate a 4-bit quantized `meta-llama/Llama-3.1-8B-Instruct` on `mmlu` and `hellaswag`, run:  

```bash
python eval_quantized.py \
    --base_model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --leanquant_path Llama-3.1-8B-Instruct-4bit.safetensors \
    --bits 4 \
    --tasks mmlu hellaswag \
    --eval_batch_size 4
```

# Acknowledgements

This code repository is based on [GPTQ](https://github.com/IST-DASLab/gptq). We thank the authors for their wonderful work.

# Citation

If you found our work useful or interesting, please kindly cite us:
```
@inproceedings{
    zhang2025leanquant,
    title={LeanQuant: Accurate and Scalable Large Language Model Quantization with Loss-error-aware Grid},
    author={Tianyi Zhang and Anshumali Shrivastava},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=ISqx8giekS}
}
```