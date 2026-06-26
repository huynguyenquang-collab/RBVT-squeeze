from setuptools import setup, find_packages

setup(
    name="leanquant",
    version="0.1.1",
    author="Tianyi Zhang",
    author_email="tonyzhang617@gmail.com",
    description="The inference kernels for LeanQuant models.",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.38",
        "accelerate",
        "safetensors",
        "torch",
    ],
    extras_require={
        "cuda11": [
            "cupy-cuda11x",
        ],
        "cuda12": [
            "cupy-cuda12x",
        ],
    },
)
