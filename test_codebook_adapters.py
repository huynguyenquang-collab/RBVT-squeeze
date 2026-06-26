"""Small correctness tests for the optional codebook adapters."""

from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.nn as nn

from quantizers.base_codebook import CodebookContext
from quantizers.codebook_factory import get_codebook
from quantizers.codebook_store import CodebookStore
from quantizers.hessian_store import HessianStore
from quantizers.rbvt import apply_rbvt
from quantizers.squeezellm_collector import collect_squeezellm_codebooks
from quantizers.sparse_residual_store import SparseResidualStore


class CodebookAdapterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.weight = torch.randn(5, 19)
        self.mean = torch.randn(19) * 0.1
        self.variance = torch.rand(19) + 0.1

    def test_quant_result_is_rbvt_compatible(self):
        for name in ("leanquant", "squeezellm"):
            for bits in (3, 4):
                with self.subTest(name=name, bits=bits):
                    codebook = get_codebook(
                        name,
                        bits,
                        group_size=-1,
                    )
                    centers = torch.randn(5, 1, 2**bits).sort(dim=-1).values
                    codebook.set_context(
                        CodebookContext(
                            precomputed_centers=centers,
                        )
                    )
                    result = codebook.quantize(self.weight, row_chunk=3)

                    self.assertEqual(result.W_dequant.shape, self.weight.shape)
                    self.assertEqual(result.indices.shape, self.weight.shape)
                    self.assertEqual(result.block_codebooks.shape, (5, 1, 2**bits))
                    self.assertEqual(result.block_size, 19)
                    self.assertTrue(
                        torch.all(
                            result.block_codebooks[:, :, 1:]
                            >= result.block_codebooks[:, :, :-1]
                        )
                    )

                    rbvt_weight, stats = apply_rbvt(
                        W_fp=self.weight,
                        qres=result,
                        mu=self.mean,
                        sigma_ii=self.variance,
                        row_chunk=2,
                    )
                    self.assertEqual(rbvt_weight.shape, self.weight.shape)
                    self.assertGreaterEqual(stats.flips, 0)

    def test_adapter_requires_upstream_centers(self):
        codebook = get_codebook("squeezellm", 3)
        codebook.set_context(CodebookContext())
        with self.assertRaisesRegex(RuntimeError, "upstream repository"):
            codebook.quantize(self.weight)

    def test_hessian_store_has_tracks_cached_files(self):
        with TemporaryDirectory() as directory:
            store = HessianStore(directory)
            store.initialize({"bits": 3})
            self.assertFalse(store.has("model.layers.0.self_attn.q_proj"))

            store.put(
                "model.layers.0.self_attn.q_proj",
                torch.eye(3),
            )
            self.assertTrue(store.has("model.layers.0.self_attn.q_proj"))

    def test_sparse_residual_store_round_trip(self):
        with TemporaryDirectory() as directory:
            store = SparseResidualStore(directory)
            store.initialize({"mode": "hybrid"})
            residual = torch.zeros(3, 5)
            residual[0, 2] = 1.25
            residual[2, 4] = -0.75
            store.put("model.layers.0.self_attn.q_proj", residual)
            store.mark_complete()

            restored = store.get("model.layers.0.self_attn.q_proj")
            self.assertTrue(torch.equal(restored, residual))

    def test_rbvt_candidate_mask_excludes_sparse_positions(self):
        codebook = get_codebook("squeezellm", 3)
        centers = torch.linspace(-1.0, 1.0, 8).reshape(1, 1, 8).repeat(5, 1, 1)
        codebook.set_context(CodebookContext(precomputed_centers=centers))
        result = codebook.quantize(self.weight, row_chunk=3)
        candidate_mask = torch.zeros_like(self.weight, dtype=torch.bool)

        masked_weight, stats = apply_rbvt(
            W_fp=self.weight,
            qres=result,
            mu=self.mean,
            sigma_ii=self.variance,
            candidate_mask=candidate_mask,
            row_chunk=2,
        )
        self.assertEqual(stats.flips, 0)
        self.assertTrue(torch.equal(masked_weight, result.W_dequant.float()))

    def test_squeezellm_dense_only_does_not_extract_sparse_values(self):
        class FakeLayer:
            def __init__(self):
                self.linear = nn.Linear(6, 3, bias=False)

            def get_submodule(self, name):
                self.assert_name(name)
                return self.linear

            @staticmethod
            def assert_name(name):
                if name != "linear":
                    raise AssertionError(name)

        class FakeModel:
            def __init__(self):
                self.model = type("Inner", (), {"layers": [FakeLayer()]})()

        class FakeSensitivity:
            path = "fisher"

            @staticmethod
            def get(_name, shape):
                return torch.ones(shape)

        class FakeModelParse:
            @staticmethod
            def get_sequential(_model_type):
                return ["linear"]

            @staticmethod
            def get_module_names(_model_type):
                return ["linear"]

        class SerialPool:
            def __init__(self, _workers):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            @staticmethod
            def imap(function, tasks):
                return map(function, tasks)

        def fake_kmeans(task):
            row, _sample_weight, clusters = task
            centers = torch.linspace(
                float(row.min()),
                float(row.max()),
                clusters,
            ).numpy()
            return centers, None

        with TemporaryDirectory() as directory:
            store = CodebookStore(directory)
            with (
                patch(
                    "quantizers.squeezellm_collector.load_squeezellm_kmeans",
                    return_value=fake_kmeans,
                ),
                patch(
                    "quantizers.squeezellm_collector.load_squeezellm_model_parse",
                    return_value=FakeModelParse,
                ),
                patch(
                    "quantizers.squeezellm_collector.load_squeezellm_remove_outliers"
                ) as remove_outliers,
                patch("quantizers.squeezellm_collector.Pool", SerialPool),
            ):
                collect_squeezellm_codebooks(
                    model=FakeModel(),
                    sensitivity_store=FakeSensitivity(),
                    store=store,
                    sparse_store=None,
                    bits=4,
                    mode="dense-only",
                )

            remove_outliers.assert_not_called()
            self.assertTrue(store.complete)
            self.assertEqual(store.metadata["mode"], "dense-only")
            self.assertEqual(
                store.get("model.layers.0.linear").shape,
                (3, 1, 16),
            )


if __name__ == "__main__":
    unittest.main()
