"""Mutation tests for the offline audit; no GPU or model weights required."""
import unittest

from gguf_reader import TensorInfo
from deepseek41_bundle_audit import payload_bytes, validate_contract, ENGRAM, BACKBONE_TYPES


def fixture():
    # Independent synthetic tensor directory, with the selected build's exact histogram.
    meta = {
        "general.architecture": "deepseek41", "deepseek41.block_count": 40,
        "deepseek41.embedding_length": 5120, "deepseek41.expert_count": 384,
        "deepseek41.expert_used_count": 6, "deepseek41.expert_feed_forward_length": 2304,
        "deepseek41.attention.compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
        "deepseek41.kv_source_layer_ids": [2, 8, 14, 20],
        "deepseek41.index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
        "deepseek41.engram.layer_ids": [1, 14], "deepseek41.engram.num_embeddings": [384006168, 384016682],
        "deepseek41.engram_head_dim": 256, "deepseek41.engram_n_heads": 8,
        "deepseek41.engram_max_ngram_size": 4, "deepseek41.engram.file": ENGRAM,
        "deepseek41.engram.source": "same-checkpoint",
    }
    types = [n for n in ("IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_S") for _ in range(BACKBONE_TYPES[n])]
    tensors = []
    for layer in range(40):
        for role in ("gate", "up", "down"):
            name = f"blk.{layer}.ffn_{role}_exps.weight"
            tensors.append(TensorInfo(name, [2304, 5120, 384] if role == "down" else [5120, 2304, 384],
                                      0, types[len(tensors)], 0))
    for ty in ("F32", "Q5_K", "Q8_0", "BF16", "I32", "I64"):
        for i in range(BACKBONE_TYPES[ty]):
            tensors.append(TensorInfo(f"synthetic.{ty}.{i}", [256], 0, ty, 0))
    em = {"general.architecture": "deepseek41-engram", "deepseek41.engram.source": "same-checkpoint"}
    et = [TensorInfo(f"blk.{l}.engram_embd.weight", [256, r], 20, "IQ4_NL", 0)
          for l, r in [(1, 384006168), (14, 384016682)]]
    return meta, tensors, em, et


class AuditTests(unittest.TestCase):
    def test_exact_contract(self):
        validate_contract(*fixture())

    def test_wrong_architecture(self):
        m, t, em, et = fixture()
        m["general.architecture"] = "deepseek4"
        with self.assertRaisesRegex(ValueError, "architecture"):
            validate_contract(m, t, em, et)

    def test_v4_attention_is_not_v41(self):
        m, t, em, et = fixture()
        m["deepseek41.attention.compress_ratios"][2] = 4
        with self.assertRaisesRegex(ValueError, "compress_ratios"):
            validate_contract(m, t, em, et)

    def test_mixed_quant_change(self):
        m, t, em, et = fixture()
        t[0].type_name = "Q2_0"
        with self.assertRaisesRegex(ValueError, "histogram"):
            validate_contract(m, t, em, et)

    def test_wrong_engram_checkpoint(self):
        m, t, em, et = fixture()
        em["deepseek41.engram.source"] = "other-checkpoint"
        with self.assertRaisesRegex(ValueError, "source mismatch"):
            validate_contract(m, t, em, et)

    def test_wrong_engram_type(self):
        m, t, em, et = fixture()
        et[1].type_name = "Q8_0"
        with self.assertRaisesRegex(ValueError, "shape/type"):
            validate_contract(m, t, em, et)

    def test_exact_block_sizes(self):
        for ty, size in [("IQ1_S", 50), ("IQ1_M", 56), ("IQ2_XXS", 66), ("IQ2_S", 82)]:
            self.assertEqual(payload_bytes(TensorInfo("x", [256, 3], 0, ty, 0)), size * 3)
        self.assertEqual(payload_bytes(TensorInfo("row", [256], 20, "IQ4_NL", 0)), 144)

    def test_row_block_alignment(self):
        with self.assertRaisesRegex(ValueError, "whole blocks"):
            payload_bytes(TensorInfo("x", [128, 2], 19, "IQ1_S", 0))


if __name__ == "__main__":
    unittest.main()
