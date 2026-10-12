"""Check that instruction diagnostics distinguish generated instructions from comments."""

import unittest

from fastinfer.kernels.ptx import summarize_ptx


class PtxSummaryTest(unittest.TestCase):
    def test_matrix_instructions_and_predication(self) -> None:
        result = summarize_ptx("""
            // mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 is a comment
            /* fma.rn.f32 is also a comment */
            .target sm_80
            mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%r1}, {%r2}, {%r3}, {%r4};
            @%p1 mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%r1}, {%r2}, {%r3}, {%r4};
            @!%p2 fma.rn.f32 %f1, %f2, %f3, %f4;
        """)
        self.assertEqual(result["matrix_instruction_statements"], 2)
        self.assertEqual(result["fma_instruction_statements"], 1)

    def test_fma_only_is_distinct_from_unavailable_ptx(self) -> None:
        result = summarize_ptx("fma.rn.f32 %f1, %f2, %f3, %f4;")
        self.assertTrue(result["available"])
        self.assertEqual(result["matrix_instruction_statements"], 0)
        self.assertEqual(result["fma_instruction_statements"], 1)
        self.assertEqual(summarize_ptx(None), {"available": False})

    def test_matrix_instruction_families(self) -> None:
        result = summarize_ptx("""
            wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 {%f1}, {%r1}, {%r2}, {%f2};
            wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%f1}, %rd1, %rd2, 1, 1, 1, 0, 0;
            tcgen05.mma.cta_group::1.kind::f16 [%r1], %rd1, %rd2, %r2, %p1;
        """)
        self.assertEqual(result["matrix_instruction_statements"], 3)
        self.assertEqual(result["fma_instruction_statements"], 0)


if __name__ == "__main__":
    unittest.main()
