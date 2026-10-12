# Tests

Run the current CPU smoke test with:

```bash
PYTHONPATH=.. python -m unittest discover -s tests -v
```

Add CPU baseline tests and GPU kernel correctness tests as their implementations are introduced. The test also runs under pytest when it is installed.

The Triton matmul tests run on a CUDA machine with Triton installed. They cover a known result, partial output tiles, and K dimensions spanning multiple 32-element blocks. They are skipped on a CPU-only machine. Run the full suite on Kaggle before benchmarking the kernel.
