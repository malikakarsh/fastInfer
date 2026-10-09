# Benchmarks

```bash
PYTHONPATH=.. python benchmarks/matmul_baseline.py
```

The script saves JSON in `results/raw/`. It defaults to a small `1 x 256` by `256 x 256` float16 matmul, with 10 warmups and 30 timed repetitions. Use `--m`, `--k`, `--n`, and `--dtype` to change the workload.
