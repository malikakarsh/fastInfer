# Tests

Run the current CPU smoke test with:

```bash
PYTHONPATH=.. python -m unittest discover -s tests -v
```

Add CPU baseline tests and GPU kernel correctness tests as their implementations are introduced. The test also runs under pytest when it is installed.
