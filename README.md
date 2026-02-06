# checkdrift

One-line drift detection for ML APIs. Like Pydantic or a rate limiter, but for data drift.

Just add @check_drift decorator to your FastAPI endpoint:

```python
@app.post("/predict")
@check_drift(baseline="baseline.json")
async def predict(application: LoanApplication):
    return model.predict(application)
```

## Installation

```bash
pip install checkdrift
```

## What It Does

- Maintains a sliding window of recent requests
- Computes drift metrics every N requests (default: 50)
- Logs warnings when drift is detected
- Minimal impact on your endpoint response (about 1ms in my tests)

Uses PSI (Population Stability Index) and KS test - industry standards from banking.

## Baseline Format

```json
{"distributions": {"feature1": [1.0, 2.0, ...], "feature2": [...]}}
```

See [examples/lendingclub](examples/lendingclub) for a complete example with sample data.

## Options

```python
@check_drift(
    baseline="baseline.json",
    window_size=100,       # Sliding window size
    check_interval=50,     # Check every N requests
    on_drift=my_callback,  # Optional callback
)
```

## License

MIT
