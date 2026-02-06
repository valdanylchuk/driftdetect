# LendingClub Example

Demonstrates drift detection on a loan triage API using LendingClub data.

## Files

- `api.py` - FastAPI app with `@check_drift` decorator
- `models.py` - Pydantic models and mock ML model
- `test_run.py` - Demo script showing drift detection
- `generate_baseline.py` - Script to create baseline.json from CSV
- `data/lendingclub_2015_sample.csv.gz` - Reference data (1% sample, ~4k rows)
- `data/lendingclub_2017_sample.csv.gz` - Test data with drift (1% sample)
- `baseline.json` - Pre-generated baseline from 2015 data

## Quick Start

```bash
# Install dependencies
pip install driftdetect fastapi uvicorn

# Run the API
uvicorn examples.lendingclub.api:app --port 8000

# Test endpoint
curl -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"annual_inc": 75000, "dti": 18.5, "loan_amnt": 15000, "int_rate": 11.99}'
```

## Demo: Detecting Drift

Run the test script to send requests to the API and see drift detection logs:

```bash
python examples/lendingclub/test_run.py
```

Output:

```
driftdetect - WARNING - Drift ALARM in ['annual_inc', 'dti', 'loan_amnt', 'int_rate'] after 50 requests: {'annual_inc': {'psi': 0.5026, ...}, 'dti': {'psi': 0.1764, ...}, 'loan_amnt': {'psi': 0.1239, ...}, 'int_rate': {'psi': 0.3146, ...}}
driftdetect - WARNING - Drift ALARM in ['annual_inc', 'dti', 'loan_amnt', 'int_rate'] after 100 requests: ...
driftdetect - WARNING - Drift ALARM in ['annual_inc', 'dti', 'loan_amnt', 'int_rate'] after 150 requests: ...
Sending 200 requests to /triage...
Drift logs will appear below:

Done.
```

The 2017 data triggers **ALARM** on multiple features - the distribution shifted between 2015 (baseline) and 2017.

## Regenerate Baseline

```bash
python generate_baseline.py data/lendingclub_2015_sample.csv.gz baseline.json
```

## Data

The sample datasets are 1% random samples of [LendingClub loan data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) for two years:
- **2015**: Reference distribution (baseline)
- **2017**: Shows drift over 2 years

Features: `annual_inc`, `dti`, `loan_amnt`, `int_rate`
