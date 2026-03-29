# FCAR Test Suite

Comprehensive unit tests for the FCAR framework — **NFR03 compliance: 95% test coverage**.

## Quick Start

```powershell
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-config=.coveragerc --cov-report=term

# Run specific test file
pytest tests/test_config_loader.py -v

# Run specific test class
pytest tests/test_social_burden.py::TestComputeSocialBurden -v
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures (toy pipeline, configs, sample data)
├── test_config_loader.py    # All config_loader.py functions (53 tests)
├── test_social_burden.py    # MISOB metric functions (28 tests)
├── test_preprocess.py       # ColumnTransformer builder (9 tests)
├── test_recourse_mip.py     # MIP solver + helpers (22 tests)
└── test_api.py              # FastAPI endpoints (15 tests)
```

## Coverage Summary

| Module                                 | Coverage | Notes                           |
| -------------------------------------- | -------- | ------------------------------- |
| `src/config/config_loader.py`          | **95%**  | All public helpers tested       |
| `src/metrics/social_burden.py`         | **100%** | Full MISOB metric coverage      |
| `src/modeling/preprocess.py`           | **100%** | Preprocessor builder            |
| `src/recourse/generic_recourse_mip.py` | **92%**  | MIP solver + extraction helpers |
| **Overall (excluding deprecated)**     | **95%**  | ✓ NFR03 met                     |

_Deprecated modules (`german_recourse.py`, `german_recourse_mip.py`) are excluded from coverage — replaced by `generic_recourse_mip.py`._

## Test Philosophy

- **No real artefacts**: Tests use lightweight synthetic data + fitted toy pipelines (defined in `conftest.py`) so they run in seconds without needing actual model files.
- **Comprehensive coverage**: Every public function in `src/` has at least one test; complex functions (MIP solver) have 10+ integration tests.
- **Fast execution**: Full suite runs in ~7 seconds.
- **Deterministic**: All fixtures use fixed random seeds — tests are reproducible.

## Key Fixtures

### From `conftest.py`

- **`german_config`** — Minimal German-like config dict matching `german.yaml` structure
- **`toy_data`** — 120-row synthetic DataFrame mimicking German Credit features
- **`toy_pipeline`** — Fitted `Pipeline([("pre", ColumnTransformer), ("clf", LogisticRegression)])` matching the structure the MIP solver expects
- **`toy_train_test`** — 80/20 train/test split
- **`rejected_applicant`** — A single rejected applicant from the toy test set (for recourse tests)

## Test Examples

### Config Loader

```python
def test_load_german():
    cfg = load_dataset_config("german")
    assert cfg["dataset"] == "german"
    assert "mutable_numeric" in cfg
```

### MISOB Metrics

```python
def test_compute_social_burden():
    recourse_df = pd.DataFrame({...})
    sb = compute_social_burden(recourse_df, "group")
    assert "social_burden" in sb.columns
```

### MIP Solver

```python
def test_feasible_solution_flips_decision(toy_pipeline, rejected_applicant):
    _, x0, p0 = rejected_applicant
    x_cf, slack = solve_recourse_mip(toy_pipeline, X_train, x0, config)
    if slack == 0.0:
        p1 = toy_pipeline.predict_proba(pd.DataFrame([x_cf]))[:, 1][0]
        assert p1 >= 0.5  # Decision flipped
```

### API Endpoints

```python
def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
```

## Test Categories

### 1. **Unit Tests** (Helpers)

- `_full_logit()` — Logit extraction from pipeline
- `_extract_numeric_beta()` — Coefficient extraction
- `_extract_cat_weights()` — One-hot weights
- `_rank_map_for()` — Categorical ordering
- All config getters (18 functions)
- All MISOB functions (6 functions)

### 2. **Integration Tests** (MIP Solver)

- ✅ Decision flips when slack=0
- ✅ Logit constraint satisfied: `logit + slack ≥ margin`
- ✅ Plausibility: duration/credit only decrease
- ✅ Plausibility: max_decrease (48 months) respected
- ✅ Plausibility: max_rel_decrease (80%) respected
- ✅ Monotonic categorical: no worsening (A11→A14 ok, A14→A11 forbidden)
- ✅ max_cat_changes: at most 1 categorical change
- ✅ Integer features stay integer
- ✅ Immutable features unchanged
- ✅ FCAR weights change objective, not constraints
- ✅ Within training data bounds

### 3. **API Tests** (FastAPI)

- `/health` — Health check
- `/datasets` — List configs
- `/recourse` — Generate recourse (with/without FCAR)
- `/audit/{dataset}` — MISOB audit scores
- `/benchmark/{dataset}` — A/B benchmark results
- Error handling (invalid dataset, out-of-range index)

## Running Specific Tests

```powershell
# All config loader tests
pytest tests/test_config_loader.py

# Only social burden tests
pytest tests/test_social_burden.py

# Only MIP solver feasibility check
pytest tests/test_recourse_mip.py::TestSolveRecourseMIP::test_feasible_solution_flips_decision

# Verbose output
pytest -v

# Stop on first failure
pytest -x

# Show local variables on failure
pytest -l
```

## CI/CD Integration

Add to GitHub Actions workflow:

```yaml
- name: Run tests with coverage
  run: |
    pytest --cov=src --cov-config=.coveragerc --cov-report=term --cov-report=xml

- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v3
```

## Extending Tests

To add tests for a new module `src/foo/bar.py`:

1. Create `tests/test_bar.py`
2. Import: `from src.foo.bar import function_to_test`
3. Write test classes: `class TestFunctionName:`
4. Use fixtures from `conftest.py` or add new ones
5. Run: `pytest tests/test_bar.py`

## NFR03 Compliance

> **NFR03**: Achieve 95% unit test coverage for all algorithmic components.

**Status**: ✅ **PASSED** — 95% coverage on active modules.

```
Name                                   Stmts   Miss  Cover
----------------------------------------------------------
src/config/config_loader.py               60      3    95%
src/metrics/social_burden.py              62      0   100%
src/modeling/preprocess.py                11      0   100%
src/recourse/generic_recourse_mip.py     198     15    92%
----------------------------------------------------------
TOTAL                                    331     18    95%
```

---

## Notes

- **Skipped tests**: 1 test (`test_valid_recourse_german`) skips when no rejected applicants exist in the toy data — this is intentional.
- **Deprecated modules**: `german_recourse.py` and `german_recourse_mip.py` have 0% coverage because they're legacy code replaced by `generic_recourse_mip.py`. They're excluded from the coverage calculation.
- **API tests**: Some API tests (audit, benchmark) skip if benchmark results don't exist locally — they still pass in CI after running benchmarks.
