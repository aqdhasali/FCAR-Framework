# ⚖️ FCAR Framework

**Fairness Constrained Actionable Recourse** — A novel framework that generates equitable algorithmic recourse for individuals denied by AI credit-scoring models.

> Final Year Project — Informatics Institute of Technology / University of Westminster  
> **Author:** Aqdhas Ali (w1954000 / 20210860)  
> **Supervisor:** Ms. Suvetha Suvendran

---

## 📌 Problem

Existing Actionable Recourse (AR) methods minimise individual cost but ignore **systemic fairness** — they place a disproportionately higher burden on protected demographic groups. FCAR addresses this by integrating the **Social Burden (MISOB)** metric into a constrained optimisation engine.

## 🎯 Research Questions

| RQ      | Question                                                                   | Result                                                     |
| ------- | -------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **RQ1** | Can Social Burden be integrated into a constrained optimisation framework? | ✅ MISOB drives asymmetric weight tuning in the MIP solver |
| **RQ2** | Does FCAR reduce burden disparity across groups?                           | ✅ 45–64% gap reduction, all p < 0.01                      |
| **RQ3** | Does FCAR preserve utility (feasibility, flip rate)?                       | ✅ 100% feasibility, 100% flip rate, lower avg burden      |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   FCAR Framework                     │
├──────────┬──────────┬───────────┬────────────────────┤
│  Config  │  Solver  │  Metrics  │   Auto-Tuner       │
│  (YAML)  │  (MIP)   │  (MISOB)  │ (Asymmetric FCAR)  │
├──────────┴──────────┴───────────┴────────────────────┤
│          Scikit-learn Pipeline (LogReg)               │
├──────────────────────────────────────────────────────┤
│  Streamlit Dashboard  │  FastAPI REST API (NFR04)    │
└──────────────────────────────────────────────────────┘
```

| Component       | Technology                                           |
| --------------- | ---------------------------------------------------- |
| Model           | Scikit-learn Logistic Regression + ColumnTransformer |
| Solver          | Pyomo MIP with HiGHS                                 |
| Fairness Metric | MISOB Social Burden (Barrainkua et al., 2025)        |
| Dashboard       | Streamlit                                            |
| API             | FastAPI + Uvicorn                                    |
| Datasets        | Adult Income, German Credit, Default Credit          |

## 📂 Project Structure

```
FCAR/
├── app.py                    # Streamlit dashboard
├── api.py                    # FastAPI REST API (NFR04)
├── requirements.txt
├── src/
│   ├── config/               # YAML configs per dataset
│   ├── metrics/              # Social Burden (MISOB) module
│   ├── modeling/             # Model training utilities
│   └── recourse/             # MIP solver (generic + German-specific)
├── scripts/
│   ├── train_baseline.py     # Train logistic regression models
│   ├── benchmark_ab.py       # A/B: Unconstrained AR vs FCAR
│   ├── auto_fcar_tune.py     # Iterative FCAR weight tuning
│   ├── batch_recourse.py     # Batch recourse generation
│   ├── evaluate_baseline.py  # Model quality evaluation
│   ├── analyze_survey.py     # Survey analysis (87 responses)
│   └── generate_*.py         # Chart generation scripts
├── data/
│   ├── raw/                  # Original datasets + survey
│   ├── processed/            # Cleaned feature matrices
│   └── splits/               # Train/test indices
├── artifacts/
│   ├── models/               # Trained .joblib pipelines
│   └── reports/              # JSONs, CSVs, charts, benchmarks
└── tests/                    # Unit test suite
```

## 🚀 Quick Start

### 1. Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Train Models

```bash
python scripts/prepare_datasets.py
python scripts/make_splits.py
python scripts/train_baseline.py
```

### 3. Run Benchmarks

```bash
python scripts/benchmark_ab.py --dataset german --group_col age_bucket
python scripts/benchmark_ab.py --dataset adult --group_col sex
python scripts/benchmark_ab.py --dataset adult --group_col race
python scripts/benchmark_ab.py --dataset default_credit --group_col SEX
```

### 4. Generate Charts

```bash
python scripts/generate_full_evaluation.py
python scripts/generate_benchmark_charts.py
```

### 5. Launch Dashboard

```bash
streamlit run app.py --server.port 8503
```

### 6. Launch API (NFR04)

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

API documentation available at: `http://localhost:8000/docs`
Deployed Frontend available at: `https://fcar-framework.streamlit.app/`

## 📡 API Endpoints

| Method | Endpoint               | Description                         | Use Case |
| ------ | ---------------------- | ----------------------------------- | -------- |
| `POST` | `/recourse`            | Generate recourse for an individual | UC-101   |
| `GET`  | `/audit/{dataset}`     | Retrieve MISOB audit scores         | UC-103   |
| `GET`  | `/benchmark/{dataset}` | Full A/B benchmark results          | UC-104   |
| `GET`  | `/datasets`            | List available datasets & configs   | —        |
| `GET`  | `/health`              | Health check                        | —        |

### Example: Generate Recourse

```bash
curl -X POST http://localhost:8000/recourse \
  -H "Content-Type: application/json" \
  -d '{"dataset": "german", "test_index": 5, "use_fcar": true}'
```

## 📊 Key Results

| Dataset        | Group      | AR Gap | FCAR Gap | Reduction | p-value |
| -------------- | ---------- | ------ | -------- | --------- | ------- |
| German         | age_bucket | 0.2131 | 0.1023   | **52.0%** | 0.008   |
| Adult          | sex        | 0.0422 | 0.0211   | **50.0%** | < 0.001 |
| Adult          | race       | 0.1234 | 0.0677   | **45.1%** | < 0.001 |
| Default Credit | SEX        | 0.0064 | 0.0023   | **64.3%** | < 0.001 |

## 📚 References

- Barrainkua, A. et al. (2025). _Who Pays for Fairness? Rethinking Recourse under Social Burden_
- Wang, Y. et al. (2024). _Achieving Fairness via Actionable Recourse_
- Ustun, B. et al. (2019). _Actionable Recourse in Linear Classification_
- Yetukuri, J. (2024). _Towards Socially Acceptable Algorithmic Models_

## 📄 License

This project is submitted as academic coursework for the BSc (Hons) Computer Science programme at IIT/University of Westminster.
