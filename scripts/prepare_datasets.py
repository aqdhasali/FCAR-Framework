from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

def save_pack(name: str, X: pd.DataFrame, y: pd.Series, A: pd.DataFrame):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    X.to_csv(d / "X.csv", index=False)
    y.to_csv(d / "y.csv", index=False, header=True)
    A.to_csv(d / "A.csv", index=False)
    print(f"[OK] {name}: X={X.shape}, y_pos_rate={y.mean():.3f}, A_cols={list(A.columns)}")

def prep_adult():
    path = RAW / "adult" / "adult.data"
    cols = [
        "age","workclass","fnlwgt","education","education-num","marital-status",
        "occupation","relationship","race","sex","capital-gain","capital-loss",
        "hours-per-week","native-country","income"
    ]
    df = pd.read_csv(path, header=None, names=cols, sep=",", skipinitialspace=True, na_values="?")
    df = df.dropna().reset_index(drop=True)

    y = (df["income"].astype(str).str.strip() == ">50K").astype(int)
    X = df.drop(columns=["income"])
    A = X[["sex", "race"]].copy()

    save_pack("adult", X, y, A)

def prep_german():
    path = RAW / "german" / "german.data"
    cols = [
        "checking_status","duration","credit_history","purpose","credit_amount",
        "savings_status","employment","installment_commitment","personal_status",
        "other_parties","residence_since","property_magnitude","age","other_payment_plans",
        "housing","existing_credits","job","num_dependents","own_telephone","foreign_worker",
        "class"
    ]
    df = pd.read_csv(path, header=None, names=cols, sep=r"\s+")
    # 1 = good, 2 = bad
    y = (df["class"].astype(int) == 1).astype(int)
    X = df.drop(columns=["class"])

    # Sensitive attributes:
    # - age is direct
    # - sex can be derived from personal_status codes (A91/A93/A94 male, A92/A95 female)
    ps = X["personal_status"].astype(str).str.strip()
    sex = ps.map(lambda v: "male" if v in {"A91","A93","A94"} else ("female" if v in {"A92","A95"} else "unknown"))
    A = pd.DataFrame({"age": X["age"], "sex": sex})

    save_pack("german", X, y, A)

def prep_default_credit():
    path = RAW / "default_credit" / "default of credit card clients.xls"
    df = pd.read_excel(path, header=1)
    df.columns = [str(c).strip() for c in df.columns]

    # Drop ID column if present
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Label column is typically: "default payment next month"
    label_col = None
    for c in df.columns:
        if "default" in c.lower() and ("next" in c.lower() or "month" in c.lower()):
            label_col = c
            break
    if label_col is None:
        # fallback: last column
        label_col = df.columns[-1]

    y = df[label_col].astype(int)
    X = df.drop(columns=[label_col])

    # Sensitive attributes commonly used
    A_cols = [c for c in ["SEX", "AGE"] if c in X.columns]
    A = X[A_cols].copy() if A_cols else pd.DataFrame()

    save_pack("default_credit", X, y, A)

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    prep_adult()
    prep_german()
    prep_default_credit()
