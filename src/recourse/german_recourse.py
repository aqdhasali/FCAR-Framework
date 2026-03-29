import numpy as np
import pandas as pd

from pyomo.environ import (
    ConcreteModel, Var, Reals, NonNegativeReals, Integers,
    Objective, ConstraintList, minimize, value
)
from pyomo.contrib.appsi.solvers.highs import Highs


def _extract_linear_terms_from_pipeline(pipe, X_row: pd.Series):
    """
    Returns:
      const_term: float   = intercept + contributions of all fixed features
      beta: dict[col,float] coefficients for ORIGINAL numeric columns (unscaled)

    Assumes:
      pipe = Pipeline([("pre", ColumnTransformer), ("clf", LogisticRegression)])
      numeric transformer includes StandardScaler(with_mean=False)
    """
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    feat_names = pre.get_feature_names_out()
    w = clf.coef_.ravel()
    b = float(clf.intercept_.ravel()[0])

    coef = dict(zip(feat_names, w))

    # transformed values for this row
    z = pre.transform(pd.DataFrame([X_row]))
    z = np.asarray(z).ravel()

    # numeric columns and scales
    num_cols = list(pre.transformers_[0][2])  # ("num", num_pipeline, num_cols)
    num_pipe = pre.named_transformers_["num"]
    scaler = num_pipe.named_steps["scaler"]

    # StandardScaler(with_mean=False): z = x / scale_
    scales = {c: float(scaler.scale_[i]) for i, c in enumerate(num_cols)}

    # beta for original numeric col: w_num__c / scale[c]
    beta = {}
    for c in num_cols:
        fname = f"num__{c}"
        if fname in coef:
            beta[c] = float(coef[fname]) / scales[c]

    # full logit at x0 = b + sum_i w_i * z_i
    full_logit_at_x0 = b + float(np.dot(w, z))

    # We'll later re-add numeric parts via beta[c] * x[c], so remove them from const_term
    subtract = sum(beta[c] * float(X_row[c]) for c in beta.keys())
    const_term = full_logit_at_x0 - subtract

    return const_term, beta


def solve_german_recourse_numeric_only(
    pipe,
    X_train: pd.DataFrame,
    x0: pd.Series,
    mutable_cols=("duration", "credit_amount", "installment_commitment", "existing_credits", "residence_since"),
    direction="auto",              # applied to non-special features only: "auto" | "increase" | "decrease" | "both"
    margin=1e-6,
    slack_penalty=1000.0,
    # ---- Moderate plausibility constraints ----
    duration_max_decrease=24,      # months
    credit_max_rel_decrease=0.50,  # 50%
    enforce_decrease_for=("duration", "credit_amount"),
    integer_cols=("duration", "credit_amount", "installment_commitment", "existing_credits", "residence_since"),
):
    """
    Numeric-only recourse with plausibility constraints + slack.

    Objective:
      minimize normalized L1 change + slack_penalty * slack
    Constraint:
      logit(x_cf) + slack >= 0

    Returns:
      x_cf (pd.Series), slack (float)
        slack == 0 => feasible flip found
        slack > 0  => closest possible under constraints, still not flipped
    """
    const_term, beta_all = _extract_linear_terms_from_pipeline(pipe, x0)

    # Keep only columns that the linear model exposes as numeric terms
    mutable_cols = [c for c in mutable_cols if c in beta_all]
    if not mutable_cols:
        raise ValueError("None of the requested mutable_cols are usable numeric cols in the trained pipeline.")

    beta = {c: beta_all[c] for c in mutable_cols}

    integer_cols = set(integer_cols)
    int_set = set(mutable_cols) & integer_cols  # only those we are actually optimizing

    # bounds from training data min/max, then apply plausibility policies
    bounds = {}
    for c in mutable_cols:
        lo = float(X_train[c].min())
        hi = float(X_train[c].max())
        x0c = float(x0[c])

        # --- Feature-specific plausibility rules (moderate) ---
        if c in enforce_decrease_for:
            # only allow decreasing
            hi = min(hi, x0c)

            if c == "duration":
                lo = max(lo, x0c - float(duration_max_decrease))
            elif c == "credit_amount":
                lo = max(lo, x0c * (1.0 - float(credit_max_rel_decrease)))

        else:
            # Apply generic directional constraints for other numeric features
            if direction == "decrease":
                hi = min(hi, x0c)
            elif direction == "increase":
                lo = max(lo, x0c)
            elif direction == "auto":
                # move only in direction that increases logit
                if beta[c] >= 0:
                    lo = max(lo, x0c)   # increasing helps
                else:
                    hi = min(hi, x0c)   # decreasing helps
            elif direction == "both":
                pass
            else:
                raise ValueError("direction must be one of: auto, increase, decrease, both")

        # If bounds got inverted, fix to a single point at x0
        if lo > hi:
            lo, hi = x0c, x0c

        bounds[c] = (lo, hi)

    # Normalize objective by feature range to prevent one feature dominating
    ranges = {c: max(float(X_train[c].max() - X_train[c].min()), 1e-9) for c in mutable_cols}

    # Build MILP
    m = ConcreteModel()

    # ✅ Domain rule: integer for int_set, real otherwise
    def _dom_rule(_m, c):
        return Integers if c in int_set else Reals

    m.x = Var(mutable_cols, within=_dom_rule)
    m.t = Var(mutable_cols, within=NonNegativeReals)
    m.s = Var(within=NonNegativeReals)  # slack (>=0)

    # Set bounds
    for c in mutable_cols:
        lo, hi = bounds[c]
        m.x[c].setlb(lo)
        m.x[c].setub(hi)

    m.cons = ConstraintList()

    # L1 abs linearization
    for c in mutable_cols:
        x0c = float(x0[c])
        m.cons.add(m.t[c] >= m.x[c] - x0c)
        m.cons.add(m.t[c] >= x0c - m.x[c])

    # Flip constraint with slack
    m.cons.add(const_term + sum(beta[c] * m.x[c] for c in mutable_cols) + m.s >= 0.0 + margin)

    # Objective: minimize change + big slack penalty
    m.obj = Objective(
        expr=sum((m.t[c] / ranges[c]) for c in mutable_cols) + float(slack_penalty) * m.s,
        sense=minimize
    )

    solver = Highs()
    solver.solve(m)

    # Extract solution
    x_cf = x0.copy()
    for c in mutable_cols:
        x_cf[c] = float(value(m.x[c]))

    slack = float(value(m.s))
    return x_cf, slack
