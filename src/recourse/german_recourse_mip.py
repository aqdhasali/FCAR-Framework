import numpy as np
import pandas as pd

from pyomo.environ import (
    ConcreteModel, Var, Reals, NonNegativeReals, Integers, Binary,
    Objective, ConstraintList, minimize, value
)
from pyomo.contrib.appsi.solvers.highs import Highs
from pyomo.contrib.appsi.base import TerminationCondition

# Monotonic "improvement" orderings
CHECKING_ORDER = ["A11", "A12", "A13", "A14"]
SAVINGS_ORDER  = ["A61", "A62", "A63", "A64", "A65"]


def _full_logit(pipe, x_row: pd.Series) -> float:
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    z = pre.transform(pd.DataFrame([x_row]))
    z = np.asarray(z).ravel()
    w = clf.coef_.ravel()
    b = float(clf.intercept_.ravel()[0])
    return b + float(np.dot(w, z))


def _extract_numeric_beta(pipe):
    """beta[col] so that numeric contribution is beta[col] * x[col]."""
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    feat_names = pre.get_feature_names_out()
    w = clf.coef_.ravel()
    coef = dict(zip(feat_names, w))

    num_cols = list(pre.transformers_[0][2])
    num_pipe = pre.named_transformers_["num"]
    scaler = num_pipe.named_steps["scaler"]
    scales = {c: float(scaler.scale_[i]) for i, c in enumerate(num_cols)}

    beta = {}
    for c in num_cols:
        fname = f"num__{c}"
        if fname in coef:
            beta[c] = float(coef[fname]) / scales[c]
    return beta


def _extract_cat_weights(pipe, mutable_cat_cols):
    """w_cat[col][cat] = coefficient for onehot(col=cat).

    Uses the encoder's own category list rather than string-splitting feature
    names to avoid breakage on column/category names that contain underscores.
    """
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    feat_names = pre.get_feature_names_out()
    w = clf.coef_.ravel()
    coef = dict(zip(feat_names, w))

    cat_pipe = pre.named_transformers_["cat"]
    enc = cat_pipe.named_steps["onehot"]
    all_cat_cols = list(pre.transformers_[1][2])

    def _normalize(name: str) -> str:
        return name.replace("-", "_").replace(" ", "_").lower()

    sklearn_to_config: dict[str, str] = {}
    for sklearn_col in all_cat_cols:
        for config_col in mutable_cat_cols:
            if sklearn_col == config_col or _normalize(sklearn_col) == _normalize(config_col):
                sklearn_to_config[sklearn_col] = config_col
                break

    w_cat = {c: {} for c in mutable_cat_cols}
    for j, sklearn_col in enumerate(all_cat_cols):
        config_col = sklearn_to_config.get(sklearn_col)
        if config_col is None:
            continue
        for cat in enc.categories_[j]:
            cat_str = str(cat)
            fname = f"cat__{sklearn_col}_{cat_str}"
            if fname in coef:
                w_cat[config_col][cat_str] = float(coef[fname])
    return w_cat


def _get_cat_categories_from_encoder(pipe, cat_cols):
    pre = pipe.named_steps["pre"]
    cat_pipe = pre.named_transformers_["cat"]
    enc = cat_pipe.named_steps["onehot"]
    all_cat_cols = list(pre.transformers_[1][2])

    cats = {}
    for col in cat_cols:
        j = all_cat_cols.index(col)
        cats[col] = [str(x) for x in enc.categories_[j]]
    return cats


def _rank_map_for(col: str, categories: list[str]):
    if col == "checking_status":
        order = [c for c in CHECKING_ORDER if c in categories]
    elif col == "savings_status":
        order = [c for c in SAVINGS_ORDER if c in categories]
    else:
        order = categories[:]

    for c in categories:
        if c not in order:
            order.append(c)

    rank = {c: i for i, c in enumerate(order)}
    return rank, order


def solve_german_recourse_mip(
    pipe,
    X_train: pd.DataFrame,
    x0: pd.Series,

    # numeric action set
    mutable_num_cols=("duration", "credit_amount", "installment_commitment", "existing_credits", "residence_since"),
    # categorical action set
    mutable_cat_cols=("checking_status", "savings_status"),

    # plausibility (Option B style)
    duration_max_decrease=48,
    credit_max_rel_decrease=0.80,
    enforce_decrease_for=("duration", "credit_amount"),
    integer_num_cols=("duration", "credit_amount", "installment_commitment", "existing_credits", "residence_since"),

    # costs (per-feature weights)
    num_weights=None,          # dict: {"credit_amount": 2.0, "duration": 1.2, ...} default 1.0
    cat_step_weights=None,     # dict: {"checking_status": 0.25, "savings_status": 0.25} default 0.25

    slack_penalty=1000.0,
    margin=1e-6,
):
    """
    Returns (x_cf, slack).
    slack==0 => feasible flip found; slack>0 => closest under constraints.
    """
    if num_weights is None:
        num_weights = {}
    if cat_step_weights is None:
        cat_step_weights = {}

    logit0 = _full_logit(pipe, x0)

    beta_all = _extract_numeric_beta(pipe)
    mutable_num_cols = [c for c in mutable_num_cols if c in beta_all]
    beta = {c: beta_all[c] for c in mutable_num_cols}

    w_cat = _extract_cat_weights(pipe, list(mutable_cat_cols))
    cat_categories = _get_cat_categories_from_encoder(pipe, list(mutable_cat_cols))

    # constant term: remove mutable numeric + mutable cat current contributions from logit0
    const_term = float(logit0)
    for c in mutable_num_cols:
        const_term -= float(beta[c]) * float(x0[c])
    for col in mutable_cat_cols:
        v0 = str(x0[col])
        const_term -= float(w_cat.get(col, {}).get(v0, 0.0))

    # numeric bounds (train min/max + plausibility)
    bounds = {}
    for c in mutable_num_cols:
        lo = float(X_train[c].min())
        hi = float(X_train[c].max())
        x0c = float(x0[c])

        if c in enforce_decrease_for:
            hi = min(hi, x0c)
            if c == "duration":
                lo = max(lo, x0c - float(duration_max_decrease))
            elif c == "credit_amount":
                lo = max(lo, x0c * (1.0 - float(credit_max_rel_decrease)))

        if lo > hi:
            lo, hi = x0c, x0c
        bounds[c] = (lo, hi)

    ranges = {c: max(float(X_train[c].max() - X_train[c].min()), 1e-9) for c in mutable_num_cols}

    m = ConcreteModel()
    int_set = set(mutable_num_cols) & set(integer_num_cols)

    def _dom_rule(_m, c):
        return Integers if c in int_set else Reals

    m.x = Var(mutable_num_cols, within=_dom_rule)
    m.t = Var(mutable_num_cols, within=NonNegativeReals)
    m.s = Var(within=NonNegativeReals)

    for c in mutable_num_cols:
        lo, hi = bounds[c]
        m.x[c].setlb(lo)
        m.x[c].setub(hi)

    # categorical binaries
    cat_pairs = [(col, cat) for col in mutable_cat_cols for cat in cat_categories[col]]
    m.g = Var(cat_pairs, within=Binary)

    m.cons = ConstraintList()

    # numeric abs linearization
    for c in mutable_num_cols:
        x0c = float(x0[c])
        m.cons.add(m.t[c] >= m.x[c] - x0c)
        m.cons.add(m.t[c] >= x0c - m.x[c])

    # categorical: exactly one, monotonic improvement only
    cat_rank = {}
    cur_rank = {}
    for col in mutable_cat_cols:
        cats = cat_categories[col]
        rank, _ = _rank_map_for(col, cats)
        cat_rank[col] = rank

        v0 = str(x0[col])
        cur_rank[col] = rank.get(v0, 0)

        # prohibit worsening
        for cat in cats:
            if rank[cat] < cur_rank[col]:
                m.g[(col, cat)].setub(0)

        m.cons.add(sum(m.g[(col, cat)] for cat in cats) == 1)

    # decision boundary with slack
    cat_logit = 0
    for col in mutable_cat_cols:
        for cat in cat_categories[col]:
            cat_logit += float(w_cat.get(col, {}).get(cat, 0.0)) * m.g[(col, cat)]

    m.cons.add(
        const_term
        + sum(float(beta[c]) * m.x[c] for c in mutable_num_cols)
        + cat_logit
        + m.s
        >= 0.0 + float(margin)
    )

    # --- Objective: per-feature weighted numeric L1 + per-col weighted cat steps + slack penalty ---
    numeric_cost = 0
    for c in mutable_num_cols:
        w = float(num_weights.get(c, 1.0))
        numeric_cost += w * (m.t[c] / ranges[c])

    cat_cost = 0
    for col in mutable_cat_cols:
        w_step = float(cat_step_weights.get(col, 0.25))  # default 0.25
        exp_rank = sum(float(cat_rank[col][cat]) * m.g[(col, cat)] for cat in cat_categories[col])
        cat_cost += w_step * (exp_rank - float(cur_rank[col]))

    m.obj = Objective(
        expr=numeric_cost + cat_cost + float(slack_penalty) * m.s,
        sense=minimize
    )

    solver = Highs()
    results = solver.solve(m)
    tc = results.termination_condition
    if tc not in (TerminationCondition.optimal, TerminationCondition.maxTimeLimit):
        return x0.copy(), float("inf")

    # build counterfactual
    x_cf = x0.copy()
    try:
        for c in mutable_num_cols:
            x_cf[c] = float(value(m.x[c]))

        for col in mutable_cat_cols:
            cats = cat_categories[col]
            best_cat = max(cats, key=lambda cat: float(value(m.g[(col, cat)])))
            x_cf[col] = best_cat

        slack = float(value(m.s))
    except Exception:
        return x0.copy(), float("inf")

    return x_cf, slack