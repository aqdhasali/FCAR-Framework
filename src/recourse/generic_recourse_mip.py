import numpy as np
import pandas as pd

from pyomo.environ import (
    ConcreteModel, Var, Reals, NonNegativeReals, Integers, Binary,
    Objective, ConstraintList, minimize, value
)
from pyomo.contrib.appsi.solvers.highs import Highs

from src.config.config_loader import (
    get_mutable_numeric_cols,
    get_mutable_categorical_cols,
    get_numeric_cost_weights,
    get_categorical_step_weights,
    get_categorical_orders,
    get_monotonic_categorical_cols,
    get_integer_cols,
    get_solver_settings,
    get_plausibility_params,
    get_max_cat_changes,
)


def _full_logit(pipe, x_row: pd.Series) -> float:
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    z = pre.transform(pd.DataFrame([x_row]))
    if hasattr(z, "toarray"):
        z = z.toarray()
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
    """w_cat[col][cat] = coefficient for onehot(col=cat)."""
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    feat_names = pre.get_feature_names_out()
    w = clf.coef_.ravel()

    w_cat = {c: {} for c in mutable_cat_cols}
    for fname, wi in zip(feat_names, w):
        if not fname.startswith("cat__"):
            continue
        rest = fname[len("cat__"):]
        if "_" not in rest:
            continue
        col, cat = rest.rsplit("_", 1)
        if col in w_cat:
            w_cat[col][cat] = float(wi)
    return w_cat


def _get_cat_categories_from_encoder(pipe, cat_cols):
    pre = pipe.named_steps["pre"]
    cat_pipe = pre.named_transformers_["cat"]
    enc = cat_pipe.named_steps["onehot"]
    all_cat_cols = list(pre.transformers_[1][2])

    cats = {}
    for col in cat_cols:
        if col in all_cat_cols:
            j = all_cat_cols.index(col)
            cats[col] = [str(x) for x in enc.categories_[j]]
        else:
            cats[col] = [] # Fallback
    return cats


def _rank_map_for(col: str, categories: list[str], cat_orders: dict):
    order = list(cat_orders.get(col, []))
    order = [c for c in order if c in categories]
    
    for c in categories:
        if c not in order:
            order.append(c)

    rank = {c: i for i, c in enumerate(order)}
    return rank, order


def solve_recourse_mip(
    pipe,
    X_train: pd.DataFrame,
    x0: pd.Series,
    config: dict,
):
    """
    Generalized MIP recourse solver.

    Returns (x_cf, slack).
    slack==0 => feasible flip found; slack>0 => closest under constraints.
    """
    # 1. Extract settings from config
    mutable_num_cols = get_mutable_numeric_cols(config)
    mutable_cat_cols = get_mutable_categorical_cols(config)
    
    num_weights = get_numeric_cost_weights(config)
    cat_step_weights = get_categorical_step_weights(config)
    cat_orders = get_categorical_orders(config)
    monotonic_cols = get_monotonic_categorical_cols(config)
    int_num_cols = set(get_integer_cols(config))
    plaus_params = get_plausibility_params(config)
    
    solver_settings = get_solver_settings(config)
    slack_penalty = solver_settings["slack_penalty"]
    margin = solver_settings["margin"]

    logit0 = _full_logit(pipe, x0)

    beta_all = _extract_numeric_beta(pipe)
    # Only keep mutable cols that actually appear in the model
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

        pp = plaus_params.get(c, {})
        
        # Apply strict decrease directions
        if "max_decrease" in pp or "max_rel_decrease" in pp:
            hi = min(hi, x0c)
            # Apply lower bounds from max decrease rules
            lo_dec = lo
            if "max_decrease" in pp:
                lo_dec = max(lo_dec, x0c - float(pp["max_decrease"]))
            if "max_rel_decrease" in pp:
                lo_dec = max(lo_dec, x0c * (1.0 - float(pp["max_rel_decrease"])))
            lo = max(lo, lo_dec)

        # Apply strict increase directions
        if "max_increase" in pp or "max_rel_increase" in pp:
            lo = max(lo, x0c)
            # Apply upper bounds from max increase rules
            hi_inc = hi
            if "max_increase" in pp:
                hi_inc = min(hi_inc, x0c + float(pp["max_increase"]))
            if "max_rel_increase" in pp:
                hi_inc = min(hi_inc, x0c * (1.0 + float(pp["max_rel_increase"])))
            hi = min(hi, hi_inc)
            
        # Direction only rules (if no explicit max amount specified)
        dir_rule = config.get("mutable_numeric", {}).get(c, {}).get("direction")
        if dir_rule == "decrease":
            hi = min(hi, x0c)
        elif dir_rule == "increase":
            lo = max(lo, x0c)

        if lo > hi:
            lo, hi = x0c, x0c
        bounds[c] = (lo, hi)

    ranges = {c: max(float(X_train[c].max() - X_train[c].min()), 1e-9) for c in mutable_num_cols}

    m = ConcreteModel()

    def _dom_rule(_m, c):
        return Integers if c in int_num_cols else Reals

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

    # categorical constraints
    cat_rank = {}
    cur_rank = {}
    for col in mutable_cat_cols:
        cats = cat_categories[col]
        rank, _ = _rank_map_for(col, cats, cat_orders)
        cat_rank[col] = rank

        v0 = str(x0[col])
        cur_rank[col] = rank.get(v0, 0)

        # prohibit worsening if monotonic
        if col in monotonic_cols:
            for cat in cats:
                if rank[cat] < cur_rank[col]:
                    m.g[(col, cat)].setub(0)

        # exactly one category selected
        m.cons.add(sum(m.g[(col, cat)] for cat in cats) == 1)

    # --- max_cat_changes constraint ---
    # Limit the total number of categorical features that change.
    max_cc = get_max_cat_changes(config)
    if max_cc is not None and len(mutable_cat_cols) > 0:
        # d[col] = 1 iff category changed from original value
        m.d = Var(list(mutable_cat_cols), within=Binary)
        for col in mutable_cat_cols:
            v0_str = str(x0[col])
            cats = cat_categories[col]
            if v0_str in cats:
                # d[col] >= 1 - g[(col, v0)]  i.e. if we move away from v0, d=1
                m.cons.add(m.d[col] >= 1 - m.g[(col, v0_str)])
                # d[col] <= 1 - g[(col, v0)]  i.e. if we stay at v0, d=0
                m.cons.add(m.d[col] <= 1 - m.g[(col, v0_str)])
            else:
                # original value not in known categories -> always counts as changed
                m.cons.add(m.d[col] == 1)
        m.cons.add(sum(m.d[col] for col in mutable_cat_cols) <= max_cc)

    # decision boundary with slack
    cat_logit = 0
    for col in mutable_cat_cols:
        for cat in cat_categories[col]:
            cat_logit += float(w_cat.get(col, {}).get(cat, 0.0)) * m.g[(col, cat)]

    # The actual constraint based on model predicting class 1 with >= 0.5 probability
    # logodds >= 0 implies P(y=1) >= 0.5. To make target class configurable we'd need more logic,
    # but since classifiers are binary, predicting positive is always logit >= 0. 
    # For Adult we want >50K (class 1). For Default Credit we want no-default (class 0).
    # Since scikit-learn LogisticRegression treats the second class as positive class (for predict_proba[:,1]),
    # if `label_positive` = 0, we flip the logit condition: logit <= 0
    target_cls = int(config.get("label_positive", 1))
    
    total_logit = const_term + sum(float(beta[c]) * m.x[c] for c in mutable_num_cols) + cat_logit
    
    if target_cls == 0:
        # P(y=1) < 0.5 => logit < 0 => total_logit - m.s <= -margin
        m.cons.add(total_logit - m.s <= 0.0 - float(margin))
    else:
        # P(y=1) >= 0.5 => logit >= 0 => total_logit + m.s >= margin
        m.cons.add(total_logit + m.s >= 0.0 + float(margin))

    # --- Objective: per-feature weighted numeric L1 + per-col weighted cat steps + slack penalty ---
    numeric_cost = 0
    for c in mutable_num_cols:
        w = float(num_weights.get(c, 1.0))
        numeric_cost += w * (m.t[c] / ranges[c])

    cat_cost = 0
    for col in mutable_cat_cols:
        w_step = float(cat_step_weights.get(col, 0.25))
        is_monotonic = col in monotonic_cols
        v0_cat = str(x0[col])
        if is_monotonic:
            # Ordered: cost is proportional to how many ranks up we move
            exp_rank = sum(float(cat_rank[col][cat]) * m.g[(col, cat)] for cat in cat_categories[col])
            cat_cost += w_step * (exp_rank - float(cur_rank[col]))
        else:
            # Unordered: flat cost of w_step if the category changes at all
            # (avoids arbitrary rank-based costs for unordered sets)
            stays = m.g[(col, v0_cat)] if v0_cat in cat_categories[col] else 0
            cat_cost += w_step * (1 - stays)

    m.obj = Objective(
        expr=numeric_cost + cat_cost + float(slack_penalty) * m.s,
        sense=minimize
    )

    solver = Highs()
    solver.solve(m)

    # build counterfactual
    x_cf = x0.copy()

    for c in mutable_num_cols:
        x_cf[c] = float(value(m.x[c]))

    for col in mutable_cat_cols:
        cats = cat_categories[col]
        best_cat = max(cats, key=lambda cat: float(value(m.g[(col, cat)])))
        x_cf[col] = best_cat

    slack = float(value(m.s))
    return x_cf, slack
