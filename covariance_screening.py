"""Compact-box screening for Gaussian LMM forward additions.

The bounds are mathematical certificates in exact arithmetic. This prototype
uses ordinary float64 with a conservative numerical cushion, not directed
interval arithmetic. Its guarantee is restricted to the supplied ratio box.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.optimize import brentq


@dataclass
class Evaluation:
    f: np.ndarray
    h: np.ndarray
    h_gradient: np.ndarray
    g: float
    g_gradient: np.ndarray
    sse: np.ndarray


class CrossedGaussian:
    """Profile ML with R = I + sum(lambda[k] Z[k] Z[k].T).

    Model index 0 is the base design; model index j+1 adds candidate j.
    No n-by-n or p-by-p matrix is formed by the fast engine. Storage is
    O(n(p+k)+r(p+k)+r^2), where r is the total number of random-effect columns.
    A dense r-by-r Cholesky is intended only for a low-rank pilot.
    """

    def __init__(self, y, base, candidates, random_designs):
        self.y = np.asarray(y, dtype=float)
        self.base = np.asarray(base, dtype=float)
        self.candidates = np.asarray(candidates, dtype=float)
        self.n, self.p = self.candidates.shape
        self.k = self.base.shape[1]
        if self.y.shape != (self.n,) or self.base.shape[0] != self.n:
            raise ValueError("Inconsistent dimensions")
        self.designs = [np.asarray(z, dtype=float) for z in random_designs]
        self.q = len(self.designs)
        self.sizes = np.array([z.shape[1] for z in self.designs])
        self.z = np.column_stack(self.designs)
        self.r = self.z.shape[1]
        self.a = self.z.T @ self.z
        self.h0 = np.column_stack((self.base, self.candidates, self.y))
        self.zh = self.z.T @ self.h0
        self.bb = self.base.T @ self.base
        self.bc = self.base.T @ self.candidates
        self.by = self.base.T @ self.y
        self.cc = np.sum(self.candidates**2, axis=0)
        self.cy = self.candidates.T @ self.y
        self.yy = self.y @ self.y
        self.calls = 0
        self.determinant_calls = 0
        self._g_cache = {}

    def _covariance(self, lam):
        lam = np.asarray(lam, dtype=float)
        if lam.shape != (self.q,) or np.any(lam < 0):
            raise ValueError("Variance ratios must be nonnegative")
        sd = np.sqrt(np.repeat(lam, self.sizes))
        small = np.eye(self.r) + sd[:, None] * self.a * sd[None, :]
        ch = cho_factor(small, lower=True, check_finite=False)
        g = float(2 * np.log(np.diag(ch[0])).sum())
        return sd, ch, g

    def logdet(self, lam):
        key = tuple(float(x) for x in lam)
        if key not in self._g_cache:
            self.determinant_calls += 1
            self._g_cache[key] = self._covariance(lam)[2]
        return self._g_cache[key]

    def evaluate(self, lam, candidate_ids=None):
        """Evaluate base and requested candidate models at one common ratio."""
        self.calls += 1
        ids = np.arange(self.p) if candidate_ids is None else np.asarray(candidate_ids, dtype=int)
        ix = np.r_[np.arange(self.k), self.k + ids, self.k + self.p]
        zh = self.zh[:, ix]
        sd, ch, g = self._covariance(lam)
        self._g_cache[tuple(float(x) for x in lam)] = g
        rhs = sd[:, None] * zh
        uu = solve_triangular(ch[0], rhs, lower=True, check_finite=False)
        ub, uc, uy = uu[:, :self.k], uu[:, self.k:-1], uu[:, -1]
        bb = self.bb - ub.T @ ub
        bc = self.bc[:, ids] - ub.T @ uc
        by = self.by - ub.T @ uy
        cc = self.cc[ids] - np.sum(uc**2, axis=0)
        cy = self.cy[ids] - uc.T @ uy
        yy = self.yy - uy @ uy
        bch = cho_factor(bb, lower=True, check_finite=False)
        beta = cho_solve(bch, by, check_finite=False)
        gamma = cho_solve(bch, bc, check_finite=False)
        s0 = float(yy - by @ beta)
        denom = cc - np.sum(bc * gamma, axis=0)
        if np.any(denom <= 1e-11 * np.maximum(1.0, self.cc[ids])):
            raise ValueError("Rank-deficient or numerically inadmissible addition")
        coef = (cy - bc.T @ beta) / denom
        s = np.r_[s0, s0 - denom * coef**2]
        if np.any(s <= 1e-12 * max(1.0, self.yy)):
            raise ValueError("Residual sum of squares too small for this prototype")
        ez = zh - (self.a * sd[None, :]) @ cho_solve(ch, rhs, check_finite=False)
        vz0 = ez[:, -1] - ez[:, :self.k] @ beta
        vzc = vz0[:, None] - (ez[:, self.k:-1] - ez[:, :self.k] @ gamma) * coef
        vz = np.column_stack((vz0, vzc))
        zrzinv = self.a - (self.a * sd[None, :]) @ cho_solve(
            ch, sd[:, None] * self.a, check_finite=False)
        hgrad = np.empty((len(ids) + 1, self.q))
        ggrad = np.empty(self.q)
        start = 0
        for kk, size in enumerate(self.sizes):
            sl = slice(start, start + size)
            hgrad[:, kk] = -self.n * np.sum(vz[sl]**2, axis=0) / s
            ggrad[kk] = np.trace(zrzinv[sl, sl])
            start += size
        h = self.n * np.log(s / self.n)
        return Evaluation(g + h, h, hgrad, g, ggrad, s)

    def fit(self, model_index, lo, hi, starts):
        """Multi-start local ML; the returned feasible value is NOT a global bound."""
        ids = [] if model_index == 0 else [model_index - 1]
        slot = 0 if model_index == 0 else 1
        best_f, best_x, success = np.inf, None, False
        for start in starts:
            x0 = np.clip(np.asarray(start, dtype=float), lo, hi)
            def fun(x):
                ev = self.evaluate(x, ids)
                return float(ev.f[slot]), ev.h_gradient[slot] + ev.g_gradient
            initial_f = fun(x0)[0]
            if initial_f < best_f:
                best_f, best_x = initial_f, x0.copy()
            opt = minimize(fun, x0, jac=True, method="L-BFGS-B",
                           bounds=list(zip(lo, hi)),
                           options={"ftol": 1e-12, "gtol": 1e-7, "maxiter": 350,
                                    "maxls": 40})
            if np.isfinite(opt.fun) and opt.fun <= best_f:
                best_f, best_x = float(opt.fun), opt.x.copy()
                success = bool(opt.success)
        return {"f": best_f, "lambda": best_x.tolist(), "optimizer_success": success}


def vertices(lo, hi):
    return np.asarray([np.where(bits, hi, lo) for bits in product([False, True], repeat=len(lo))])


def box_lower(model, lo, hi, anchor=None):
    """Theorem bound: min_vertex[g(vertex)+h(anchor)+h'(anchor)(vertex-anchor)]."""
    lo, hi = np.asarray(lo), np.asarray(hi)
    center = (lo + hi) / 2 if anchor is None else np.asarray(anchor)
    ev = model.evaluate(center)
    vv = vertices(lo, hi)
    gg = np.array([model.logdet(v) for v in vv])
    tangent = ev.h[:, None] + ev.h_gradient @ (vv - center).T
    lower = np.min(tangent + gg[None, :], axis=1)
    # Ordinary floating-point cushion. This is not a rigorous roundoff enclosure.
    padding = 2e-8 * (1 + np.maximum(np.abs(lower), np.abs(ev.f)))
    return lower - padding, ev, center


def screen_step(model, lo, hi, initial_lambda=None, max_splits=2000,
                time_budget=30.0, bound="tangent"):
    """Identify the globally best single addition over a *specified compact box*.

    Only local optimization supplies feasible upper bounds; all rivals are
    excluded with analytic lower bounds. Status 'separated' requires strict
    separation. Budget exhaustion is returned explicitly as 'unresolved'.
    The base-model fit is common overhead and is included in total time.
    """
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if lo.shape != (model.q,) or np.any(lo < 0) or np.any(hi <= lo):
        raise ValueError("Invalid compact box")
    started = time.perf_counter()
    starts = [lo, (lo+hi)/2, lo+0.1*(hi-lo), lo+0.9*(hi-lo)]
    base_fit = model.fit(0, lo, hi, starts) if initial_lambda is None else None
    lam0 = np.asarray(base_fit["lambda"] if base_fit else initial_lambda)
    ev0 = model.evaluate(lam0)
    winner = int(np.argmin(ev0.f[1:])) + 1
    candidate_fits = 0
    fitted_ids = set()
    best_f, best_lambda = np.inf, lam0

    def local_refit(j, start):
        nonlocal candidate_fits
        candidate_fits += 1
        fitted_ids.add(j)
        fit_starts = [lam0, *starts]
        if not any(np.array_equal(start, a) for a in fit_starts):
            fit_starts.append(start)
        return model.fit(j, lo, hi, fit_starts)

    fit = local_refit(winner, lam0)
    best_f, best_lambda = fit["f"], np.asarray(fit["lambda"])
    def make_box(a, b):
        lb, ev, point = box_lower(model, a, b)
        if bound == "loewner":
            upper = model.evaluate(b)
            lb = model.logdet(a) + model.n * np.log(upper.sse/model.n)
            lb -= 2e-8*(1 + np.abs(lb))
        return {"lo": a, "hi": b, "lower": lb, "ev": ev, "point": point}

    boxes = [make_box(lo.copy(), hi.copy())]
    # A second supporting tangent at the current base fit is valid globally.
    if bound == "tangent":
        extra, _, _ = box_lower(model, lo, hi, lam0)
        boxes[0]["lower"] = np.maximum(boxes[0]["lower"], extra)
    active = np.ones(model.p, dtype=bool)
    split_count = 0
    root_lower = boxes[0]["lower"].copy()
    status = "unresolved"
    last_bounds = root_lower
    while True:
        lower_matrix = np.array([b["lower"][1:] for b in boxes])
        last_bounds = np.r_[min(b["lower"][0] for b in boxes), lower_matrix.min(axis=0)]
        cushion = 2e-8 * (1 + abs(best_f))
        active &= last_bounds[1:] <= best_f + cushion
        active[winner-1] = True
        rival_mask = active.copy()
        rival_mask[winner-1] = False
        if not np.any(rival_mask):
            status = "separated"
            break
        if split_count >= max_splits or time.perf_counter() - started > time_budget:
            break
        box_index = int(np.argmin(np.min(lower_matrix[:, rival_mask], axis=1)))
        box = boxes.pop(box_index)
        widths = (box["hi"]-box["lo"])/(hi-lo)
        dim = int(np.argmax(widths))
        mid = (box["hi"][dim] + box["lo"][dim])/2
        children = []
        for side in range(2):
            aa, bb = box["lo"].copy(), box["hi"].copy()
            if side == 0:
                bb[dim] = mid
            else:
                aa[dim] = mid
            child = make_box(aa, bb)
            # Retain the valid parent lower bound as well as the new one.
            child["lower"] = np.maximum(child["lower"], box["lower"])
            children.append(child)
            candidate_values = child["ev"].f[1:]
            j = int(np.argmin(candidate_values)) + 1
            if candidate_values[j-1] < best_f - cushion:
                fit = local_refit(j, child["point"])
                if fit["f"] < best_f:
                    winner, best_f = j, fit["f"]
                    best_lambda = np.asarray(fit["lambda"])
        boxes.extend(children)
        split_count += 1
    rival_lowers = np.delete(last_bounds[1:], winner-1)
    sep = float(rival_lowers.min() - best_f) if len(rival_lowers) else np.inf
    sigma2_base = ev0.sse[0] / model.n
    q_scores = (ev0.sse[0] - ev0.sse[1:]) / sigma2_base
    # These compare alternative global minima to a fixed, feasible base value.
    lr_upper = ev0.f[0] - last_bounds[1:]
    return {
        "winner": winner-1, "status": status, "best_feasible_f": best_f,
        "best_lambda": best_lambda.tolist(), "base_lambda": lam0.tolist(),
        "base_feasible_f": float(ev0.f[0]),
        "candidate_fits": candidate_fits, "distinct_refitted_candidates": len(fitted_ids),
        "fitted_candidates": sorted(j-1 for j in fitted_ids),
        "omitted_full_fits": model.p-len(fitted_ids), "splits": split_count,
        "leaves": len(boxes), "remaining_candidates": int(active.sum()),
        "separation_margin": sep, "time_seconds": time.perf_counter()-started,
        "shared_evaluations": model.calls, "vertex_factorizations": model.determinant_calls,
        "q_scores": q_scores.tolist(), "lr_upper": lr_upper.tolist(),
        "b_c": (lr_upper-q_scores).tolist(),
        "global_f_lower": last_bounds[1:].tolist(),
        "root_f_lower": root_lower[1:].tolist(), "box_lo": lo.tolist(), "box_hi": hi.tolist(),
        "scope": "single-addition ranking on supplied compact ratio box; float64, not interval arithmetic"
    }


def exhaustive_reference(model, lo, hi):
    """All-candidate, multi-start local-ML comparator; not a global oracle."""
    started = time.perf_counter()
    lo, hi = np.asarray(lo,float), np.asarray(hi,float)
    starts = [lo, (lo+hi)/2, lo+0.1*(hi-lo), lo+0.9*(hi-lo)]
    base = model.fit(0,lo,hi,starts)
    fits = [model.fit(j,lo,hi,[np.asarray(base["lambda"]),*starts]) for j in range(1,model.p+1)]
    ff = np.array([fit["f"] for fit in fits])
    return {"winner":int(np.argmin(ff)), "f":ff.tolist(), "base":base,
            "fits":fits, "time_seconds":time.perf_counter()-started,
            "candidate_fits":model.p, "evaluations":model.calls}


def residual_floors(model):
    """S_C(lambda) >= distance(y, span(base, candidate C, Z))**2."""
    joint = np.column_stack((model.z, model.base))
    uu, ss, _ = np.linalg.svd(joint, full_matrices=False)
    rank = np.count_nonzero(ss > max(joint.shape)*np.finfo(float).eps*ss[0])
    qq = uu[:, :rank]
    yr = model.y - qq @ (qq.T @ model.y)
    xr = model.candidates - qq @ (qq.T @ model.candidates)
    denom = np.sum(xr*xr,axis=0)
    numer = xr.T @ yr
    reduction = np.zeros(model.p)
    usable = denom > 1e-11*np.maximum(1.,model.cc)
    reduction[usable] = numer[usable]**2 / denom[usable]
    floor = np.r_[yr@yr, yr@yr-reduction]
    # Numerical cushion for a computed residual lower bound, not outward rounding.
    floor -= 2e-8*(1+model.yy)
    if np.any(floor <= 0):
        raise ValueError("Positive residual floor not established; tail theorem is unavailable")
    return floor


def automatic_tail_box(model, feasible_threshold, margin=1.0):
    """Exclude the unbounded tails if all one-addition residual floors are positive.

    Outside the returned box every candidate objective exceeds the supplied
    feasible incumbent by at least margin, in exact arithmetic. Thus best-support
    screening inside this box also covers all nonnegative variance ratios.
    The box need not contain an irrelevant rival's global minimizer.
    """
    floors = residual_floors(model)
    floor = float(floors[1:].min())
    constant = model.n*np.log(floor/model.n)
    target = float(feasible_threshold+margin)
    hi=[];tail_values=[]
    for z in model.designs:
        eig=np.linalg.eigvalsh(z.T@z)
        eig=eig[eig>1e-12*max(1.,float(eig.max()))]
        if len(eig)==0:
            raise ValueError("Remove zero covariance components before tail bounding")
        def value(log_one_plus_lambda):
            lam=np.expm1(log_one_plus_lambda)
            return float(np.log1p(lam*eig).sum()+constant-target)
        if value(0)>=0:
            root=0.0
        else:
            right=1.0
            while value(right)<0 and right<60:
                right*=2
            if right>=60:
                raise ValueError("Tail box too large for this numerical prototype")
            root=brentq(value,0,right,xtol=1e-12)
        upper=max(1e-8,float(np.expm1(root)))*(1+1e-8)+1e-8
        hi.append(upper)
        tail_values.append(float(np.log1p(upper*eig).sum()+constant))
    return {"lo":[0.0]*model.q,"hi":hi,"residual_floors":floors.tolist(),
            "uniform_floor":floor,"outside_objective_lower":min(tail_values),
            "feasible_threshold":feasible_threshold,"margin":margin,
            "qualification":"Requires positive residual floors; float64 cushions are not interval proof."}
