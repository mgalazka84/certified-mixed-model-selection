"""Stage 8: controlled decision gaps and sensitivity to local starts.

No earlier engine or experiment is changed. Analytic minima apply only to
balanced one-group data with a within-group predictor. The crossed study has
no analytic oracle. This is a diagnostic study, not a population performance
comparison. Run with BLAS thread counts fixed before Python starts.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import brentq, minimize
from flint import arb, ctx

from certified_path import BlockGaussian
from certified_global_path import TailArbGaussian, global_family_search, verify_global_decision
from certified_crossed_backend import CrossedArbGaussian
from comparison_methods import LocalFits

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'results/difficult_decisions'
CORE = ['run_difficult_decisions.py', 'certified_path.py',
        'certified_global_path.py', 'certified_crossed_backend.py', 'comparison_methods.py']
TARGETS = [1., -1., .01, -.01, .0001, -.0001, .000001, -.000001, 1e-8, -1e-8]
STARTS = [[0., 0.], [.01, .01], [1., 1.], [4., 4.],
          [.1, 10.], [10., .1], [64., 0.], [0., 64.], [64., 64.]]
BITS, SPLITS, SECONDS = 128, 500, 8.


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def models(a):
    return BlockGaussian(a['y'], a['base'], a['x'],
                         [a[k] for k in sorted(a) if k.startswith('z')])


def analytic_float(y, x, labels):
    """Profile exact balanced-group formula evaluated in float for construction."""
    n, g = len(y), int(labels.max()) + 1
    m = n // g
    means = np.bincount(labels, weights=y) / m
    b = m * np.sum((means - y.mean()) ** 2)
    w0 = np.sum((y - means[labels]) ** 2)
    w1 = w0 - (x @ y) ** 2 / (x @ x)
    out = []
    for w in [w0, w1]:
        r = max(1., (n-g)*b/(g*w))
        out.append(g*np.log(r) + n*np.log((w+b/r)/n))
    return out[1] - out[0] + np.log(n)


def analytic_oracle(a):
    """Independent 256-bit group-sum oracle on the exact stored binary input."""
    ctx.prec = 256
    y, x = [list(map(arb, v)) for v in [a['y'], a['x'][:, 0]]]
    labels = a['label0']; n = len(y); g = int(labels.max()) + 1; m = n // g
    assert all(np.count_nonzero(labels == k) == m for k in range(g))
    assert all(sum(x[i] for i in range(n) if labels[i] == k) == 0 for k in range(g))
    total = sum(y); sq = sum(v*v for v in y)
    group_sums = [sum(y[i] for i in range(n) if labels[i] == k) for k in range(g)]
    between = sum(v*v/m for v in group_sums) - total*total/n
    w0 = sq - sum(v*v/m for v in group_sums)
    xy = sum(u*v for u, v in zip(x, y)); xx = sum(v*v for v in x)
    within = [w0, w0 - xy*xy/xx]
    fs, ratios, branches = [], [], []
    for w in within:
        assert w > 0
        ratio = (n-g)*between/(g*w)
        if ratio > 1:
            branches.append('interior'); ratios.append((ratio-1)/m)
            fs.append(g*(between/g).log() + (n-g)*(w/(n-g)).log())
        elif ratio < 1:
            branches.append('zero_variance'); ratios.append(arb(0))
            fs.append(n*((w+between)/n).log())
        else:
            raise ValueError('Oracle branch cannot be resolved')
    delta = fs[1] - fs[0] + arb(n).log()
    # Exact-tie control also has an exact symbolic identity, checked below.
    tie = bool(n == 4 and within[0] == 8 and within[1] == 4 and between == 16)
    if tie:
        assert branches == ['interior', 'interior']
        winner = None
    elif delta > 0:
        winner = 0
    elif delta < 0:
        winner = 1
    else:
        raise ValueError('Oracle gap sign not resolved')
    return {'penalized_gap_interval': str(delta), 'gap_for_display': float(delta.mid()),
            'winner_index': winner, 'exact_tie': tie, 'branches': branches,
            'f_intervals': list(map(str, fs)), 'optimal_lambda_intervals': list(map(str, ratios)),
            'optimal_lambda_for_display': [float(v.mid()) for v in ratios],
            'minimum_within_residual': str(min(within)), 'precision_bits': 256}


def controlled(seed, target):
    rng = np.random.default_rng(seed); g, m = 12, 8; n = g*m
    labels = np.repeat(np.arange(g), m)
    x = np.tile([1., -1.] * (m//2), g)
    e = rng.normal(size=n)
    e -= np.repeat(e.reshape(g, m).mean(axis=1), m)
    e -= x*(x@e)/(x@x)
    e *= np.sqrt(n/(e@e))
    u = rng.normal(size=g); u -= u.mean(); u *= 2/np.sqrt(np.mean(u*u))
    y0 = u[labels] + e
    amp = brentq(lambda t: analytic_float(y0+t*x, x, labels)-target, 0., 2.,
                 xtol=5e-15, rtol=1e-14)
    a = {'y': y0+amp*x, 'base': np.ones((n, 1)), 'x': x[:, None],
         'z0': np.eye(g)[labels], 'label0': labels}
    return a, {'seed': seed, 'target_gap': target, 'calibrated_amplitude': amp,
               'oracle': analytic_oracle(a)}


def exact_tie():
    labels = np.array([0, 0, 1, 1])
    return {'y': np.array([4., 0., -2., -2.]), 'base': np.ones((4, 1)),
            'x': np.array([1., -1., 1., -1.])[:, None],
            'z0': np.eye(2)[labels], 'label0': labels}


def crossed(seed, regime):
    rng = np.random.default_rng(seed); n, p = 120, 3
    if regime in ['ordinary', 'boundary']:
        labels = [rng.permutation(np.arange(n) % r) for r in [8, 10]]
    else:
        one = rng.permutation(np.arange(n) % 8); two = one.copy()
        ix = rng.choice(n, size=6, replace=False)
        two[ix] = (two[ix] + rng.integers(1, 8, size=len(ix))) % 8
        labels = [one, two]
    zs = [np.eye(int(v.max())+1)[v] for v in labels]
    x = rng.normal(size=(n, p))
    if regime == 'group_aligned':
        for k in range(2):
            x[:, k] = rng.normal(size=zs[k].shape[1])[labels[k]] + .05*x[:, k]
    x = (x-x.mean(axis=0))/x.std(axis=0)
    sd = [.7, 0. if regime == 'boundary' else .8]
    y = .35*x[:, 2] + rng.normal(size=n)
    for z, s in zip(zs, sd):
        y += z @ rng.normal(scale=s, size=z.shape[1])
    a = {'y': y, 'base': np.ones((n, 1)), 'x': x}
    for k, (z, v) in enumerate(zip(zs, labels)):
        a[f'z{k}'] = z; a[f'label{k}'] = v
    return a


def prepare(pilot=False):
    dest = ROOT/'results/difficult_decisions_development' if pilot else DEST
    dest.mkdir(parents=True, exist_ok=True)
    cases = []
    seeds = [967099] if pilot else [967100, 967101, 967102]
    targets = [1e-2, 1e-6, -1e-8] if pilot else TARGETS
    for seed in seeds:
        for j, target in enumerate(targets):
            a, meta = controlled(seed, target)
            ident = f'gap_{seed}_{j:02d}'
            cases.append({'id': ident, 'kind': 'controlled', **meta})
            np.savez_compressed(dest/f'{ident}.npz', **a)
    a = exact_tie(); ident = 'exact_tie'
    cases.append({'id': ident, 'kind': 'exact_tie', 'oracle': analytic_oracle(a)})
    np.savez_compressed(dest/f'{ident}.npz', **a)
    for ri, regime in enumerate(['ordinary', 'boundary', 'near_alias', 'group_aligned']):
        for rep in range(1 if pilot else 6):
            seed = (967300 if pilot else 967400) + 100*ri + rep
            ident = f'{regime}_{seed}'
            np.savez_compressed(dest/f'{ident}.npz', **crossed(seed, regime))
            cases.append({'id': ident, 'kind': 'crossed', 'seed': seed, 'regime': regime})
    protocol = {'version': 'CSDA-stage8-v1', 'pilot_excluded': pilot,
                'cases': cases, 'bits': BITS, 'maximum_splits': SPLITS,
                'soft_seconds_per_family': SECONDS, 'subprocess_timeout_seconds': 180,
                'q1_starts': [0., .01, .1, 1., 4., 16., 64.], 'q2_starts': STARTS,
                'single_start_optimizer': {'method': 'L-BFGS-B', 'ftol': 1e-12, 'gtol': 1e-7,
                                          'maxiter': 350, 'maxls': 40, 'initial_upper': 128.,
                                          'upper_expansions': 6, 'expansion_factor': 4},
                'multistart': 'Unchanged LocalFits: base first, then base-warm-started rivals.',
                'estimand': 'One complete family: current intercept model versus every single addition.',
                'controlled_design': 'Three independent residual draws, ten calibrated perturbations each; perturbations are dependent. No Monte Carlo population inference.',
                'crossed_design': '24 independent datasets, six per regime. No analytic global minimum assumed.',
                'timing': 'One fresh process per case; BLAS one thread; cyclic order of native verification, multistart, and all single starts. Includes constructor, tail, search and in-search checker, excludes extra replay, I/O and imports.',
                'offline_replay': 'Apply unchanged full-orthant verifier afresh to every saved witness, including unresolved outcomes. Report separately; never relabel the online outcome.',
                'oracle': 'Independent exact-stored-input 256-bit balanced-group formula, plus symbolic exact tie.',
                'status_rule': 'Only verified online results are counted as certified algorithm completions; unresolved incumbents are proposals, not decisions.'}
    if (dest/'protocol.json').exists():
        assert json.loads((dest/'protocol.json').read_text()) == protocol
    dump(dest/'protocol.json', protocol)
    hashes = {f: sha(ROOT/f) for f in CORE}
    if (dest/'code_sha256.json').exists():
        assert json.loads((dest/'code_sha256.json').read_text()) == hashes
    dump(dest/'code_sha256.json', hashes)
    dump(dest/'input_sha256.json', {c['id']+'.npz': sha(dest/(c['id']+'.npz')) for c in cases})
    print(json.dumps({'prepared': len(cases), 'pilot': pilot}), flush=True)


def single_start(a, start):
    begun = time.perf_counter(); model = models(a)
    family = [(), *[(j,) for j in range(model.p)]]; fits = []
    for support in family:
        lo = np.zeros(model.q); hi = np.full(model.q, 128.); point = np.asarray(start, float)
        for expansion in range(7):
            def fun(lam):
                f, h, d, gg = model.evaluate(lam, [support])
                return float(f[0]), d[0]+gg
            opt = minimize(fun, point, jac=True, method='L-BFGS-B', bounds=list(zip(lo, hi)),
                           options={'ftol': 1e-12, 'gtol': 1e-7, 'maxiter': 350, 'maxls': 40})
            point = opt.x
            if np.any(point >= .99*hi) and expansion < 6:
                hi *= 4
                continue
            value, gradient = fun(point)
            pg = np.where((point <= 1e-7) & (gradient > 0), 0., gradient)
            fits.append({'f': value, 'lambda': point.tolist(), 'optimizer_success': bool(opt.success),
                         'message': str(opt.message), 'nit': int(opt.nit), 'nfev': int(opt.nfev),
                         'projected_gradient_max': float(np.max(np.abs(pg))),
                         'stationary': bool(np.max(np.abs(pg)) <= 1e-4),
                         'final_upper_active': bool(np.any(point >= .99*hi)),
                         'expansions': expansion})
            break
    values = [f['f']+len(s)*np.log(model.n) for f, s in zip(fits, family)]
    return {'start': list(start), 'fits': fits, 'values': values,
            'winner_index': int(np.argmin(values)), 'seconds': time.perf_counter()-begun}


def multistart(a):
    begun = time.perf_counter(); model = models(a); fitter = LocalFits(model)
    base = fitter.fit(())
    fits = [base, *[fitter.fit((j,), base['lambda']) for j in range(model.p)]]
    values = [f['f']+(j > 0)*np.log(model.n) for j, f in enumerate(fits)]
    return {'fits': fits, 'values': values, 'winner_index': int(np.argmin(values)),
            'stationarity_failures': fitter.stationarity_failures,
            'seconds': time.perf_counter()-begun}


def online(a):
    begun = time.perf_counter(); model = models(a)
    factory = CrossedArbGaussian if model.q == 2 else TailArbGaussian
    verifier = factory(model, BITS)
    out = global_family_search(model, (), [(j,) for j in range(model.p)], verifier,
                               max_splits=SPLITS, time_budget=SECONDS)
    out['total_seconds'] = time.perf_counter()-begun
    return out


def worker(ident, pilot=False):
    dest = ROOT/'results/difficult_decisions_development' if pilot else DEST
    protocol = json.loads((dest/'protocol.json').read_text())
    for file, value in json.loads((dest/'code_sha256.json').read_text()).items():
        assert sha(ROOT/file) == value, 'Frozen source changed: '+file
    assert sha(dest/f'{ident}.npz') == json.loads((dest/'input_sha256.json').read_text())[ident+'.npz']
    case = next(c for c in protocol['cases'] if c['id'] == ident)
    with np.load(dest/f'{ident}.npz') as f:
        a = {k: f[k] for k in f.files}
    q = len([k for k in a if k.startswith('z')])
    starts = [[v] for v in protocol['q1_starts']] if q == 1 else STARTS
    out = {'case': case, 'data_sha256': models(a).data_hash, 'methods': {}}
    order = ['verified', 'multistart', 'single_starts']
    ix = protocol['cases'].index(case) % 3; order = order[ix:]+order[:ix]
    out['execution_order'] = order
    for method in order:
        try:
            if method == 'verified':
                value = online(a)
            elif method == 'multistart':
                value = multistart(a)
            else:
                value = [single_start(a, start) for start in starts]
            out['methods'][method] = value
        except Exception as exc:
            out['methods'][method] = {'error': type(exc).__name__+': '+str(exc)}
    # Save primary outcomes before potentially expensive independent replay.
    dump(dest/f'{ident}.json', out)
    result = out['methods']['verified']
    if result.get('witness') is not None:
        model = models(a)
        factory = CrossedArbGaussian if q == 2 else TailArbGaussian
        out['offline_replay'] = verify_global_decision(model, result['witness'], factory(model, BITS))
    if q == 1:
        out['oracle_recheck'] = analytic_oracle(a)
        if result.get('status') == 'verified':
            assert not out['oracle_recheck']['exact_tie']
            assert result['winner_index'] == out['oracle_recheck']['winner_index']
        # Evaluate each analytic minimizer independently with the generic Arb engine.
        vg = TailArbGaussian(models(a), 256)
        out['oracle_point_checks'] = []
        for j, lam in enumerate(out['oracle_recheck']['optimal_lambda_for_display']):
            native = vg.profile([lam], () if j == 0 else (0,))[0]
            target = arb(out['oracle_recheck']['f_intervals'][j])
            out['oracle_point_checks'].append({'profile_minus_oracle': str(native-target),
                                                'absolute_difference_upper': float(abs(native-target).upper())})
    dump(dest/f'{ident}.json', out)
    print(json.dumps({'id': ident, 'status': result.get('status', 'error'),
                      'splits': result.get('splits'), 'seconds': result.get('total_seconds'),
                      'offline_valid': out.get('offline_replay', {}).get('valid')}), flush=True)


def run(pilot=False):
    dest = ROOT/'results/difficult_decisions_development' if pilot else DEST
    protocol = json.loads((dest/'protocol.json').read_text())
    for c in protocol['cases']:
        path = dest/(c['id']+'.json')
        if path.exists() and 'offline_replay' in json.loads(path.read_text()):
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()), 'worker', '--id', c['id']]
        if pilot:
            cmd.append('--pilot')
        try:
            completed = subprocess.run(cmd, text=True, capture_output=True, timeout=180,
                                       env={**os.environ, 'OPENBLAS_NUM_THREADS': '1',
                                            'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
            (dest/(c['id']+'.log')).write_text(completed.stdout+completed.stderr)
            print(completed.stdout.strip(), flush=True)
            if completed.returncode:
                dump(dest/(c['id']+'_failure.json'), {'returncode': completed.returncode,
                                                    'stderr': completed.stderr[-4000:]})
        except subprocess.TimeoutExpired:
            dump(dest/(c['id']+'_failure.json'), {'timeout_seconds': 180})
            print(json.dumps({'id': c['id'], 'timeout': True}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'worker'])
    parser.add_argument('--pilot', action='store_true'); parser.add_argument('--id')
    args = parser.parse_args()
    if args.action == 'prepare': prepare(args.pilot)
    elif args.action == 'run': run(args.pilot)
    else: worker(args.id, args.pilot)
