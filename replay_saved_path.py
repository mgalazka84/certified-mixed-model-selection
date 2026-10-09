"""Replay a saved path certificate without rerunning any optimization.

Example: python3 replay_saved_path.py results/path/suppressor_914360.json
"""
import argparse
import json
from pathlib import Path
import numpy as np
from certified_path import replay_path
from certified_global_path import replay_global_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result',type=Path)
    parser.add_argument('--data',type=Path,help='Original NPZ arrays when the result filename has a backend suffix')
    parser.add_argument('--backend',choices=['generic','crossed'],default='generic')
    args=parser.parse_args()
    record=json.loads(args.result.read_text())
    data=np.load(args.data or args.result.with_name(args.result.stem+'_data.npz'),allow_pickle=False)
    result=record['verified_path']
    arrays=data['y'],data['base'],data['x'],[data['z0'],data['z1']]
    if result.get('domain')=='nonnegative_orthant':
        factory=None
        if args.backend=='crossed':
            from certified_crossed_backend import CrossedArbGaussian
            factory=CrossedArbGaussian
        checked=replay_global_path(*arrays,result,verifier_factory=factory)
    else:
        if args.backend!='generic':
            parser.error('The crossed backend is exposed here for global-domain witnesses only')
        checked=replay_path(*arrays,result)
    print(json.dumps(checked,indent=2))
    if not checked['valid']:
        raise SystemExit(1)


if __name__=='__main__':main()
