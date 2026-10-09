"""Replay the exported stage-eight family witnesses without optimization."""
from pathlib import Path
import hashlib,json
import numpy as np
from certified_path import BlockGaussian
from certified_global_path import TailArbGaussian,verify_global_decision
from certified_crossed_backend import CrossedArbGaussian

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'results/difficult_decisions'
protocol=json.loads((DEST/'protocol.json').read_text())
inputs=json.loads((DEST/'input_sha256.json').read_text())
for name,digest in json.loads((DEST/'code_sha256.json').read_text()).items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
checks=[]
for case in protocol['cases']:
    file=DEST/(case['id']+'.npz')
    assert hashlib.sha256(file.read_bytes()).hexdigest()==inputs[file.name]
    with np.load(file) as f:a={k:f[k] for k in f.files}
    result_file=DEST/(case['id']+'.json')
    result=json.loads(result_file.read_text())
    model=BlockGaussian(a['y'],a['base'],a['x'],[a[k] for k in sorted(a) if k.startswith('z')])
    online=result['methods']['verified'];witness=online['witness']
    # The generic checker checks an arbitrary specified family. This replay also
    # verifies that no member of the prescribed single-addition family is missing.
    assert witness['family']==[[],*[[j] for j in range(model.p)]]
    assert witness['added_sizes']==[0,*[1]*model.p]
    factory=CrossedArbGaussian if model.q==2 else TailArbGaussian
    checked=verify_global_decision(model,witness,factory(model,128))
    assert checked['valid']==result['offline_replay']['valid']
    assert online['status']!='verified' or checked['valid']
    checks.append({'id':case['id'],'complete_family_verified':True,
                   'primary_result_sha256':hashlib.sha256(result_file.read_bytes()).hexdigest(),
                   'online_status':online['status'],'replay':checked})
out={'cases':len(checks),'online_accepted':sum(c['online_status']=='verified' for c in checks),
     'replay_valid':sum(c['replay']['valid'] for c in checks),'checks':checks,
     'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
assert out['cases']==55 and out['online_accepted']==42 and out['replay_valid']==51
(DEST/'fresh_replay_checks.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='checks'},indent=2))
