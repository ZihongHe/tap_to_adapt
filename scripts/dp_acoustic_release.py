"""Build a public acoustic reference and a participant-private joint histogram.

The reference uses only the authorized pilot corpus. Private inputs must be
consent-filtered raw pooled features, not standardized replay vectors. A user
contributes at most 24 feature-label pairs. Independent Laplace noise with
scale 24/epsilon protects the joint histogram under add/remove-user adjacency.
No private feature, label row, timestamp, group identifier, or exact total is
written. The output contains 16 noisy weights for eight public prototypes.

The release function requires a private, exclusive ledger file. Re-running a
release is an additional disclosure and must be budgeted, never done to select
more favorable noise. Library default randomness is checked to be SystemRandom.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

DIM=10752
CAP=24
CELLS=8


def normalized(x: np.ndarray) -> np.ndarray:
    """Normalize raw temporal mean/std features without fitting private values."""
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=DIM or not np.isfinite(x).all():
        raise ValueError('Expected finite raw pooled features of dimension 10752')
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)


def bins(x: np.ndarray, reference: dict) -> np.ndarray:
    """Assign samples using boundaries learned exclusively from public data."""
    z=(normalized(x)-reference['mean'])/reference['scale']
    score=z@reference['weight']+reference['bias']
    acoustic=z@reference['acoustic_direction']
    return (2*np.searchsorted(reference['score_edges'],score,side='right')+
            (acoustic>=reference['acoustic_edge']).astype(np.int64))


def build_reference(public_features: Path, target: Path) -> None:
    """Fit a fixed four-score-bin by two-acoustic-bin public reference."""
    if target.exists():
        raise FileExistsError('Public reference already exists')
    with np.load(public_features,allow_pickle=False) as source:
        raw,y=source['features'],source['labels']
    if len(raw)!=1000 or not np.isin(y,[0,1]).all():
        raise ValueError('Reference requires the complete authorized pilot corpus')
    with threadpool_limits(limits=2):
        scaler=StandardScaler().fit(normalized(raw))
        z=scaler.transform(normalized(raw))
        clf=LogisticRegression(C=.1,max_iter=2000,random_state=731).fit(z,y)
        pca=PCA(n_components=1,svd_solver='randomized',random_state=731).fit(z)
        ref={'mean':scaler.mean_,'scale':scaler.scale_,'weight':clf.coef_[0],
             'bias':clf.intercept_[0], 'acoustic_direction':pca.components_[0],
             'score_edges':np.quantile(clf.decision_function(z),[.25,.5,.75]),
             'acoustic_edge':np.median(z@pca.components_[0])}
        code=bins(raw,ref)
        counts=np.bincount(code,minlength=CELLS)
        if np.any(counts==0):
            raise ValueError('Empty public cell; reference specification must be reviewed before private access')
        ref['prototypes']=np.stack([raw[code==k].mean(axis=0) for k in range(CELLS)])
        ref['public_cell_counts']=counts
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:
        np.savez_compressed(f,**ref)


def histogram(x: np.ndarray,y: np.ndarray,groups: np.ndarray,reference: dict) -> np.ndarray:
    """Bound each participant's entire contribution before aggregation.

    If a caller supplies more than 24 rows for a person, only that person's
    first 24 rows are retained. Other participants' selected rows are unaffected.
    The integer query has L1 sensitivity at most 24 for add/remove adjacency.
    """
    y=np.asarray(y);groups=np.asarray(groups)
    if y.shape!=(len(x),) or groups.shape!=y.shape or not np.isin(y,[0,1]).all():
        raise ValueError('Invalid labels or participant grouping')
    if groups.dtype.kind not in 'iuUS':
        raise ValueError('Group identifiers must be integer or string values')
    selected=[]
    for user in np.unique(groups):
        selected.extend(np.flatnonzero(groups==user)[:CAP])
    if not selected:
        return np.zeros((CELLS,2),dtype=np.int64)
    selected=np.asarray(selected,dtype=np.int64)
    cell=bins(x[selected],reference)
    return np.bincount(2*cell+y[selected].astype(np.int64),minlength=CELLS*2).reshape(CELLS,2)


def release(x: np.ndarray,y: np.ndarray,groups: np.ndarray,reference_path: Path,
            output: Path,ledger: Path,epsilon: float=1.0) -> dict:
    """Write one DP result; keep the ledger outside any public repository."""
    from diffprivlib.mechanisms import Laplace
    if not np.isfinite(epsilon) or not 0<epsilon<=1:
        raise ValueError('This release protocol permits epsilon in (0,1]')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refusing to overwrite or resample an existing release')
    ref_hash=hashlib.sha256(reference_path.read_bytes()).hexdigest()
    with ledger.open('x',encoding='utf-8') as f:
        json.dump({'status':'reserved','epsilon':epsilon,'delta':0,'reference_sha256':ref_hash},f)
    with np.load(reference_path,allow_pickle=False) as z:
        ref={k:z[k] for k in z.files}
    h=histogram(x,y,groups,ref)
    mechanism=Laplace(epsilon=epsilon,delta=0,sensitivity=CAP,random_state=None)
    if not isinstance(mechanism._rng,random.SystemRandom):
        raise RuntimeError('Cryptographic system randomness is required')
    # The vector's L1 sensitivity is CAP; shared noise scale CAP/epsilon gives
    # epsilon for the entire vector, not epsilon independently spent per cell.
    noisy=np.array([mechanism.randomise(int(v)) for v in h.flat]).reshape(CELLS,2)
    noisy=np.maximum(noisy,0)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'weights.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['prototype_id','correction_label','noisy_weight'])
        for cell in range(CELLS):
            for label in (0,1):writer.writerow([cell,label,format(noisy[cell,label],'.12g')])
    import diffprivlib
    info={'schema':'tap-to-adapt-public-prototype-dp-weights/v1',
          'data_type':'Differentially private joint statistics of real corrective-feedback records; not individual or synthetic observation rows.',
          'epsilon':epsilon,'delta':0,'adjacency':'Add or remove all contributed records of one participant.',
          'replacement_epsilon_bound':2*epsilon,'maximum_rows_per_participant':CAP,
          'l1_sensitivity':CAP,'laplace_scale':CAP/epsilon,'cells':CELLS,'label_values':[0,1],
          'label_semantics':{'0':'Manual interruption, associated with response-initiation context.','1':'Manual activation.'},
          'reference_sha256':ref_hash,'reference_source':'Authorized public 1000-record pilot corpus only.',
          'postprocessing':'Negative noisy cell weights clipped to zero. Weights are not exact counts.',
          'sampler':'diffprivlib.mechanisms.Laplace with default SystemRandom; no stored noise seed.',
          'diffprivlib_version':diffprivlib.__version__,
          'composition':'One released histogram. All future releases must be composed. The guarantee applies to this mechanism and does not undo earlier disclosures.',
          'weights_sha256':hashlib.sha256((output/'weights.csv').read_bytes()).hexdigest()}
    (output/'MANIFEST.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
    ledger.write_text(json.dumps({'status':'released_locally','epsilon':epsilon,'delta':0,
                     'reference_sha256':ref_hash,'weights_sha256':info['weights_sha256']},indent=2)+'\n',encoding='utf-8')
    return info


def main() -> None:
    """Build only the public reference from CLI; private release is an API call."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-features',type=Path,required=True)
    parser.add_argument('--output-reference',type=Path,required=True)
    args=parser.parse_args()
    build_reference(args.public_features,args.output_reference)
    print('Public acoustic reference saved; no private records accessed.')


if __name__=='__main__':
    main()
