"""Check participant contribution bounds and one-release safeguards on fixtures."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest

PATH=Path(__file__).resolve().parents[1]/'scripts/dp_acoustic_release.py'
spec=importlib.util.spec_from_file_location('dp_acoustic_release',PATH)
dp=importlib.util.module_from_spec(spec)
spec.loader.exec_module(dp)


def reference():
    """Use a fixed synthetic reference with eight possible cells."""
    w=np.zeros(dp.DIM);w[0]=1
    acoustic=np.zeros(dp.DIM);acoustic[1]=1
    return {'mean':np.zeros(dp.DIM),'scale':np.ones(dp.DIM),'weight':w,
            'bias':0.,'acoustic_direction':acoustic,'score_edges':np.array([-.5,0,.5]),
            'acoustic_edge':0.,'prototypes':np.zeros((8,dp.DIM)),'public_cell_counts':np.ones(8)}


def fixture():
    """Two participants, one exceeding the contribution cap."""
    x=np.zeros((40,dp.DIM));x[:,0]=np.linspace(-1,1,40);x[:,1]=1
    return x,np.arange(40)%2,np.array([0]*30+[1]*10)


def test_add_remove_person_sensitivity_and_cap():
    x,y,g=fixture();ref=reference();full=dp.histogram(x,y,g,ref)
    assert full.sum()==34
    for person in (0,1):
        mask=g!=person
        neighbor=dp.histogram(x[mask],y[mask],g[mask],ref)
        assert np.abs(full-neighbor).sum()<=24
    assert dp.histogram(x[:0],y[:0],g[:0],ref).sum()==0


def test_reject_invalid_labels():
    x,y,g=fixture();y[0]=2
    with pytest.raises(ValueError):dp.histogram(x,y,g,reference())


def test_release_has_no_private_rows_and_cannot_repeat(tmp_path):
    x,y,g=fixture();ref=tmp_path/'reference.npz'
    np.savez_compressed(ref,**reference())
    ledger=tmp_path/'private-ledger.json';out=tmp_path/'public'
    manifest=dp.release(x,y,g,ref,out,ledger)
    assert manifest['epsilon']==1 and manifest['delta']==0
    assert manifest['l1_sensitivity']==24
    assert len((out/'weights.csv').read_text().splitlines())==17
    assert set(p.name for p in out.iterdir())=={'weights.csv','MANIFEST.json'}
    assert 'participants' not in manifest and 'raw_counts' not in manifest
    with pytest.raises(FileExistsError):dp.release(x,y,g,ref,out,ledger)
    with pytest.raises(FileExistsError):dp.release(x,y,g,ref,tmp_path/'other',ledger)


def test_bins_depend_on_fixed_public_reference():
    x,y,g=fixture();ref=reference()
    assert np.array_equal(dp.bins(x[:5],ref),dp.bins(x,ref)[:5])
