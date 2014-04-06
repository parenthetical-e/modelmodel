import numpy as np
import pandas as pd
from modelmodel import dm
from modelmodel import hrf

def test_convolve_hrf():
    # Create HRF to convolve with
    dg = hrf.double_gamma(
            width=32, TR=1, a1=6.0, a2=12., b1=0.9, b2=0.9, c=0.35
            )
    
    # -----------
    # 1d convolve
    # -----------
    dm1 = np.zeros(32)
    dm1[0] = 1
    assert np.allclose(dm.convolve_hrf(dm1, dg), dg), (
            "1d convolve doesn't match template")
    assert np.allclose(dm.convolve_hrf(dm1, dg, [0]), dg), (
            "1d convolve cols malfunction")

    # -----------
    # 2d convolve
    # -----------
    dm1 = np.zeros([32, 2])
    dm1[0,:] = 1
    
    # Basic 2d
    assert np.allclose(dm.convolve_hrf(dm1, dg), np.vstack([dg,dg]).T), (
            "2d doesn't match template")

    # Parial dm
    assert np.allclose(dm.convolve_hrf(dm1[:,0], dg), dg), (
            "2d (col 0) convolve doesn't match template")
    
    # Select cols
    assert np.allclose(dm.convolve_hrf(dm1, dg, [0]), dg[:,np.newaxis]), (
            "2d cols (0) malfunction")
    assert np.allclose(dm.convolve_hrf(dm1, dg, [1]), dg[:,np.newaxis]), (
            "2d cols (1) malfunction")
    assert np.allclose(dm.convolve_hrf(dm1, dg, [0,1]), np.vstack([dg,dg]).T), (
            "2d cols (0,1) malfunction")


def test_orthogonalize():
    # --
    # np
    # --
    # Two lines:
    dm1 = np.vstack([np.arange(10), np.arange(10)]).T
    dmo = dm.orthogonalize(dm1, [0,1])
    
    # Col 0 is unchanged
    # Orth col 1 is all zeros as all 
    # variance was assigned to 0.
    assert np.allclose(dmo[:,0], dm1[:,0])
    assert np.allclose(dmo[:,1], np.zeros_like(dmo[:,1]))

    # -----------------    
    # repeat for pandas
    # -----------------
    dm1 = pd.DataFrame(dm1, columns=['0', '1'])
    dmo = dm.orthogonalize(dm1, ['0','1'])
    assert np.allclose(dmo['0'].values, dm1['0'].values)
    assert np.allclose(dmo['1'].values, np.zeros_like(dmo['1'].values))
    
    