"""Create lists of randomized trials."""
# import os
import numpy as np

from modelmodel.misc import process_prng


def isi(trials1, trials2, code=0, fraction=0.5, jit=(1,4), prng=None):
    """Add ISI jitter between time-locked events trials1 and trials2"""

    prng = process_prng(prng)
    jittimes = np.arange(*jit, dtype=np.int)

    isi1, isi2 = [], []
    for t1, t2 in zip(trials1, trials2):
        # Add ISI?
        if (t1 > code) or (t2 > code):
            prng.shuffle(jittimes)
            jit = [code, ] * jittimes[0]

            t1_isi = [t1] + jit
            t2_isi = jit + [t2]

            isi1.extend(t1_isi)
            isi2.extend(t2_isi)
        # Otherwise maintain ITI
        else:
            isi1.append(t1)
            isi2.append(t2)

    return isi1, isi2, prng


def jitter(trials, code=0, fraction=.5, jit=(1,7), prng=None):
    """Introduce random (uniform) jitter to trials"""

    prng = process_prng(prng)

    jittimes = np.arange(*jit, dtype=np.int)
    jittered = []
    for t in trials:
        if fraction >= prng.rand(1):
            prng.shuffle(jittimes)
            jt = [t, ] + [code, ] * jittimes[0]
            jittered.extend(jt)
        else:
            jittered.append(t)

    return np.asarray(jittered), prng


def random(N, k, prng=None):
    """Creates a randomly list of trials (int) of N cond with
    k trials / cond.
    """

    prng = process_prng(prng)

    trials = []
    [trials.extend([n, ] * k) for n in range(1, N+1)]

    trials = np.asarray(trials)
    prng.shuffle(trials)

    return trials, prng


