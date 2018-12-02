""" Uses trials and acc to simulate a behavioral experiment"""
import numpy as np
from copy import deepcopy
from modelmodel.behave import acc as acclib
from modelmodel.behave import probability
from modelmodel.behave import trials as tlib
from modelmodel.misc import process_prng


def random(N, k, prng=None):
    """Simulate random behavior

    Parameters
    ----------
    N : int
        Number of conditions
    k : int
        Number of trials / condition
    prng : RandomState object, None
        Explicit passing of random state.  None create
        a new state
    """
    prng = process_prng(prng)

    trials, prng = tlib.random(N, k, prng=prng)
    trials, prng = tlib.jitter(trials, prng=prng)
    l = trials.shape[0]

    acc = np.zeros(l)
    p = np.zeros(l)

    conds = np.unique(trials)
    for n in conds:
        ## Skip null trials
        if (n == 0) | (n == '0'):
            continue

        p_n, prng = probability.random(k, prng=prng)
        acc_n, prng = acclib.accuracy(p_n, prng=prng)

        for t in enumerate(trials):
            acc[trials == n] = acc_n
            p[trials == n] = p_n

    return trials, acc, p, prng


def learn(N, k, loc=3, prng=None):
    """Simulate learning behavior"""

    trials, _, _, prng = random(N, k, prng=prng)
    l = trials.shape[0]
    acc = np.zeros(l)
    p = np.zeros(l)
    conds = np.unique(trials)

    for n in conds:
        ## Skip null trials
        if (n == 0) | (n == '0'):
            continue

        p_n, prng = probability.learn(k, loc, prng=prng)
        acc_n, prng = acclib.accuracy(p_n, prng=prng)
        for t in enumerate(trials):
            acc[trials == n] = acc_n
            p[trials == n] = p_n

    return trials, acc, p, prng


# def some_learn(N, k, N_learn, loc, event=True, rand_learn=True, prng=None):
#     """
#     Creates 'uneven' acc, and p value distributions for k trials in
#     N conditions in the returned trials where N_learn is the number
#     of conditions that show learning (via sim_acc_learn()).
#
#     N minus N_learn condtions simulated data will be governed instead
#     by sim_acc_rand. If event is True an event-related trials is
#     created.
#     """
#     raise NotImplementedError("Needs DEBUG")
#
#     prng = process_prng(prng)
#     if N == N_learn:
#         raise ValueError('N_learn must be less than N.')
#     if N_learn <= 0:
#         raise ValueError('N_learn must be 1 or greater.')
#
#     if event:
#         trials, prng = event_randomtrial(N, k, mult=1, prng=prng)
#     else:
#         trials, prng = randomtrial(N, k, prng)
#
#     N_c = deepcopy(N)
#     if event:
#         N_c += 1
#
#     acc = [0, ] * (N_c*k)
#     p = [0, ] * (N_c*k)
#
#     names = list(set(trials))
#     for ii,n in enumerate(names):
#         ## Skip null trials
#         if (n == 0) | (n == '0'):
#             continue
#
#         ## How many trials/condition?
#         acc_n = []
#         p_n = []
#
#         ## How many trials/condition?
#         if ii <= N_learn:
#             print('Learning in iteration {0}.'.format(ii))
#             if rand_learn:
#                 acc_n, p_n, prng = random_learn(k, 1./N, loc, prng)
#             else:
#                 acc_n, p_n, prng = learn(k, loc, prng)
#         else:
#             acc_n, p_n, prng = randomacc(k,1./N)
#
#         for jj,t in enumerate(trials):
#             if t == n:
#                 acc[jj] = acc_n.pop(0)
#                 p[jj] = p_n.pop(0)
#
#     return trials, acc, p, prng
