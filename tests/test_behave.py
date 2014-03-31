from modelmodel import behave
import numpy as np

SEED=45
def test_trials_trials():
    prng = np.random.RandomState(SEED)
    
    # Simplet trials is a trial: [1, ]
    trials, prng = behave.trials.random(N=1, k=1, prng=prng)
    assert np.allclose(trials, np.array([1,])), (
        "simplest trials breaks")
    
    # Does k work right?
    trials, prng = behave.trials.random(N=1, k=10, prng=prng)
    assert np.allclose(np.sum(trials), 10), "k is off"
    
    # N?
    trials, prng = behave.trials.random(N=2, k=1, prng=prng)
    assert np.allclose(np.sum(trials), 3), "N is off"
    
    # is L ok?
    assert trials.shape[0] == 2, "l (N*k) is off"
    

def test_trials_jitter():
    prng = np.random.RandomState(SEED)
    
    # Jitter should not change N, k
    trials, prng = behave.trials.random(2, 2, prng=prng)
    trials, prng = behave.trials.jitter(trials, prng=prng)
    assert np.allclose(np.sum(trials), 6), "N of k is off"


def test_probability_random():
    prng = np.random.RandomState(SEED)
    
    # Random ps should avg to 0.5
    trials, prng = behave.trials.random(1, 1000, prng=prng)
    l = trials.shape[0]
    ps, prng = behave.probability.random(l, prng=prng)
    assert np.allclose(np.mean(ps), .5, atol=.05), "Bad avg"
    
    # dim check
    assert ps.shape[0] == trials.shape[0], "l off"
    
    # Same avg for N > 1 conds
    trials, prng = behave.trials.random(3, 1000, prng=prng)
    l = trials.shape[0]
    ps, prng = behave.probability.random(l, prng=prng)
    assert np.allclose(np.mean(ps), .5, atol=.05), "Bad avg with 3 cond"
    
    # dim check
    assert ps.shape[0] == trials.shape[0], "l off"
    

def test_probability_learn():    
    prng = np.random.RandomState(SEED)
    
    # Vis
    trials, prng = behave.trials.random(1, 20, prng=prng)
    l = trials.shape[0]
    ps, prng = behave.probability.learn(l, loc=3, prng=prng)
    
    # dim check
    assert ps.shape[0] == trials.shape[0], "l off"
    print(ps)
    
    # ps should avg to more than 0.5
    trials, prng = behave.trials.random(1, 1000, prng=prng)
    l = trials.shape[0]
    ps, prng = behave.probability.learn(l, loc=3, prng=prng)
    assert np.mean(ps) > .5, "Bad avg"
    
    # dim check
    assert ps.shape[0] == trials.shape[0], "l off"


def test_acc_accuracy():
    prng = np.random.RandomState(SEED)

    # For lots of random ps acc should avg to 0.5
    k = 5000
    ps = np.asarray([0.5] * k)
    acc, prng = behave.acc.accuracy(ps, prng=prng)
    assert np.allclose(np.sum(acc)/float(k), .5, atol=.05) 
    
    # dim check
    assert ps.shape == acc.shape, "l off"


def test_behave_random():
    prng = np.random.RandomState(SEED)
    trials, acc, ps, prng = behave.behave.random(N=1, k=5, prng=prng)
    
    # dim check
    assert trials.shape == acc.shape, "l off: trials and acc"
    assert trials.shape == ps.shape, "l off: trials and ps"
    
    # Check trials comp, then ps and acc avg
    k = 3000
    trials, acc, ps, prng = behave.behave.random(N=1, k=k, prng=prng)
    assert np.allclose(np.sum(trials), k), "k is off"
    assert np.allclose(np.unique(trials), np.array([0, 1])), "N if off"
    assert np.allclose(np.mean(ps[ps > 0.0]), .5, atol=.05), "Bad avg"
   
   
def test_behave_learn():
    prng = np.random.RandomState(SEED)
    trials, acc, ps, prng = behave.behave.learn(N=1, k=5, prng=prng)
    
    # dim check
    assert trials.shape == acc.shape, "l off: trials and acc"
    assert trials.shape == ps.shape, "l off: trials and ps"
    
    # Check trials comp, then ps and acc avg
    k = 3000
    trials, acc, ps, prng = behave.behave.learn(N=1, k=k, prng=prng)
    assert np.allclose(np.sum(trials), k), "k is off"
    assert np.allclose(np.unique(trials), np.array([0, 1])), "N if off"
    assert np.mean(ps[ps > 0.0]) > .5, "Bad avg"
    