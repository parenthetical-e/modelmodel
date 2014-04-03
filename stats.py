import numpy as np


def conjunction(stats, ps=None):
    """Run a conjunction test.
    
    Parameters
    ----------
    stats : np.array([float, ...])
        A list of statistics, e.g *t*-values.
    ps : np.array([float, ...]), None
        A list of (optional) p-values matching elements 
        in stats.
    
    Notes
    -----
    A too brief history of conjunction:
    
    * Price and Friston (1997) first suggested the idea of a conjunction for
    use in fMRI. They implemented it by looking for significance for `A` (
    i.e reject `A = 0`), and `B` (`B = 0`) but not the interaction `A-B` 
    (`A - B = 0`)).  
     - Or said in english, a conjunction can be found where A and B are 
     both active but they are not different from each other.
     
    * Friston et al (1999, 199a) took a different approach, using
    minimum statistics that wanted to know whether `A > 0 OR B > 0`
    (or as null `A = 0 AND B = 0`.)
    
    * Nichols et al (2005) pointed out that the null we want to test is
    really `A > 0 AND B > 0` (or as null `A = 0 OR B = 0`), and implemented 
    a way to do that. It more-or-less reduces down to 
    `test_stat = np.min(stats)`.  
     - This approach is used here.
    
    * Friston, Penny and Glaser (2005) replied that while Nichols is right
    in their idea, the test design is too conservative. They offered an
    alternative.
    
    Given this package is designed for model-based comparisons, and
    as I (EJP) have concerns about the precision and utility of NHT 
    for relating computational models to BOLD signals, the conservative
    Nichols seemed best. 
    
    Sorry, Karl.
    

    Citation
    -------
    Nichols TE, el al, Valid conjunction inference with the minimum 
    statistic, Neuroimage, 25, 653-660 (2005).
    """
    
    # So many words for so simple a test
    min_s = np.min(stats)
    min_p = ps[stats == min_s][0]
    
    return min_s, min_p
    
    