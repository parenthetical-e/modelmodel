SHELL := /bin/bash

# A quick fast test
data/rw10.hdf5: models.ini
	python rw.py data/rw10.hdf5 \
		-N 10 \
		--n_trials 60 \
		--behave learn \
		--models models.ini 


# More realistic sims, both learn and random
rwBoth: data/rw500_l.hdf5 data/rw500_r.hdf5	

data/rw500_l.hdf5: moremodels.ini
	python rw.py data/rw500_l.hdf5 \
		-N 500 \
		--n_trials 60 \
		--behave learn \
		--models moremodels.ini 

data/rw500_r.hdf5: moremodels.ini
	python rw.py data/rw500_r.hdf5 \
		-N 500 \
		--n_trials 60 \
		--behave random \
		--models moremodels.ini 