SHELL := /bin/bash
BINPATH=~/Code/modelmodel/bin

# -----------------
# Save some behave data
# -----------------
behave50: data/behave50_l.csv data/behave50_r.csv
	
data/behave50_l.csv:
	python $(BINPATH)/behave.py \
	data/behave50_l.csv \
	-N 50 \
	--behave learn \
	--n_cond 1 \
	--n_trials 60 \

data/behave50_r.csv:
	python $(BINPATH)/behave.py \
	data/behave50_r.csv \
	-N 50 \
	--behave random \
	--n_cond 1 \
	--n_trials 60 \


# -----------------
# A quick fast test
# -----------------
data/rw10.hdf5: models.ini
	python rw.py data/rw10.hdf5 \
		-N 10 \
		--n_trials 60 \
		--behave learn \
		--models models.ini 


# --------------------------------
# Save all behave data (for debug)
# --------------------------------
rwBeh: data/rw50_l_beh.hdf5 data/rw50_r_beh.hdf5
	
data/rw50_l_beh.hdf5: models.ini
	python rw.py data/rw50_l_beh.hdf5 \
		-N 50 \
		--n_trials 60 \
		--behave learn \
		--models models.ini \
		--save_behave True

data/rw50_r_beh.hdf5: models.ini
	python rw.py data/rw50_r_beh.hdf5 \
		-N 50 \
		--n_trials 60 \
		--behave random \
		--models models.ini \
		--save_behave True


# --------------------
# Increase iteration n
# --------------------
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

# ------------
# Extract data
# ------------
data/rw_*.csv:
	python $(BINPATH)/extract.py \
		--hdf data/rw10.hdf5 \
		--names data/rw_fvalue.csv data/rw_pvalue.csv \
		--paths /*/*/tests/fvalue /*/*/tests/pvalue \
		--dims 0 0
