"""Misc or helper functions for behave"""
import numpy as np


def map_to_last_intra(conditions,impulses,num_intra_event,terminal=True):
	"""
	This allows reuse of simBehave.acc.* routines for intra-event 
	conditions by mapping impulses (which should be in the same temporal 
	space as conditions) to the last intra-event in each intra-event set.  

	Note: this follows the same intra-event scheme as simBehave.trials.
	intra_event()
	"""
	## Doing this seperatly from intra_event() (above) is certainly 
	## wasteful but will, I hope, lead to a cleaner/simpler user 
	## experience.

	intra_impulses = []
	for imp,cond in zip(impulses,conditions):
		if (cond == 0) | (cond == '0'):
			intra_impulses.extend([0,]*num_intra_event)
				# Null conditions do not have terminal states/conditions
		else:
			if terminal:
				intra_impulses.extend([0,]*(num_intra_event-1)+[imp,0])
					# tack on a terminal state/conditions, if needed.
			else:
				intra_impulses.extend([0,]*(num_intra_event-1)+[imp,])

	return intra_impulses
