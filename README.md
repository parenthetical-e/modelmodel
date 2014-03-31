`modelmodel` is a python package for *creating programs* to do detailed, and if we're optimistic, quantitative analyses of BOLD timecourses - real and simulated. 

The focus is on model-based (i.e. parametric) designs.

It is still *very much* a work in progress.

---

I've written the analysis tool I have always wanted.  

Specifically:

* Designed as a programmer's analysis package first.  I wanted a clean but powerful and pythonic way to do a fMRI analysis. So that is what I tried to write.
* Designed to make specifying a model-based design trivial; It's the only simulation environment focused on model-based designs
* If you need to integrate computational model parameter fits, it can do that too.
* Has builtin access to over 500 anatomical ROIs from 8 separate atlases.
* It's trivial to add your own (functional) ROIs.
* Model-comparison is the default approach, with AIC the favored statistic.  But BIC, F-values, and other others are supported. 
* It's very easy to swap in sophisticated regression techniques in place of OLS. Any regression method from statsmodels will do.
* It's the only fMRI simulation environment for the python programming language, as least as far as I am aware (if this is wrong, please let me know). *Note*: There are very nice systems for R (neuRosim) and MATLAB (simTB  - which looks quite fantastic these days).

That said, I really hope it is useful to you.

---

This is not a beginners tool.

- I expect you have a solid grasp of fMRI analysis methods.
- I expect you can program in python.
- I expect you can preprocess the data elsewhere. Data must be in Nifti1 (and MNI152/352 space, to use the ROI features).

If any of these expectations are not met, you are going to have a bad time.
