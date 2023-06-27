# Computational Cognitive Science 3
This repo was part of the exam in Computational Cognitive Science 3 at the University of Copenhagen.

# Abstract
In this study, we present an adjusted decoder (Yot- sutsuji et al., 2021) that demonstrates proper per- formance in extracting words from a novel multilin- gual fMRI scanning with naturalistic stimuli (Li et al., 2022), outperforming random prediction in both inter- and cross-subject configurations. Our discussion suggest that successful decoding of small fMRI sample sizes requires a decoder carefully designed to learn quickly. We also explore the differences in decoder effectiveness for Chinese and English from a linguistic perspective, and discuss the implications of further extending the current work, including the potential for adjusting the controversial label annotations. Overall, our results represent a promising first step towards investigating linguistic fMRI data using a deep learning approach.

# Dependencies
pytorch, nibabel, nilearn, numpy

# Dataset 
Data can be acquired here: https://openneuro.org/datasets/ds003643/versions/2.0.1
Please put the data under the raw_data/ folder

# Resources
The NiLearn package docs:
https://nilearn.github.io/stable/index.html
fMRI preproccesing tool:
https://github.com/nipreps/fmriprep
NiLearn example:
https://nilearn.github.io/stable/auto_examples/02_decoding/plot_haxby_glm_decoding.html#sphx-glr-auto-examples-02-decoding-plot-haxby-glm-decoding-py
.nii file explanation:
https://nipy.org/nibabel/coordinate_systems.html 
http://www.newbi4fmri.com/tutorial-1u-data 