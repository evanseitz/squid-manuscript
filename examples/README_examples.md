# README: Examples

The current directory contains code for running the SQUID surrogate modeling framework using the predictions from several example deep learning models.

Altogether, this example pipeline applies surrogate models (e.g., additive or pairwise models) based on deep learning model predictions (e.g., ResidualBind-32) for the different analyses applied in the SQUID paper.

The most involved of these tasks is the *attribution error* analysis. For this, we apply surrogate models individually on a collection of the N highest ranking loci (e.g., ranked using saliency scores) corresponding to a user-defined recognition site (e.g., AP-1) or pairs of recognition sites (e.g., AP-1/AP-1) present in a cell line, with each instance also varying based on number of core mutations and distance between motif pairs. The result is a collection of N surrogate models, which can analyzed independently to analyze biological mechanisms or as an ensemble to quantify statistical properties.

Briefly, the pipeline is as follows:
- 1_locate_patterns.py: script for discovery of single or paired instances of user-defined recognition sites within a cell-type specific genome. Users can define the maximum number of core mutations present in the desired recognition site, as well as maximum distance between pairs.
- 2_generate_mave.py: script for generating an ensemble of mutagenized genomic sequences in conjunction with respective predictions from a user-defined deep learning model to form a MAVE dataset.
- 3_surrogate_modeling.py: script to perform surrogate modeling on the ensemble of mutagenized genomic sequences generated in the previous script, with options to focus modeling on specific recognition site instances in a given sequence.
- 4_analyze_outputs.py: script for comparing and analyzing statistical and biophysical properties of attribution maps produced in the previous script.

The above Python scripts are intended to be run in order, starting with '1_locate_patterns.py'. Instructions for running each module are contained within the corresponding script, along with detailed comments. All of these scripts rely on the standalone script 'set_parameters.py' for loading in hyperparameters. These hyperparameters should be set by the user before running the four scripts above. After this initial setup, folders will be created (e.g., in the `examples_GOPHER` directory) containing intermediate and final outputs.