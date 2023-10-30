# README: Environments

The current directory contains example Python code for running SQUID on several previously-trained and published deep learning models. The Python scripts will require user creation of a separate environment for each of the deep learning models analyzed in our work. Instructions for setting up these environments are provided below.

## Instructions:
First, install [Anaconda](https://docs.anaconda.com/anaconda/install), and follow the instructions below for one of three example deep learning models.

In general, if your deep learning model relies on an outdated version of Tensorflow (1.x), as is the case for BPNet and DeepSTARR, a separate environment for MAVE-NN (which uses Tensorflow 2.x) must be created. Given this conflict, the scripts in our pipeline must be run in a piecewise fashion: e.g., sourcing the 'bpnet' environment for running '1_locate_patterns' and '2_generate_mave.py', then deactivating the environment via 'source deactivate', and then sourcing the 'mavenn' environment for running '3_surrogate_modeling.py' and '4_analyze_outputs.py'


### GOPHER Environment:
The GOPHER environment should be used for the models ResidualBind-32 and Basenji-32. With Anaconda sourced, create a new Anaconda environment for GOPHER:

`conda create -n gopher python=3`

Next, activate this environment via `conda activate gopher`, and install the following packages:

- `conda install -c conda-forge tensorflow-gpu==2.8.0`
- `pip install --upgrade --upgrade-strategy only-if-needed mavenn`
- `conda install -c anaconda pandas`
- `python -m pip install --no-cache-dir https://github.com/p-koo/tfomics/tarball/master`
- `pip3 install --upgrade jax jaxlib==0.3.22`
- `pip install logomaker`
- `pip install -U --no-deps biopython`
- `pip install pyyaml`


### CAGI5 Environment:
For either GOPHER or ENFORMER-related inference of CAGI5 loci, the GOPHER environment detailed above can be used.


### DeepSTARR Environment:
With Anaconda sourced, create a new Anaconda environment for DeepSTARR:

`conda create --name deepstarr python=3.7 tensorflow=1.14.0 keras=2.2.4` #or tensorflow-gpu if you are using GPUs (may have to install separately below via pip)

Next, activate this environment via `conda activate deepstarr`, and install the following packages:

- `pip install kipoi`
- `pip install kipoiseq`
- `pip install h5py==2.10.0`
- `python -m pip install --no-cache-dir https://github.com/p-koo/tfomics/tarball/master`
- `pip install logomaker`
- `pip install deeplift==0.6.13.0`
- `python -m pip install https://github.com/kundajelab/shap/tarball/master` #for deeplift
- `pip3 install --upgrade protobuf==3.20.0` #possibly optional and system dependent


### BPNet Environment:
With Anaconda sourced, first create a new Anaconda environment for BPNet:

`conda create -n bpnet -c bioconda python=3.6 pybedtools bedtools pybigwig pysam genomelake`

Next, activate this environment via `conda activate bpnet`, and install the following packages:

- `pip install tensorflow~=1.0` #or tensorflow-gpu~=1.0 if you are using GPUs
- `pip install https://github.com/kundajelab/DeepExplain/tarball/master`
- `pip install https://github.com/kundajelab/bpnet/tarball/master`
- `python -m pip install --no-cache-dir https://github.com/p-koo/tfomics/tarball/master`
- `pip install logomaker`


### MAVE-NN Environment:
Creating a standalone environment for MAVE-NN is only required when using deep learning models that rely on Tensorflow 1.x (as described in the instructions above). To avoid issues when reading Pickle files, we also require that the version of `pandas` is identical between these environments. Also, make sure that Anaconda is up to date via `conda update -n base conda`

With Anaconda sourced, create a new Anaconda environment for MAVE-NN:

`conda create -n mavenn python=3.7`

Next, activate this environment via `conda activate mavenn`, and install the following packages:

- `pip install mavenn`
- `pip install mavenn --upgrade`
- `conda install -c anaconda pandas=1.3.5`
- `python -m pip install --no-cache-dir https://github.com/p-koo/tfomics/tarball/master`

Once created, this environment can be updated at any time to reflect new changes in the MAVE-NN repository via `pip install mavenn --upgrade`


---
Alternatively, we have provided `.yml` files for both Linux and Mac in the respective `a_model_assets/environment` folder. Create a conda environment from it as follows:
- `conda env create -f {file name}.yml`

Once the appropriate environment has been activated, run a script via `python {script name}.py`

Finally, when you are done using an environment, always exit via `conda deactivate`. If you have any issues installing MAVE-NN, please see:
- https://mavenn.readthedocs.io/en/latest/installation.html
- https://github.com/jbkinney/mavenn/issues

### GPUs:
For users who installed `tensorflow-gpu` in any of the above environments, we suggest setting CPU/GPU instructions in the CLI before running our scripts. For example, set `CUDA_VISIBLE_DEVICES=''` to disable GPUs or, for example, `CUDA_VISIBLE_DEVICES='7'` to enable GPU device 7.

Alternatively, users may be interested in running our scripts for a batch of sequences using GPUs in parallel. First, install the gpu scheduler in the proper environment required for the desired script via `pip install simple_gpu_scheduler`

Next, with the CLI in the same directory as the desired scripts, run one of the following; e.g.:
- `for i in {0..49}; do echo "python 2_generate_mave.py $i device=\$CUDA_VISIBLE_DEVICES && sleep 3"; done | simple_gpu_scheduler --gpus 0,1,2`
- `for i in {0..49}; do echo "python 3_surrogate_modeling.py $i device=\$CUDA_VISIBLE_DEVICES && sleep 3"; done | simple_gpu_scheduler --gpus 0,1,2`

If errors arise using `tensorflow-gpu~=1.0`, try instead using:
- `python -m pip install nvidia-pyindex`
- `python -m pip install nvidia-tensorflow`

Documentation is available at <https://github.com/NVIDIA/tensorflow>

## Attribution:
For the external assets contained in this directory, including deep learning models and corresponding scripts created externally to SQUID, please attribute the following:

**GOPHER / ResidualBind-32 and Basenji-32**
- Toneyan et al., "Evaluating deep learning for predicting epigenomic profiles", *bioRxiv*, 2022. https://github.com/shtoneyan/gopher

**DeepSTARR**
- Almeida et al., "DeepSTARR predicts enhancer activity from DNA sequence and enables the de novo design of synthetic enhancers", *Nature Genetics*, 2022.
- https://github.com/bernardo-de-almeida/DeepSTARR

**BPNet**
- Avsec et al., "Base-resolution models of transcription-factor binding reveal soft motif syntax", *Nature Genetics*, 2021.
- https://github.com/kundajelab/bpnet

**ENFORMER:
- Avsec et al., "Effective gene expression prediction from sequence by integrating long-range interactions", *Nature Methods*, 2021.
- https://github.com/google-deepmind/deepmind-research/tree/master/enformer

