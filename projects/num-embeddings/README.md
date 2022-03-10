# On Embeddings for Numerical Features in Tabular Deep Learning<!-- omit in toc -->

This is the official implementation of the paper "On Embeddings for Numerical Features in Tabular Deep Learning" (the link will be added once the paper becomes available on arXiv).

---
- [The main results](#the-main-results)
- [Code overview and technical notes](#code-overview-and-technical-notes)
- [Setup the environment](#setup-the-environment)
    - [Software](#software)
    - [Data](#data)
- [Example](#example)
- [How to cite](#how-to-cite)

---

## The main results

See [bin/results.ipynb](./bin/results.ipynb).

## Code overview and technical notes

The code is organized as follows:
- `bin`
    - `train4.py` for neural networks
    - `xgboost_.py` for XGBoost
    - `catboost_.py` for CatBoost
    - `tune.py` for tuning
    - `evaluate.py` for evaluation
    - `ensemble.py` for ensembling
    - `results.ipynb` for summarizing results
    - `synthetic.py` for generating the synthetic GBDT-friendly datasets
    - `train1_synthetic.py` for the experiments with synthetic data
    - `datasets.py` was used to build the dataset splits
- `lib` contains common tools used by programs in `bin`
- `exp` contains experiment configs and results (metrics, tuned configurations, etc.)

Technical notes:
- for saving and loading configs, use `lib.dump_config` and `lib.load_config` instead of bare TOML libraries
- ignore `bin/train{0,1,3}.py`. They are older versions of `bin/train4.py` used by some experiments. These older versions provide only some subsets of features (importantly, the implementations of those features are **identical** to their implementations in `train4.py`).
- You must explicitly set `CUDA_VISIBLE_DEVICES` when running scripts

## Setup the environment

### Software

Preliminaries:
- You may need to change the CUDA-related commands and settings below depending on your setup
- Make sure that `/usr/local/cuda-11.1/bin` is always in your `PATH` environment variable
- Install [conda](https://docs.conda.io/en/latest/miniconda.html)

```bash
export REPO_DIR=<ABSOLUTE path to the REPOSITORY root>
# example: export REPO_DIR=/home/myusername/repositories/rtdl-num-embeddings
export PROJECT_DIR="${REPO_DIR}/projects/num-embeddings"
git clone https://github.com/Yura52/rtdl $REPO_DIR
cd $PROJECT_DIR

conda create -n rtdl-num-embeddings python=3.9.7
conda activate rtdl-num-embeddings

pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

# if the following commands do not succeed, update conda
conda env config vars set PYTHONPATH=${PYTHONPATH}:${PROJECT_DIR}
conda env config vars set PROJECT_DIR=${PROJECT_DIR}
# the following command appends ":/usr/local/cuda-11.1/lib64" to LD_LIBRARY_PATH;
# if your LD_LIBRARY_PATH already contains a path to some other CUDA, then the content
# after "=" should be "<your LD_LIBRARY_PATH without your cuda path>:/usr/local/cuda-11.1/lib64"
conda env config vars set LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-11.1/lib64
conda env config vars set CUDA_HOME=/usr/local/cuda-11.1
conda env config vars set CUDA_ROOT=/usr/local/cuda-11.1

# (optional) get a shortcut for toggling the dark mode with cmd+y
conda install nodejs
jupyter labextension install jupyterlab-theme-toggle

conda deactivate
conda activate rtdl-num-embeddings
```

### Data

LICENSE: by downloading our dataset you accept licenses of all its components. We do not impose any new restrictions in addition to those licenses. You can find the list of sources in the paper.

```bash
cd $PROJECT_DIR
wget "https://www.dropbox.com/s/r0ef3ij3wl049gl/data.tar?dl=1" -O rtdl_num_embeddings_data.tar
tar -xvf rtdl_num_embeddings_data.tar
```

## Example

The code below reproduces the results for MLP on the California Housing dataset. The pipeline for other algorithms and datasets is absolutely the same.

```
# You must explicitly set CUDA_VISIBLE_DEVICES if you want to use GPU
export CUDA_VISIBLE_DEVICES="0"

# Create a copy of the 'official' config
cp exp/mlp/california/0_tuning.toml exp/mlp/california/1_tuning.toml

# Run tuning (on GPU, it takes ~30-60min)
python bin/tune.py exp/mlp/california/1_tuning.toml

# Evaluate single models with 15 different random seeds
python bin/evaluate.py exp/mlp/california/1_tuning 15

# Evaluate ensembles (by default, three ensembles of size five each)
python bin/ensemble.py exp/mlp/california/1_evaluation

# Then use bin/results.ipynb to view the obtained results
```

## How to cite
```
@article{gorishniy2022embeddings,
    title={On Embeddings for Numerical Features in Tabular Deep Learning},
    author={Yury Gorishniy and Ivan Rubachev and Artem Babenko},
    journal={arXiv},
    volume={???},
    year={2022},
}
```
