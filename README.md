# Revisiting Tabular Deep Learning<!-- omit in toc -->
This repository contains:
- the official implementation of the paper "Revisiting Deep Learning Models for Tabular Data" ([link](https://arxiv.org/abs/2106.11959))
- `rtdl` (**R**evisiting **T**abular **D**eep **L**earning):
  - It is a PyTorch-based package that provides a user-friendly API for the main models (FT-Transformer, ResNet, MLP) used in the paper
  - It can be used by practitioners looking for Deep Learning models for tabular data
  - It can serve as a source of baselines for researchers (**excluding FT-Transformer**, see the warning below)
  - You can follow releases by hitting "Watch" / "Custom" / "Releases" in the right upper corner of the GitHub interface
  - See the [website](https://yandex-research.github.io/rtdl) for more details
  - See the [discussion](https://github.com/yandex-research/rtdl/discussions/1)

**Warning**: if you are a *researcher* (not a practitioner) and plan to use the
FT-Transformer model as a baseline in your paper, please, use the original implementation
from [ft_transformer.py](bin/ft_transformer.py). We will remove this limitation soon
(i.e. `rtdl` will become the recommended way to use FT-Transformer in papers).

The rest of this document is dedicated to the implementation of the paper.

---

*Note that the paper reports results based on **thousands** of experiments, so there can be rough edges in the implementation. Feel free to open [issues](https://github.com/yandex-research/rtdl/issues) and ask questions in [discussions](https://github.com/yandex-research/rtdl/discussions).*

- [1. The main results](#1-the-main-results)
- [2. Overview](#2-overview)
- [3. Setup the environment](#3-setup-the-environment)
  - [3.1. PyTorch environment](#31-pytorch-environment)
  - [3.2. TensorFlow environment](#32-tensorflow-environment)
  - [3.3. Data](#33-data)
- [4. Tutorial (how to reproduce results)](#4-tutorial-how-to-reproduce-results)
  - [4.1. Check the environment](#41-check-the-environment)
  - [4.2. Tuning](#42-tuning)
  - [4.3. Evaluation](#43-evaluation)
  - [4.4. Ensembling](#44-ensembling)
  - [4.5. "Visualize" results](#45-visualize-results)
  - [4.6. What about other models and datasets?](#46-what-about-other-models-and-datasets)
- [5. How to work with the repository](#5-how-to-work-with-the-repository)
  - [5.1. How to run scripts](#51-how-to-run-scripts)
  - [5.2. `stats.json` and other results](#52-statsjson-and-other-results)
  - [5.3. Conclusion](#53-conclusion)
- [6. How to cite](#6-how-to-cite)

## 1. The main results
The main table, where all models are sorted by their final (test) performance on all datasets,
can be found in the last cell of [this notebook](bin/report.ipynb).

## 2. Overview
The code is organized as follows:
- `bin`:
  - training code for all the models
  - `ensemble.py` performs ensembling
  - `tune.py` tunes models
  - `report.ipynb` summarizes all the results
  - `create_tables.ipynb` builds Table 1 and Table 2 from the paper
  - code for the section "When FT-Transformer is better than ResNet?" of the paper:
    - `analysis_gbdt_vs_nn.py` runs the experiments
    - `create_synthetic_data_plots.py` builds plots
- `lib` contains common tools used by programs in `bin`
- `output` contains configuration files (inputs for programs in `bin`) and results (metrics, tuned configurations, etc.)
- the remaining files and directories are mostly related to the `rtdl` package

The results are represented with numerous JSON files that are scatterd all over
the `output` directory. The notebook `bin/report.ipynb` summarizes them into a report
(a _much_ more detailed version of Table 1 and Table 2 from the paper). Use this notebook to:
- build a detailed report where all the models are compared on all datasets
- get insights about test, validation and train scores

## 3. Setup the environment

### 3.1. PyTorch environment
Install [conda](https://docs.conda.io/en/latest/miniconda.html)

```bash
export REPO_DIR=<ABSOLUTE path to the desired repository directory>
git clone <repository url> $REPO_DIR
cd $REPO_DIR

conda create -n rtdl python=3.8.8
conda activate rtdl

conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1.243 numpy=1.19.2 -c pytorch -y
conda install cudnn=7.6.5 -c anaconda -y
pip install -r requirements.txt
conda install -c conda-forge nodejs -y
jupyter labextension install @jupyter-widgets/jupyterlab-manager

# if the following commands do not succeed, update conda
conda env config vars set PYTHONPATH=${PYTHONPATH}:${REPO_DIR}
conda env config vars set PROJECT_DIR=${REPO_DIR}
conda env config vars set LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
conda env config vars set CUDA_HOME=${CONDA_PREFIX}
conda env config vars set CUDA_ROOT=${CONDA_PREFIX}

conda deactivate
conda activate rtdl
```

### 3.2. TensorFlow environment
_This environment is needed only for experimenting with TabNet. For all other cases use the PyTorch environment._

The instructions are the same as for the PyTorch environment (including installation of PyTorch!), but:
- `python=3.7.10`
- `cudatoolkit=10.0`
- _right before_ `pip install -r requirements.txt` do the following:
  - `pip install tensorflow-gpu==1.14`
  - comment out `tensorboard` in `requirements.txt`

### 3.3. Data
**LICENSE**: _by downloading our dataset you accept licenses of all its components. We
do not impose any new restrictions in addition to those licenses. You can find the list
of sources in the section "References" of our paper._
1. Download the data: `wget https://www.dropbox.com/s/o53umyg6mn3zhxy/rtdl_data.tar.gz?dl=1 -O rtdl_data.tar.gz`
2. Move the archive to the root of the repository: `mv rtdl_data.tar.gz $PROJECT_DIR`
3. Go to the root of the repository: `cd $PROJECT_DIR`
4. Unpack the archive: `tar -xvf rtdl_data.tar.gz`

## 4. Tutorial (how to reproduce results)
*This section only provides specific commands with few comments. After completing the tutorial, we recommend checking the next section for better understanding of how to work with the repository. It will also help to better understand the tutorial.*

In this tutorial, we will reproduce the results for MLP on the California Housing dataset. We will cover:
- tuning
- evaluation
- ensembling
- comparing models with each other

Note that the chances to get **exactly** the same results are rather low, however, they should not differ much from ours. Before running anything, go to the root of the repository and explicitly set `CUDA_VISIBLE_DEVICES` (if you plan to use GPU):
```bash
cd $PROJECT_DIR
export CUDA_VISIBLE_DEVICES=0
```

### 4.1. Check the environment
Before we start, let's check that the environment is configured successfully. The following
commands should train one MLP on the California Housing dataset:
```bash
mkdir draft
cp output/california_housing/mlp/tuned/0.toml draft/check_environment.toml
python bin/mlp.py draft/check_environment.toml
```
The result should be in the directory `draft/check_environment`. For now, the content of the result is not important.

### 4.2. Tuning
Our config for tuning MLP on the California Housing dataset is located at `output/california_housing/mlp/tuning/0.toml`.
In order to reproduce the tuning, copy our config and run your tuning:
```bash
# you can choose any other name instead of "reproduced.toml"; it is better to keep this
# name while completing the tutorial
cp output/california_housing/mlp/tuning/0.toml output/california_housing/mlp/tuning/reproduced.toml
# let's reduce the number of tuning iterations to make tuning fast (and ineffective)
python -c "
from pathlib import Path
p = Path('output/california_housing/mlp/tuning/reproduced.toml')
p.write_text(p.read_text().replace('n_trials = 100', 'n_trials = 5'))
"
python bin/tune.py output/california_housing/mlp/tuning/reproduced.toml
```
The result of your tuning will be located at `output/california_housing/mlp/tuning/reproduced`, you can compare it with ours: `output/california_housing/mlp/tuning/0`. The file `best.toml` contains the best configuration that we will evaluate in the next section.

### 4.3. Evaluation
Now we have to evaluate the tuned configuration with 15 different random seeds.

```bash
# create a directory for evaluation
mkdir -p output/california_housing/mlp/tuned_reproduced

# clone the best config from the tuning stage with 15 different random seeds
python -c "
for seed in range(15):
    open(f'output/california_housing/mlp/tuned_reproduced/{seed}.toml', 'w').write(
        open('output/california_housing/mlp/tuning/reproduced/best.toml').read().replace('seed = 0', f'seed = {seed}')
    )
"

# train MLP with all 15 configs
for seed in {0..14}
do
    python bin/mlp.py output/california_housing/mlp/tuned_reproduced/${seed}.toml
done
```

Our directory with evaluation results is located right next to yours, namely, at `output/california_housing/mlp/tuned`.

### 4.4. Ensembling
```bash
# just run this single command
python bin/ensemble.py mlp output/california_housing/mlp/tuned_reproduced
```
Your results will be located at `output/california_housing/mlp/tuned_reproduced_ensemble`, you can compare it with ours: `output/california_housing/mlp/tuned_ensemble`.

### 4.5. "Visualize" results
Use `bin/report.ipynb`:
- find the cell with the list of results; the list includes many lines of this kind:
  `('algorithm/experiment', N_SEEDS, ENSEMBLES_3_5, 'PrettyAlgorithmName', datasets)`
- uncomment the line relevant to the tutorial; it should look like this:
  `('mlp/tuned_reproduced', N_SEEDS, ENSEMBLES_3_5, 'MLP | reproduced', [CALIFORNIA]),`
- run the notebook from scratch

### 4.6. What about other models and datasets?
Similar steps can be performed for all models and datasets. The tuning process is
slightly different in the case of grid search: you have to run all desired
configurations and manually choose the best one **based on the validation performance**.
For example, see `output/epsilon/ft_transformer`.

## 5. How to work with the repository

### 5.1. How to run scripts
You should run Python scripts from the root of the repository. Most programs expect a
configuration file as the only argument. The output will be a directory with the same
name as the config, but without the extention. Configs are written in
[TOML](https://toml.io). The lists of possible arguments for the programs are not
provided and should be inferred from scripts (usually, the config is represented with
the `args` variable in scripts). If you want to use CUDA, you must explicitly set the
`CUDA_VISIBLE_DEVICES` environment variable. For example:
```bash
# The result will be at "path/to/my_experiment"
CUDA_VISIBLE_DEVICES=0 python bin/mlp.py path/to/my_experiment.toml

# The following example will run WITHOUT CUDA
python bin/mlp.py path/to/my_experiment.toml
```
If you are going to use CUDA all the time, you can save the environment variable in the
Conda environment:
```bash
conda env config vars set CUDA_VISIBLE_DEVICES="0"
```
The `-f` (`--force`) option will remove the existing results and run the script from scratch:
```bash
python bin/whatever.py path/to/config.toml -f  # rewrites path/to/config
```
`bin/tune.py` supports continuation:
```bash
python bin/tune.py path/to/config.toml --continue
```

### 5.2. `stats.json` and other results
For all scripts, `stats.json` is the most important part of output. The content varies
from program to program. It can contain:
- metrics
- config that was passed to the program
- hardware info
- execution time
- and other information

Predictions for train, validation and test sets are usually also saved.

### 5.3. Conclusion
Now, you know everything you need to reproduce all the results and extend
this repository for your needs. The [tutorial](#4-tutorial-how-to-reproduce-results) also
should be more clear now. Feel free to open issues and ask questions.

## 6. How to cite
```
@article{gorishniy2021revisiting,
    title={Revisiting Deep Learning Models for Tabular Data},
    author={Yury Gorishniy and Ivan Rubachev and Valentin Khrulkov and Artem Babenko},
    journal={arXiv},
    volume={2106.11959},
    year={2021},
}
```
