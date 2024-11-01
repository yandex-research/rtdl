# RTDL (Research on Tabular Deep Learning)

RTDL (**R**esearch on **T**abular **D**eep **L**earning) is a collection of papers and packages on deep learning for tabular data.

:bell: *To follow announcements on new papers and projects, subscribe to releases in this GitHub repository: "Watch -> Custom -> Releases".*

> [!NOTE]
> The previous `rtdl` package is now replaced
> with individual packages (see the next sections).
> If you used <code>rtdl</code>, please, read the details.
>
> <details>
> <summary>Show details</summary>
>
> 1. This repository is **NOT** deprecated.
> 2. However, the package `rtdl`
>    is deprecated and replaced with individual packages.
> 3. If you used the latest `rtdl==0.0.13` installed from PyPI (not from GitHub!)
>    as `pip install rtdl`, then the same models
>    (MLP, ResNet, FT-Transformer) can be found in the `rtdl_revisiting_models` package,
>    though API is slightly different.
> 4. :exclamation: **If you used the unfinished code from the main branch, it is highly**
>    **recommended to switch to the new packages.** In particular,
>    the unfinished implementation of embeddings for continuous features
>    contained many unresolved issues (the new `rtdl_num_embeddings` package, in turn,
>    is more efficient and correct).
>
> </details>

# Installation

**The documentation is available through the "Package" links in the ["Papers"](#papers) section.**

The following snippet installs all available packages
including optional dependencies.

```
pip install rtdl_num_embeddings
pip install rtdl_revisiting_models

pip install "scikit-learn>=1.0,<2"
```

# Papers

(2024) TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling
<br> [Paper](https://arxiv.org/abs/2410.24210)
&nbsp; [Code](https://github.com/yandex-research/tabm)

(2024) TabReD: A Benchmark of Tabular Machine Learning in-the-Wild
<br> [Paper](https://arxiv.org/abs/2406.19380)
&nbsp; [Code](https://github.com/yandex-research/tabred)

(2023) TabR: Tabular Deep Learning Meets Nearest Neighbors
<br> [Paper](https://arxiv.org/abs/2307.14338)
&nbsp; [Code](https://github.com/yandex-research/tabular-dl-tabr)

(2022) TabDDPM: Modelling Tabular Data with Diffusion Models
<br> [Paper](https://arxiv.org/abs/2209.15421)
&nbsp; [Code](https://github.com/yandex-research/tab-ddpm)

(2022) Revisiting Pretraining Objectives for Tabular Deep Learning
<br> [Paper](https://arxiv.org/abs/2207.03208)
&nbsp; [Code](https://github.com/puhsu/tabular-dl-pretrain-objectives)

(2022) On Embeddings for Numerical Features in Tabular Deep Learning
<br> [Paper](https://arxiv.org/abs/2203.05556)
&nbsp; [Code](https://github.com/yandex-research/rtdl-num-embeddings)
&nbsp; [Package (rtdl_num_embeddings)](https://github.com/yandex-research/rtdl-num-embeddings/tree/main/package/README.md)

(2021) Revisiting Deep Learning Models for Tabular Data
<br> [Paper](https://arxiv.org/abs/2106.11959)
&nbsp; [Code](https://github.com/yandex-research/rtdl-revisiting-models)
&nbsp; [Package (rtdl_revisiting_models)](https://github.com/yandex-research/rtdl-revisiting-models/tree/main/package/README.md)

(2019) Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data
<br> [Paper](https://arxiv.org/abs/1909.06312)
&nbsp; [Code](https://github.com/Qwicen/node)
