# RTDL (Research on Tabular Deep Learning)

RTDL (**R**esearch on **T**abular **D**eep **L**earning) is a collection of papers and packages on deep learning for tabular data.

:bell: *To follow announcements on new papers and projects,
subscribe to releases in the GitHub interface: "Watch -> Custom -> Releases".*

> [!NOTE]
> The `rtdl` *package* (but not this *repository*!) is deprecated.
> If you used it, please, read the details:
>
> <details>
> <summary>Show</summary>
>
> 1. Now, some of the papers have their individual packages (see the table below).
> 2. If you used the latest `rtdl==0.0.13` installed from PyPI (not from GitHub!)
>    as `pip install rtdl`, then the same models
>    (MLP, ResNet, FT-Transformer) can be found in the `rtdl_revisiting_models` package,
>    though API is slightly different.
> 3. :exclamation: **If you used the unfinished code from the main branch, it is highly**
>    **recommended to switch to the new packages.** In particular,
>    the unfinished implementation of embeddings for continuous features
>    contained many unresolved issues (the new `rtdl_num_embeddings` package, in turn,
>    is more efficient and correct).
>
> </details>
> 


# Papers & Packages

> [!IMPORTANT]
> 
> Python packages are located inside paper repositories.

| Name                                                                   | Year  | Paper                                     | Code                                                                        | Package                  |
| :--------------------------------------------------------------------- | :---: | :---------------------------------------- | :-------------------------------------------------------------------------- | :----------------------- |
| TabR: Unlocking the Power of Retrieval-Augmented Tabular Deep Learning | 2023  | [arXiv](https://arxiv.org/abs/2307.14338) | [GitHub](https://github.com/yandex-research/tabular-dl-tabr)                | -                        |
| TabDDPM: Modelling Tabular Data with Diffusion Models                  | 2022  | [arXiv](https://arxiv.org/abs/2209.15421) | [GitHub](https://github.com/yandex-research/tab-ddpm)                       | -                        |
| Revisiting Pretraining Objectives for Tabular Deep Learning            | 2022  | [arXiv](https://arxiv.org/abs/2207.03208) | [GitHub](https://github.com/yandex-research/tabular-dl-pretrain-objectives) | -                        |
| On Embeddings for Numerical Features in Tabular Deep Learning          | 2022  | [arXiv](https://arxiv.org/abs/2203.05556) | [GitHub](https://github.com/yandex-research/tabular-dl-num-embeddings)      | `rtdl_num_embeddings`    |
| Revisiting Deep Learning Models for Tabular Data                       | 2021  | [arXiv](https://arxiv.org/abs/2106.11959) | [GitHub](https://github.com/yandex-research/tabular-dl-revisiting-models)   | `rtdl_revisiting_models` |
| Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data  | 2019  | [arXiv](https://arxiv.org/abs/1909.06312) | [GitHub](https://github.com/Qwicen/node)                                    | -                        |
