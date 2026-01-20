# TimeMM: Time-as-Operator Spectral Filtering for Dynamic Multimodal Recommendation

**TimeMM** is a time-conditioned spectral filtering framework for **dynamic multimodal recommendation**.

- *TimeMM: Time-as-Operator Spectral Filtering for Dynamic Multimodal Recommendation*

---

## Overview

Multimodal recommendation improves user modeling by integrating collaborative signals with heterogeneous item content. In real applications, user interests evolve over time and exhibit non-stationary dynamics, where different preference factors change at different rates. This challenge is amplified in multimodal settings because visual and textual cues can dominate decisions under different temporal regimes. Despite strong progress, most multimodal recommenders still rely on static interaction graphs or coarse temporal heuristics, which limits their ability to model continuous preference evolution with fine-grained temporal adaptation.

To address these limitations, we propose **TimeMM**, a **time-conditioned spectral filtering** framework for dynamic multimodal recommendation. TimeMM instantiates **Time-as-Operator** by mapping interaction recency to a family of parametric temporal kernels that reweight edges on the user–item graph, producing component-specific representations **without explicit eigendecomposition**. To capture non-stationary interests, we introduce **Adaptive Spectral Filtering** that mixes the operator bank according to temporal context, yielding prediction-specific effective spectral responses. To account for modality-specific temporal sensitivity, we further propose **Spectral-Aware Modality Routing** that calibrates visual and textual contributions conditioned on the same temporal context. Finally, a ranking-space **Spectral Diversity Regularization** encourages complementary expert behaviors and prevents filter-bank collapse. Extensive experiments on real-world benchmarks demonstrate that TimeMM consistently outperforms state-of-the-art multimodal recommenders while maintaining **linear-time scalability**.

---

## Installation

We provide two options to set up the conda environment.

### Option A: Install from `environment.yaml`

```bash
conda env create -f environment.yaml
conda activate mmrec
```

### Option B: Install from `environment.txt` (explicit spec)

```bash
conda create -n mmrec --file environment.txt
conda activate mmrec
```

If your environment name is different, replace `mmrec` with your own env name.

---

## Data

Our datasets follow the same format as MMRec.  
Please download the datasets and place them under the `data/` directory, then you can run experiments directly.

```text
TimeMM/
├── data/
│   ├── dataset_name_1/
│   ├── dataset_name_2/
│   └── dataset_name_3/
├── src/
└── run_exp.sh
```

---

## Copyright Notice

Due to data copyright restrictions, we cannot redistribute third-party datasets. Please download the raw Amazon data directly from the official Amazon source. We provide a Video Games subset as a demo dataset for running examples and quick sanity checks.

---

## Running

Run the following script to start training and evaluation:

```bash
sh run_games.sh
```

---

## Notes and Suggestions

- **Timestamp quality matters.** Please ensure your dataset contains high-quality timestamps. The cleaner and more reliable the timestamps, the more significant the gains from TimeMM. In our industrial dataset with high-quality timestamps, the improvements are particularly strong. For datasets with noisy, irregular, or extremely sparse timestamps, the benefit may be limited.
- **Timestamp scale.** Our example uses second-level timestamps. If your dataset spans a very long horizon, consider rescaling timestamps or filtering samples with excessively large spans based on your task needs.
- **Dataset-specific tuning.** Timestamp distributions and span statistics differ substantially across datasets. Please tune hyperparameters per dataset. The default configuration is a reasonable average-case setting, but it is not guaranteed to be optimal everywhere.
- **Multi-granularity timestamps in practice.** In real online settings, we also observed strong gains when timestamps mix multiple granularities (seconds/days/weeks/months). TimeMM remains effective under such multi-scale temporal signals.
- **Reproducibility.** In principle, the absolute reproduction gap should not exceed 0.2%–0.3%. If performance varies widely across datasets, please first check timestamp quality and dataset integrity.

If you have any questions, please try to contact us in authorship order first.
