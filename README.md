# Advanced Fraud Detection: Leveraging K-SMOTEENN and Stacking Ensemble to Tackle Data Imbalance and Extract Insights

This repository contains the implementation code for our paper published in **IEEE Access (2025)**.

> **Nurafni Damanik and Chuan-Ming Liu**, "Advanced Fraud Detection: Leveraging K-SMOTEENN and Stacking Ensemble to Tackle Data Imbalance and Extract Insights," *IEEE Access*, vol. 13, pp. 10356–10370, 2025.
> DOI: [10.1109/ACCESS.2025.3528079](https://doi.org/10.1109/ACCESS.2025.3528079)

---

## Overview

Credit card fraud detection is challenged by extreme class imbalance, where fraudulent transactions make up a tiny fraction of all records. This work proposes an innovative solution that combines:

- **K-SMOTEENN** — a three-phase hybrid resampling pipeline (K-Means clustering → SMOTE oversampling → Edited Nearest Neighbors cleaning) that balances the minority class while reducing synthetic-data noise.
- **Stacking Ensemble** — a hierarchical model combining diverse base learners (Decision Tree, Random Forest) with a Random Forest meta-learner for a more generalized decision boundary.
- **LIME** — applied at both base-learner and meta-learner levels for interpretable, auditable predictions.

## Key Results

Evaluated on the PaySim dataset, the proposed approach achieves:

| Metric | Score |
|--------|-------|
| F1-score | 0.92 |
| Precision | 0.95 |
| Recall | 0.88 |
| AUPRC | 0.96 |
| ROC-AUC | 1.00 |

These results significantly outperform standalone models such as XGBoost and Decision Tree.

## Repository Structure

```
.
├── README.md
├── fraud_detection_ksmoteenn.ipynb   # Main experiment notebook
└── dataset_link.txt                  # Where to obtain the dataset
```

## Dataset

This study uses the **PaySim synthetic financial dataset**, which is publicly available on Kaggle:
[https://www.kaggle.com/datasets/ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1)

The dataset is not included in this repository due to its size. Please download it from the link above and place it in the working directory before running the notebook.

## How to Run

1. Open the notebook `fraud_detection_ksmoteenn.ipynb` in [Google Colab](https://colab.research.google.com/) or Jupyter Notebook.
2. Download the PaySim dataset (see link above) and upload it to your environment.
3. Run the cells in order.

**Main libraries used:** `scikit-learn`, `imbalanced-learn`, `pandas`, `numpy`, `lime`, `matplotlib`.

## Method Summary

1. **Preprocessing** — data cleaning, exploratory analysis, and construction of the `type2` interaction feature.
2. **Resampling (K-SMOTEENN)** — cluster the minority class, oversample within clusters, then remove boundary noise with ENN.
3. **Classification (Stacking)** — train base learners on resampled data, then train a meta-learner on their predictions.
4. **Explainability (LIME)** — generate instance-level explanations at both ensemble levels.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{damanik2025advanced,
  title={Advanced Fraud Detection: Leveraging K-SMOTEENN and Stacking Ensemble to Tackle Data Imbalance and Extract Insights},
  author={Damanik, Nurafni and Liu, Chuan-Ming},
  journal={IEEE Access},
  volume={13},
  pages={10356--10370},
  year={2025},
  publisher={IEEE},
  doi={10.1109/ACCESS.2025.3528079}
}
```

## License

This paper is published in IEEE Access under a Creative Commons Attribution 4.0 (CC BY 4.0) license. The code in this repository is released for academic and research use.

## Contact

**Nurafni Damanik**
PhD Candidate, National Taipei University of Technology
Email: nurafnidamanik36@gmail.com
