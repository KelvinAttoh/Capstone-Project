# 🏥 Capstone Project
### Frailty Prediction, Social Participation And Malnutrition Analysis In Older Adults Americans Using ML And Statistical Modeling

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20RF%20%7C%20LR%20%7C%20SVM-orange?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Background](#background)
- [Research Questions](#research-questions)
- [Dataset](#dataset)
- [Methods](#methods)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

---

## Overview

This capstone project presents a comprehensive analytical framework for
understanding **frailty** and **social participation** among community-dwelling
older adults. Using a dataset of **n = 7,086 participants**, the project combines
rigorous statistical modelling with machine learning to:

- Predict frailty phenotype (Frail / Prefrail / Robust) using clinical and
  demographic predictors
- Investigate whether psychosocial, clinical, and environmental factors
  moderate the relationship between social participation and frailty
- Model social participation as both a binary outcome and a continuous score
  stratified by residence type (Rural vs Urban)

The project addresses a critical gap in gerontological research by examining
how **rurality**, **comorbidity**, **depression**, and **malnutrition** interact
with social participation to influence frailty outcomes in older adults.

---

## Background

Frailty is a clinical syndrome characterised by decreased physiological reserve
and increased vulnerability to stressors, affecting approximately **10–15% of
community-dwelling older adults** worldwide. It is associated with adverse
outcomes including hospitalisation, disability, and mortality.

Social participation — defined as engagement in activities that provide
interaction with others — has been identified as a protective factor against
frailty progression. However, the moderating role of clinical and environmental
factors on this relationship remains poorly understood, particularly in the
context of rural-urban disparities.

This project applies both **inferential statistics** and **supervised machine
learning** to explore these relationships systematically, contributing to
evidence-based strategies for frailty prevention and intervention.

---

## Research Questions

| RQ | Question | Key Moderator | Statistical Approach |
|----|----------|---------------|---------------------|
| **RQ1** | Does residence type moderate the relationship between social participation and frailty? | Residence Type (Rural=0 / Urban=1) | Multinomial logistic regression + interaction term |
| **RQ2** | Does comorbidity level moderate the relationship between social participation and frailty? | Comorbidity Positive (Low=0 / High=1) | Multinomial logistic regression + interaction term |
| **RQ3** | Does depression screen status moderate the relationship between social participation and frailty? | Depression Screen Positive (PHQ-2) | Multinomial logistic regression + interaction term |
| **RQ4** | Does rurality moderate the relationship between malnutrition and frailty? | Residence Type (Rural=0 / Urban=1) | Multinomial logistic regression + interaction term |

All RQ analyses follow a two-model structure:
- **Model 1:** Main effects only
- **Model 2:** Main effects + interaction term (moderation test)
- **Model comparison:** Likelihood-ratio test (LRT), ΔAIC, ΔBIC

---

## Dataset

> ⚠️ **Data Privacy Notice:** The datasets used in this project are not
> publicly available in this repository due to data governance and privacy
> considerations. See [`data/README.md`](data/README.md) for access information.

### Overview

| File | n | Variables | Primary Target |
|------|---|-----------|----------------|
| `Frailty dataset.csv` | 7,086 | 99 | Frailty Status (Frail / Prefrail / Robust) |
| `SP dataset.csv` | 7,086 | 99 | Socially Active (0/1) · SP Score (0–5) |

### Frailty Phenotype Distribution

| Category | n | % |
|----------|---|---|
| Robust | 2,482 | 35.0% |
| Prefrail | 3,372 | 47.6% |
| Frail | 1,232 | 17.4% |

### Social Participation Distribution

| Category | n | % |
|----------|---|---|
| Socially Active (1) | 5,609 | 79.2% |
| Not Active (0) | 1,477 | 20.8% |
| Rural Residents | 1,185 | 16.7% |
| Urban Residents | 5,901 | 83.3% |

### Key Variables

| Variable | Type | Description |
|----------|------|-------------|
| `Frailty Status` | Categorical | Frail / Prefrail / Robust (Fried criteria) |
| `Frailty Phenotype` | Ordinal | 1=Frail, 2=Prefrail, 3=Robust |
| `Social Participation Score` | Continuous | 0–5 ordinal composite score |
| `Socially Active` | Binary | 0=Not Active, 1=Active |
| `Residence Type` | Binary | 0=Rural, 1=Urban |
| `Age Group` | Binary | 0=Under 75, 1=75 and above |
| `Gender` | Binary | 0=Female, 1=Male |
| `BMI Score` | Binary | 0=Normal/Underweight, 1=Overweight/Obese |
| `Education Group` | Binary | 0=No High School, 1=High School and Above |
| `Comorbidity Positive` | Binary | 0=Low (<4 diseases), 1=High (≥4 diseases) |
| `Race/Ethnicity` | Categorical | White, Black, Hispanic, Other, Mixed |
| `Depression Screen Positive` | Binary | PHQ-2 ≥ 3 = Positive |
| `Anxiety Screen Positive` | Binary | GAD-2 ≥ 3 = Positive |
| `Maximum Gait Speed` | Continuous | m/s (−9 coded as missing) |
| `Malnutrition Positive` | Binary | 0=Not Malnourished, 1=Malnourished |

---

## Methods

### 1. Data Preprocessing

- Column name stripping and type coercion
- Missing value encoding: `−9` → `NaN` for gait speed and frailty components
- Derived variables:
  - `Age Group`: Young-old (< 75) = 0, Middle/Oldest-old (≥ 75) = 1
  - `Education Group`: No High School = 0, High School and Above = 1
- Train / Validation / Test split: **60% / 20% / 20%** (stratified)
- Preprocessing pipeline per split (fit on train only):
  - `OrdinalEncoder` for ordinal categories
  - `OneHotEncoder` for nominal categories (Race/Ethnicity, Gender)
  - `StandardScaler` + median imputation for numeric features

### 2. Statistical Analysis (RQ1–RQ4)

Each RQ follows an identical analytical structure:
```
Step 1 — Descriptive statistics (Mean, SD, Median, IQR)
Step 2 — Normality testing (Shapiro-Wilk, n ≤ 50 subsample)
Step 3 — Primary inference test (Mann-Whitney U or Kruskal-Wallis)
Step 4 — Effect size (rank-biserial r, Hedges' g, Spearman ρ)
Step 5 — Model 1: Main effects (SP + moderator)
Step 6 — Model 2: Moderation model (SP + moderator + SP × moderator)
Step 7 — Likelihood-ratio test: Model 1 vs Model 2
Step 8 — Bootstrap 95% CIs (500 iterations)
Step 9 — Figure 2: 4-panel visualization
```

**Statistical tests used:**

| Test | Purpose |
|------|---------|
| Mann-Whitney U | Primary group comparison (non-parametric) |
| Welch t-test | Secondary parametric comparison |
| Kruskal-Wallis | Multi-group comparison |
| Spearman ρ | Continuous variable correlation |
| Shapiro-Wilk | Normality testing (subsample n ≤ 50) |
| Likelihood-ratio test | Model comparison (M1 vs M2) |
| Bootstrap CI | 95% confidence intervals (500 iterations) |

**Regression output per model:**

| Output | Description |
|--------|-------------|
| β | Log-odds coefficient |
| RRR | Relative Risk Ratio (exp(β)) |
| 95% CI | Bootstrap confidence interval |
| z-statistic | β / SE |
| p-value | Two-tailed from normal distribution |
| McFadden R² | Pseudo R-squared |
| AIC / BIC | Model fit criteria |
| ΔAIC / ΔBIC | Change from M1 to M2 |

### 3. Machine Learning Pipeline (Frailty Prediction)

**Target:** Frailty Status → Frail=0, Prefrail=1, Robust=2

**Leakage prevention:** All frailty-defining criteria, aggregate scores, and
direct frailty derivatives removed before modelling (18 columns dropped).

**Models trained:**

| Model | Key Hyperparameters |
|-------|-------------------|
| Logistic Regression | L1 penalty, liblinear solver, class_weight=balanced |
| Random Forest | 300 trees, max_depth=50, class_weight=balanced |
| XGBoost | multi:softmax, max_depth=4, lr=0.1, 300 estimators |
| SVM | RBF kernel, class_weight=balanced, probability=True |

**Class imbalance handling:**
- `RandomOverSampler` applied to training set only
- Validation and test sets retain original distribution

**Evaluation metrics:** Accuracy, F1 weighted, F1 macro, Precision, Recall, ROC-AUC

**Explainability:** SHAP values computed for best model (XGBoost) using
`TreeExplainer` for feature importance and `waterfall` plots per prediction class.

---

## Results

### RQ Summary

| RQ | Primary Finding | LRT | Moderation |
|----|----------------|-----|-----------|
| RQ1 | SP significantly associated with frailty across residence groups | χ²(df) | Tested |
| RQ2 | High comorbidity associated with reduced SP and increased frailty | χ²(df) | Tested |
| RQ3 | Depression screen positive significantly associated with lower SP | χ²(df) | Tested |
| RQ4 | Rural malnutrition associated with higher frailty burden | χ²(df) | Tested |

### Machine Learning — Frailty Prediction

| Model | Accuracy | F1 Weighted | 
|-------|----------|-------------|
| **XGBoost** ⭐ | **85.0%** | **0.85** |
| Random Forest | 81.0% | 0.81 |
| Logistic Regression | 81.0% | 0.81 | 
| SVM | 81.0% | 0.80 |

> ⭐ XGBoost selected as best model based on weighted F1-score.


---

## Acknowledgements

This project was completed as part of a capstone research programme.
Special thanks to the data custodians and research supervisors who
supported this work.

---

<div align="center">
  <sub>Built with Python · Analysed with scikit-learn & XGBoost ·
  Visualised with matplotlib & seaborn</sub>
</div>
