# 🏦 Optimizing Premium Pricing: Predictive Analytics for Insurance Loss

> **DSO 530 — Applied Modern Statistical Learning Methods**
> University of Southern California, Marshall School of Business | May 2026

---

## 📌 Project Overview

Insurance companies face a critical challenge — setting the right premium for each policyholder. Charge too much and low-risk customers leave. Charge too little and high-risk customers stay, creating financial losses through **adverse selection**.

This project uses machine learning to accurately predict three key insurance metrics for 13,310 new policyholders, enabling data-driven premium pricing and a profitable portfolio.

---

## 🎯 What We Predicted

| Target | Definition | Type |
|---|---|---|
| **LC** | Loss Cost per Exposure Unit = X.15 / X.16 | Regression |
| **HALC** | Historically Adjusted Loss Cost = LC × X.18 | Regression |
| **CS** | Claim Status = 1 if policyholder made a claim, else 0 | Classification |

---

## 📂 Repository Structure

```
insurance-loss-analytics/
│
├── 📁 code/
│   └── insurance_analysis.ipynb     # Full data analysis & modeling notebook
│
└── 📁 report-and-presentation/
    ├── report.pdf          # Full project report (6-page main + appendix)
    └── poster.pptx         # Poster presentation slide
```

---

## 📊 Dataset

| | Training | Test |
|---|---|---|
| **Records** | 39,928 | 13,310 |
| **Variables** | 28 | 23 |
| **Claim Rate** | 11.23% | — |

> ⚠️ Data files are not included in this repository as they are proprietary course materials from USC Marshall.

---

## 🔧 Methods & Models

### Data Preparation
- Dropped 453 corrupt training rows (impossible values in horsepower, doors, fuel type)
- Imputed 140 missing fuel type values in test data using a **Random Forest classifier** based on vehicle characteristics — predicted 130 Petrol, 10 Diesel (vs naively filling all as Diesel which would have been 93% wrong)
- Engineered 5 new features: driver age, license years, policy duration, experience ratio, power-to-weight ratio, cancellation rate

### Task 1 — Predicting LC & HALC (Tweedie Regression)

| Model | LC Val MSE | HALC Val MSE |
|---|---|---|
| Tweedie GLM (Baseline) | 324,175 | 1,026,099 |
| XGBoost Tweedie | 317,711 | 1,009,443 |
| LightGBM Tweedie | 317,852 | 1,008,616 |
| Regularized LightGBM | 316,612 | 1,007,285 |
| **Stacking Ensemble ✅** | **316,465** | **1,005,405** |

> 💡 **Innovation:** Tuned Tweedie variance power parameter (LC: 1.3, HALC: 1.4) instead of assuming the standard 1.5 — improving accuracy through data-driven parameter selection.

### Task 2 — Predicting Claim Status (Classification)

| Model | Val AUC | vs Baseline |
|---|---|---|
| Logistic Regression (Baseline) | 0.6965 | — |
| Random Forest | 0.7521 | +7.98% |
| **XGBoost Classifier ✅** | **0.7884** | **+12.78%** |

> 💡 **Innovation:** Calibrated classification threshold to 0.71 (from default 0.5) achieving predicted claim rate of 11.01% vs actual 11.23% — only 0.22% difference.

---

## 🔍 Key Innovations Beyond Requirements

1. **Optimal Tweedie Power Tuning** — Systematically tuned power from 1.1 to 1.9 instead of assuming 1.5
2. **Stacking Ensemble** — Combined LightGBM + XGBoost + Random Forest with Ridge meta-model
3. **SHAP Feature Attribution** — Explained model predictions using SHapley Additive exPlanations
4. **KMeans Risk Segmentation** — Identified 4 distinct policyholder risk profiles using unsupervised learning
5. **Loss Ratio Analysis** — Assessed premium adequacy using industry-standard LC/Premium and HALC/Premium ratios
6. **Threshold Calibration** — Fine-tuned CS classification threshold for real-world deployment

---

## 💡 Key Findings

- **Policy Duration and Cancellation Rate dominate all 3 targets** across SHAP analysis — behavioral features outperform vehicle characteristics, challenging conventional insurance pricing wisdom
- Short-term policies (0-1 year) have a **22.4% claim rate** vs only **3.3%** for 10+ year policies
- Agricultural vehicles showed **0% claim rate** across all 237 policies — suggesting a need for a completely separate pricing model
- HALC loss ratios in the $0-200 and $400-600 premium bands approach **0.45** — indicating potential underpricing in these segments

---

## 👥 Team

| Name |
|---|
| Aditi Attavar |
| Richa Sudin Desai |
| Trevor Shepherd |
| Islam Sultan |
| Jiaying Wang |
| Yuqi Zhang |

---

## 🛠️ Tech Stack

```
Python          pandas          numpy
scikit-learn    XGBoost         LightGBM
SHAP            matplotlib      seaborn
Google Colab
```

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/insurance-loss-analytics.git

# Install required packages
pip install lightgbm shap xgboost scikit-learn pandas numpy matplotlib seaborn
```

---

## 📄 License

This project was completed as part of DSO 530 at USC Marshall School of Business. Data is proprietary and not included. Code and analysis are for educational purposes.

---

*USC Marshall School of Business | DSO 530 | May 2026*
