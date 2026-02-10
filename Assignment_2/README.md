# Predicting Firm Fast Growth

**Authors:** Konstantinos Evagorou, Fazile Brahimi  
**Course:** Data Analysis 3, Central European University  

## Project Overview

This project predicts which Hungarian SMEs will achieve fast growth (>25% sales increase) using machine learning classification models. We analyze 21,723 firms from 2012 and predict their 2013 growth outcomes using three models: Logistic Regression, Random Forest, and Gradient Boosting.

**Key Finding:** Gradient Boosting achieves the best performance with 86.1% recall, successfully identifying most high-growth opportunities while minimizing costly false negatives.

## Research Question

Can we predict which small-to-medium enterprises (SMEs) will achieve fast growth in the next year based on observable firm characteristics?

## Dataset

- **Source:** [Bisnode Firms Panel Data (OSF)](https://osf.io/download/qsk8y/)
- **Panel Period:** 2010-2015
- **Base Year:** 2012 (predictors) → 2013 (outcome)
- **Sample:** 21,723 Hungarian SMEs with €1K-€10M revenue
- **Target:** Fast growth = sales increase > 25% (5,480 firms, 25.2%)

**Data Processing:**
- Started from `cs_bisnode_panel.csv`
- Filtered to active firms with positive sales in 2012
- Removed large corporations (>€10M) and micro firms (<€1K)
- Merged with 2013 sales data to define growth outcome

## Methodology

### Target Definition
**Fast Growth:** Sales increase > 25% from 2012 to 2013

**Justification (Corporate Finance):**
- 25% growth substantially exceeds typical economic expansion (3-5%)
- Indicates successful capital deployment and market opportunities
- Alternative measures considered: absolute growth (favors large firms), 2-year growth (reduces sample due to attrition)
- Threshold balances meaningful growth with sufficient positive cases for robust modeling

### Features (8 → 15 after encoding)
- **sales_mil_log:** Log-transformed sales (handles skewness)
- **age, age²:** Firm age and quadratic term (captures lifecycle)
- **new:** Binary indicator for firms ≤1 year old
- **foreign_management:** Majority (≥50%) foreign ownership
- **gender_m:** CEO gender (categorical)
- **m_region_loc:** Geographic region (categorical)
- **ind2_cat:** Industry category (NACE 2-digit, aggregated)

### Models
1. **Logistic Regression** (Baseline)
   - L2 regularization with 5-fold CV
   - Interpretable coefficients
   - AUC: 0.636 ± 0.008

2. **Random Forest** (Ensemble)
   - 100 trees, balanced class weights
   - Handles non-linearities
   - AUC: 0.580 ± 0.008

3. **Gradient Boosting** (Winner) ⭐
   - 100 trees, depth 3, learning rate 0.1
   - Sequential error correction
   - **AUC: 0.646 ± 0.010** (best)

### Evaluation
- **Cross-Validation:** 5-fold stratified CV for AUC
- **Loss Function:** Asymmetric costs (FP=1, FN=5) reflecting venture capital context
- **Holdout Test:** 80-20 split for final evaluation

## Key Results

### 1. Probability Prediction (Cross-Validation)
| Model | Mean AUC | Std Dev | Rank |
|-------|----------|---------|------|
| **Gradient Boosting** | **0.646** | ±0.010 | 🏆 1st |
| Logistic Regression | 0.636 | ±0.008 | 2nd |
| Random Forest | 0.580 | ±0.008 | 3rd |

**Winner:** Gradient Boosting (best discrimination and consistent across folds)

### 2. Classification with Loss Function
| Model | Avg Loss | Optimal Threshold | vs. Best |
|-------|----------|-------------------|----------|
| **Gradient Boosting** | **2,894.60** | 0.14 | **Best** |
| Logistic Regression | 2,922.20 | 0.16 | +1.0% |
| Random Forest | 3,262.60 | 0.10 | +12.7% |

**Loss Function:** Expected Loss = FP × 1 + FN × 5 (venture capital context)

### 3. Holdout Performance (Gradient Boosting, threshold=0.18)

**Confusion Matrix:**
```
               Predicted
              No    Yes
Actual No    911   2,060
       Yes   134     828
```

**Metrics:**
- **Precision:** 28.7% (of predicted growth, 28.7% actually grow)
- **Recall:** 86.1% (catch 86% of fast-growth firms) ⭐
- **Specificity:** 30.7%
- **F1-Score:** 43.0%
- **Expected Loss:** 2,730

**Key Insight:** High recall (86.1%) aligns with venture capital strategy—better to investigate many firms than miss growth opportunities (5:1 cost ratio).

### 4. Feature Importance
| Feature | Importance | Interpretation |
|---------|-----------|----------------|
| **Age²** | **37%** | Non-linear lifecycle effects (young/old firms behave differently) |
| **Sales (log)** | **30%** | Larger firms have more resources for expansion |
| **Age** | **16%** | Younger firms more dynamic |
| **Industry** | **13%** | Sector-specific growth patterns |
| Foreign Mgmt | 1.3% | Marginal effect |

**Pattern:** Firm lifecycle (age, age²) and scale (sales) drive 83% of predictive power.

### 5. Industry Analysis (Holdout Set)

| Industry | Firms | Fast Growth % | AUC | Expected Loss |
|----------|-------|---------------|-----|---------------|
| **Services** | 13,721 | 23.3% | **0.652** | 1,912 |
| **Manufacturing** | 5,752 | 27.1% | 0.576 | 853 |

**Insight:** Services sector has better predictability (AUC 0.652) despite lower growth rate, possibly due to more stable patterns.

## Business Implications

### For Venture Capitalists
- **Use Case:** Pre-screen investment pipeline
- **Strategy:** Gradient Boosting at threshold 0.18 captures 86% of growth opportunities
- **Trade-off:** Accept 70% false positive rate to avoid missing high-potential firms (5× cost of FN vs FP)
- **Focus:** Young firms (age < 5), mid-sized (€1-5M), in services sector

### Model Limitations
- **Moderate AUC (0.646):** Significant unexplained variance remains
- **Low Precision (28.7%):** Most predicted growth firms don't achieve 25%+ growth
- **Missing Variables:** Innovation metrics, management quality, market trends not captured
- **Temporal:** Trained on 2012→2013 outcomes; may not generalize to different economic conditions

## Files in Repository

```
├── firm-growth-prediction.ipynb    # Main analysis (28 cells)
├── summary_report.pdf               # Executive summary (≤5 pages)
├── technical_report.pdf             # Technical documentation with code
├── README.md                        # This file
├── .gitignore                       # Git exclusions
├── sales_by_growth.png              # Sales distribution (300 DPI)
├── age_distribution.png             # Age histogram (300 DPI)
├── growth_by_industry.png           # Industry analysis (300 DPI)
├── roc_curves.png                   # ROC curve comparison (300 DPI)
├── confusion_matrix.png             # Classification results (300 DPI)
└── feature_importance.png           # Top 15 features (300 DPI)
```

**Note:** `cs_bisnode_panel.csv` (97MB) not included in repository due to size.

## Installation & Requirements

### Python Environment
```bash
Python 3.9+
Jupyter Notebook
```

### Required Packages
```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
```

## Running the Analysis

1. **Download the dataset:**
   - Visit: https://osf.io/download/qsk8y/
   - Save as `cs_bisnode_panel.csv` in project directory

2. **Run the notebook:**
   ```bash
   jupyter notebook firm-growth-prediction.ipynb
   ```

3. **Execute all cells** (Runtime ~5 minutes):
   - Data loading and cleaning
   - Feature engineering
   - 5-fold cross-validation (3 models)
   - Threshold optimization with loss function
   - Holdout evaluation
   - Industry analysis
   - Visualization generation

## Repository

**GitHub:** [https://github.com/constantinosevag-rgb/Data-Analysis-3/tree/main/Assignment_2](https://github.com/constantinosevag-rgb/Data-Analysis-3/tree/main/Assignment_2)

## Authors

- **Konstantinos Evagorou** - constantinosevagorou@gmail.com
- **Fazile Brahimi**
---
