# Vox Finance — Credit Risk Modelling

> An end-to-end machine learning system for quantifying borrower default risk, generating credit scores, and delivering real-time risk ratings through an interactive web application.

**Live App:** [https://vox-credit-risk-modelling.streamlit.app/](https://vox-credit-risk-modelling.streamlit.app/)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Business Context](#business-context)
3. [Dataset Architecture](#dataset-architecture)
4. [Project Structure](#project-structure)
5. [Machine Learning Pipeline](#machine-learning-pipeline)
   - [Data Ingestion & Merging](#1-data-ingestion--merging)
   - [Data Cleaning](#2-data-cleaning)
   - [Exploratory Data Analysis](#3-exploratory-data-analysis)
   - [Feature Engineering](#4-feature-engineering)
   - [Feature Selection](#5-feature-selection)
   - [Model Training & Attempts](#6-model-training--attempts)
   - [Model Evaluation](#7-model-evaluation)
   - [Credit Score Calculation](#8-credit-score-calculation)
6. [Model Performance Summary](#model-performance-summary)
7. [Web Application](#web-application)
8. [Tech Stack](#tech-stack)
9. [Inference Flow](#inference-flow)

---

## Project Overview

Vox Finance's Credit Risk Modelling system predicts the **probability of loan default** for individual borrowers and translates that probability into a human-readable **credit score (300–900)** and a **risk rating** (Poor / Average / Good / Excellent). The system is built on a Logistic Regression model trained with SMOTE-Tomek resampling and hyperparameter-tuned via Optuna, then deployed as a Streamlit web application.

---

## Business Context

Lending institutions lose substantial capital to loan defaults every year. Accurately predicting default risk at the time of loan application allows a lender to:

- **Price risk appropriately** — adjust interest rates based on borrower risk profile
- **Set credit limits** — restrict exposure to high-risk borrowers
- **Automate decisions** — reduce manual credit review overhead
- **Maintain portfolio health** — monitor aggregate risk across the loan book

The model directly outputs three actionable signals:

| Output | Description | Range |
|---|---|---|
| Default Probability | Likelihood the borrower will default | 0% – 100% |
| Credit Score | Calibrated risk score derived from log-odds | 300 – 900 |
| Risk Rating | Human-readable creditworthiness label | Poor / Average / Good / Excellent |

---

## Dataset Architecture

The system uses **three raw CSV datasets**, each with **50,000 records**, merged on `cust_id`.

### customers.csv — Applicant Demographics

| Column | Type | Description |
|---|---|---|
| `cust_id` | string | Unique customer identifier |
| `age` | int | Applicant age in years |
| `gender` | string | Gender of the applicant |
| `marital_status` | string | Marital status |
| `employment_status` | string | Employment type (Salaried / Self-employed / etc.) |
| `income` | float | Annual income in INR |
| `number_of_dependants` | int | Number of financial dependants |
| `residence_type` | string | Owned / Rented / Mortgage |
| `years_at_current_address` | float | Residential stability indicator |
| `city` | string | City of residence |
| `state` | string | State of residence |
| `zipcode` | string | Postal code |

**Shape:** 50,000 rows × 12 columns

---

### loans.csv — Loan Transaction Details

| Column | Type | Description |
|---|---|---|
| `loan_id` | string | Unique loan identifier |
| `cust_id` | string | Foreign key to customers |
| `loan_purpose` | string | Education / Home / Auto / Personal |
| `loan_type` | string | Secured / Unsecured |
| `sanction_amount` | float | Amount sanctioned by the lender |
| `loan_amount` | float | Actual disbursed amount |
| `processing_fee` | float | Fee charged on loan processing |
| `gst` | float | GST applied on processing fee |
| `net_disbursement` | float | Amount received by the borrower |
| `loan_tenure_months` | int | Loan repayment period in months |
| `principal_outstanding` | float | Remaining principal balance |
| `bank_balance_at_application` | float | Applicant's bank balance at time of application |
| `disbursal_date` | date | Date of loan disbursement |
| `installment_start_dt` | date | Date repayments commenced |
| `default` | bool | **Target variable** — whether the borrower defaulted |

**Shape:** 50,000 rows × 15 columns  
**Class Distribution:**

| Class | Count | Percentage |
|---|---|---|
| Non-Default (0) | 45,703 | ~91.4% |
| Default (1) | 4,297 | ~8.6% |

> The dataset has a significant **class imbalance** (~10:1 ratio), which was addressed during model training.

---

### bureau_data.csv — Credit Bureau History

| Column | Type | Description |
|---|---|---|
| `cust_id` | string | Foreign key to customers |
| `number_of_open_accounts` | int | Active open loan accounts |
| `number_of_closed_accounts` | int | Historical closed loan accounts |
| `total_loan_months` | int | Total months across all loans |
| `delinquent_months` | int | Months in which payments were missed |
| `total_dpd` | int | Total Days Past Due across all loans |
| `enquiry_count` | int | Number of credit enquiries |
| `credit_utilization_ratio` | float | % of available credit currently used |

**Shape:** 50,000 rows × 8 columns

---

## Project Structure

```
1.ML Project Credit Risk Modelling/
│
├── dataset/
│   ├── customers.csv          # Applicant demographic data (50K rows)
│   ├── loans.csv              # Loan transaction & target data (50K rows)
│   └── bureau_data.csv        # Credit bureau history (50K rows)
│
├── models/
│   └── model_data.joblib      # Serialized model bundle:
│                              #   - Trained LogisticRegression model
│                              #   - Fitted MinMaxScaler
│                              #   - Selected feature list
│                              #   - Columns to scale
│
├── notebooks/
│   └── Credit Risk Model.ipynb  # Full ML pipeline: EDA → Training → Evaluation
│
├── app/
│   ├── main.py                # Streamlit application UI
│   └── prediction_helper.py   # Inference logic & credit score computation
│
└── README.md
```

---

## Machine Learning Pipeline

### 1. Data Ingestion & Merging

All three datasets are loaded and merged into a single analytical dataframe:

```
customers.csv  ──┐
                  ├── merge on cust_id ──> df  ──┐
loans.csv      ──┘                               ├──> merge on cust_id ──> df1 (50K × 34)
bureau_data.csv ─────────────────────────────────┘
```

The merged dataset contains **34 columns** across demographics, loan attributes, bureau history, and the target variable `default`.

---

### 2. Data Cleaning

| Issue | Resolution |
|---|---|
| Null values in `residence_type` | Imputed with training set **mode** |
| Duplicate records | Verified none present in training set |
| Outlier `processing_fee` values | Rows where `processing_fee / loan_amount > 3%` were removed as erroneous |
| Misspelled category `"Personaal"` in `loan_purpose` | Corrected to `"Personal"` in both train and test sets |

A **train/test split of 75/25** was applied with `stratify=y` to preserve class proportions.

---

### 3. Exploratory Data Analysis

Continuous distributions were analyzed using KDE plots segmented by `default=0` and `default=1`:

**Key EDA Insights:**

| Feature | Observation |
|---|---|
| `age` | Younger applicants skew towards higher default rates — the default=1 KDE is shifted left |
| `loan_tenure_months` | Longer tenures correlate with higher default probability |
| `delinquent_months` | Strong separation — defaulters have significantly more delinquent months |
| `total_dpd` | High DPD values are strongly associated with defaults |
| `credit_utilization_ratio` | Higher utilization ratio associated with higher default risk |
| `loan_amount` / `income` | Individually weak predictors; combined as Loan-to-Income ratio they become meaningful |

Correlation and distribution analyses confirmed that raw `loan_amount` and `income` are individually poor predictors but their **ratio** is highly informative.

---

### 4. Feature Engineering

Three derived features were constructed to capture non-linear risk signals:

| Engineered Feature | Formula | Rationale |
|---|---|---|
| `loan_to_income` | `loan_amount / income` | Measures borrower leverage — higher ratio = higher stress |
| `delinquency_ratio` | `(delinquent_months / total_loan_months) × 100` | % of loan life spent in missed payments — standardises across loan lengths |
| `avg_dpd_per_delinquency` | `total_dpd / delinquent_months` | Average severity of each delinquency event, 0 for non-delinquent |

KDE plots confirmed all three engineered features show **clear separation** between default and non-default classes.

---

### 5. Feature Selection

Two rigorous statistical methods were used to select the final feature set:

#### Variance Inflation Factor (VIF) — Multicollinearity Removal

VIF analysis identified highly collinear features. The following were removed:

```
sanction_amount, processing_fee, gst, net_disbursement, principal_outstanding
```

These variables carry redundant information relative to `loan_amount` and `net_disbursement`.

#### Information Value (IV) — Predictive Power Ranking

IV quantifies how well each feature discriminates between defaulters and non-defaulters:

| IV Range | Predictive Power |
|---|---|
| < 0.02 | Unpredictive |
| 0.02 – 0.1 | Weak |
| 0.1 – 0.3 | Medium |
| > 0.3 | Strong |

Features with `IV > 0.02` were retained. This threshold eliminated low-signal variables and produced a lean, interpretable feature set including:

```
age, loan_tenure_months, number_of_open_accounts, credit_utilization_ratio,
loan_to_income, delinquency_ratio, avg_dpd_per_delinquency,
residence_type, loan_purpose, loan_type
```

Categorical features were one-hot encoded with `drop_first=True` to avoid the dummy variable trap.

---

### 6. Model Training & Attempts

Three training strategies were explored, each progressively improving handling of class imbalance:

#### Attempt 1 — Baseline (No Imbalance Handling)

Three algorithms were trained on the raw imbalanced data:

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear model |
| Random Forest | Ensemble tree-based model |
| XGBoost | Gradient boosted tree model |

Hyperparameter tuning via **RandomizedSearchCV** was applied to both Logistic Regression and XGBoost.  
Result: Models were biased toward the majority class — poor recall on defaulters.

---

#### Attempt 2 — Random Under-Sampling

The majority class (non-defaulters) was down-sampled using `RandomUnderSampler` to achieve a balanced class distribution.

| Model | Notes |
|---|---|
| Logistic Regression | Improved minority class recall |
| XGBoost | Applied best params from Attempt 1 |

---

#### Attempt 3 — SMOTETomek + Optuna (Final Approach)

**SMOTETomek** was used for combined over-sampling (SMOTE on minority) and under-sampling (Tomek links removal) — preserving informative boundary samples.

Hyperparameter optimisation was performed using **Optuna** (Bayesian optimisation) with 50 trials optimising macro F1-score across 3-fold cross-validation:

**Final Logistic Regression Parameters (Optuna-tuned):**

```
C              : (optimised via log-uniform search: 1e-4 to 1e4)
solver         : (from: lbfgs, liblinear, saga, newton-cg)
tol            : (log-uniform: 1e-6 to 1e-1)
class_weight   : balanced
max_iter       : 10,000
```

This was selected as the **final production model**.

---

### 7. Model Evaluation

#### Classification Report

The final Logistic Regression with SMOTETomek achieved meaningful recall on the minority (default) class while maintaining reasonable precision on the majority class.

#### ROC Curve & AUC

```
AUC Score:         ~0.98 (area under ROC curve)
Gini Coefficient:  2 × AUC − 1  ≈  0.96
```

A high AUC indicates strong rank-ordering capability — the model effectively separates high-risk from low-risk borrowers.

#### KS Statistic (Kolmogorov-Smirnov)

The KS statistic measures maximum separation between the cumulative distribution of defaulters and non-defaulters across deciles:

| Decile (High → Low Risk) | Cum Event Rate | Cum Non-event Rate | KS |
|---|---|---|---|
| 10 | Highest probability segment | Lowest | Maximum |
| ... | ... | ... | ... |
| 1 | Lowest probability segment | Highest | Near 0 |

A high KS value in the top 2–3 deciles confirms strong **rank ordering** — the model correctly concentrates defaults in the high-probability deciles.

#### Feature Importance (Logistic Regression Coefficients)

The most influential features by coefficient magnitude, in order of predictive impact:

| Feature | Direction | Interpretation |
|---|---|---|
| `avg_dpd_per_delinquency` | + | Higher average days past due → higher default risk |
| `delinquency_ratio` | + | Greater % of delinquent months → higher risk |
| `credit_utilization_ratio` | + | Over-extended credit → higher risk |
| `loan_to_income` | + | Higher leverage → higher risk |
| `loan_tenure_months` | + | Longer tenure → higher risk |
| `age` | − | Older applicants → lower risk |
| `number_of_open_accounts` | ± | Context-dependent |
| `loan_type_Unsecured` | + | Unsecured loans carry more risk |

---

### 8. Credit Score Calculation

The credit score is derived from the model's raw log-odds output — a standard practise in lendingscorecard design:

$$
\text{Credit Score} = 300 + 600 \times \log\!\left(\frac{P(\text{non-default})}{P(\text{default})}\right)
$$

| Credit Score | Risk Rating |
|---|---|
| 300 – 499 | Poor |
| 500 – 549 | Average |
| 650 – 749 | Good |
| 750 – 900 | Excellent |

This log-odds transformation ensures the score is **monotonically decreasing with default probability** and maps naturally to the industry-standard 300–900 credit score range.

---

## Model Performance Summary

| Metric | Value |
|---|---|
| Algorithm | Logistic Regression |
| Resampling | SMOTETomek |
| Hyperparameter Tuning | Optuna (50 trials, 3-fold CV) |
| AUC | ~0.98 |
| Gini Coefficient | ~0.96 |
| Score Range | 300 – 900 |
| Training Set Size | 75% of 50,000 records |
| Test Set Size | 25% of 50,000 records |

---

## Web Application

Built with **Streamlit**, the application provides a real-time credit risk assessment interface.

### Input Sections

| Section | Fields |
|---|---|
| **Applicant Profile** | Age, Annual Income, Loan Amount |
| **Loan Details** | Loan-to-Income Ratio (auto-computed), Loan Tenure, Avg DPD |
| **Risk Indicators** | Delinquency Ratio, Credit Utilization, Open Accounts |
| **Classification** | Residence Type, Loan Purpose, Loan Type |

### Output Panel

After clicking **Calculate Risk**, three metric cards are rendered:

| Card | Content | Color Logic |
|---|---|---|
| Default Probability | % likelihood of default | Red (≥ 50%) / Green (< 50%) |
| Credit Score | Numeric score 300–900 | Purple accent |
| Risk Rating | Poor / Average / Good / Excellent | Red / Orange / Green / Blue |

---

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.14 |
| Data Processing | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Machine Learning | scikit-learn |
| Gradient Boosting | XGBoost |
| Imbalance Handling | imbalanced-learn (SMOTETomek, RandomUnderSampler) |
| Hyperparameter Tuning | Optuna |
| Statistical Analysis | statsmodels (VIF) |
| Model Serialisation | joblib |
| Web Application | Streamlit |
| Environment | Python venv |

---

## Inference Flow

```
User Input (Streamlit UI)
        │
        ▼
prediction_helper.prepare_df()
  - Constructs feature DataFrame
  - Applies MinMaxScaler to numeric columns
  - Selects model feature columns
        │
        ▼
prediction_helper.calculate_credit_risk()
  - Computes log-odds: X·coef + intercept
  - Derives default probability via sigmoid
  - Converts to credit score via log-odds scaling
  - Maps score to rating band
        │
        ▼
Output: default_probability, credit_score, rating
        │
        ▼
Streamlit renders three metric cards
```

---

*Vox Finance — Intelligent Credit Risk Intelligence Platform*
