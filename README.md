
---

## Problem Description

- Task: Binary classification
- Target variable: diagnosed_diabetes
- Evaluation metric: ROC-AUC
- Training data: approximately 124,000 rows
- Test data: approximately 300,000 rows

### Dataset Characteristics

- Entirely synthetic dataset
- Generated from a deep learning model
- Key medical predictors such as HbA1c, fasting glucose, postprandial glucose, and diabetes risk score were removed
- Significant distribution shift from the original synthetic dataset

---

## Methodology

### 1. Baseline Modeling

The project began with strong baseline models commonly used for tabular data:
- CatBoost
- LightGBM

These models were selected because they handle non-linear relationships well and are well-suited for structured health data. CatBoost was particularly useful due to its native handling of categorical features.

Initial validation and leaderboard results showed ROC-AUC scores around 0.69–0.70, which was unexpectedly low for a diabetes prediction task. This prompted further investigation into the dataset.

---

### 2. Dataset Investigation

Upon reviewing Kaggle discussions and comparing the competition dataset to the original source data, it became clear that:
- Highly predictive clinical features were intentionally removed
- Several remaining features had altered or clipped distributions
- The dataset exhibited strong distribution shift

This reframed the problem from "predicting diabetes accurately" to "extracting weak signal from intentionally degraded data."

---

### 3. Feature Engineering

To compensate for the missing clinical markers, domain-inspired feature engineering was applied. This included:
- Physiological ratios (e.g., triglycerides to HDL cholesterol)
- Interaction terms (e.g., BMI × age)
- Blood pressure derivatives (pulse pressure, mean arterial pressure)
- Lifestyle normalization features (activity per BMI, sleep relative to screen time)
- Composite risk indicators based on known medical heuristics

Feature importance analysis confirmed that several engineered features contributed meaningful signal.

This step improved cross-validation ROC-AUC to approximately 0.72.

---

### 4. Model Optimization

Both CatBoost and LightGBM were tuned further by:
- Increasing the number of boosting iterations
- Using early stopping
- Running out-of-fold validation where computationally feasible

GPU acceleration was explored but limited by runtime and environment constraints. Final models were trained using CPU-based configurations that balanced performance and stability.

---

### 5. Ensembling and Blending

To improve robustness under distribution shift, multiple ensembling strategies were evaluated:
- Soft probability averaging
- Rank averaging
- Weighted blends of LightGBM and CatBoost predictions

Rank averaging proved to be the most stable approach, although gains were modest due to high correlation between models.

---

## Results

- Best local validation ROC-AUC: approximately 0.726
- Public leaderboard ROC-AUC: approximately 0.695–0.700
- Top leaderboard score at the time: approximately 0.707

The gap between local validation and leaderboard performance highlighted the difficulty introduced by distribution shift and synthetic noise.

---

## Key Learnings

- Strong baseline models are essential before attempting complex techniques
- Feature engineering can recover meaningful signal even when key predictors are removed
- Ensembling only helps when models are sufficiently diverse
- High validation scores do not guarantee leaderboard performance under distribution shift
- Understanding the data generation process is often more important than model complexity

---

## Conclusion

This project demonstrates a realistic machine learning workflow: starting with assumptions, testing them, encountering limitations, and adapting strategy based on data behavior. Rather than focusing solely on leaderboard rank, the primary outcome was a deeper understanding of model robustness, feature importance, and the challenges posed by synthetic and shifted datasets.

The notebooks in this repository reflect that learning process end-to-end.
