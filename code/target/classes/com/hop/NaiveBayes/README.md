# Implementation of Naive Bayes Classifier

## 1. Description

This module is responsible for implementing and evaluating the Naive Bayes classification model. The primary objective is to build a robust predictive tool to accurately assess the mortality risk of COVID-19 patients (Target: `DIED`).

By computing the probabilistic output of the Naive Bayes algorithm, we aim to establish a high-performance tool that serves as the **best available solution** for identifying critical clinical outcomes. This model prioritizes the early identification of high-risk patients to support clinical decision-making.

**Task Owner:** Pham Hoang Phuong

## 2. Algorithm Implementation

The classification model is implemented using the **Naive Bayes** algorithm provided by the Weka library (`weka.classifiers.bayes.NaiveBayes`).

To ensure the model is robust and reliable, we apply a specific validation workflow:

1.  **10-Fold Cross-Validation:** The data is split into 10 folds to test the model's stability across different subsets of data.
2.  **Strategic Undersampling (Inside the Loop):**
    *   **The Problem:** The raw data is heavily imbalanced (93% Survived vs. 7% Died). A standard model would simply guess "Survived" and achieve high accuracy but fail to detect actual deaths.
    *   **The Solution:** Inside each validation fold, we apply `strategicUndersample` exclusively to the **Training Set**. We balance the distribution to 50/50 (Died vs. Survived).
    *   **The Effect:** This forces the Naive Bayes model to treat mortality cases as equally important as survival cases during the learning process.
3.  **Realistic Evaluation:** The model is evaluated on the **unmodified Test Set**, which retains the original "real-world" imbalance. This ensures our reported metrics reflect true performance in a clinical setting.

## 3. Feature Selection & Justification

From the available attributes, we selected a focused subset of 7 features: `"SEVERITY_INDEX", "AGE_GROUP", "SEX", "PNEUMONIA", "COPD", "ASTHMA", "TOBACCO"` and the target variable `"DIED"`.

We selected these specific features for two key reasons:

### 1. The "Independence" Assumption
Naive Bayes assumes that every feature is mathematically independent.
*   **Issue:** In the raw data, attributes like `INTUBED`, `ICU`, and `PATIENT_TYPE` are highly correlated. Including them would cause the model to "double count" the risk signal, violating the algorithm's probability rules.
*   **Fix:** We replaced these correlated columns with the single **`SEVERITY_INDEX`** to provide a clean signal (0-3) without breaking the independence assumption.

### 2. The "Sparsity" Problem
Naive Bayes relies on probability tables for feature combinations.
*   **Issue:** With too many features (e.g., 22), the probability table becomes massive, and specific combinations of rare attributes might have zero examples in the training set (Zero Frequency error).
*   **Fix:** Keeping the feature set small ensures that every combination is robustly represented in the data, preventing the model from crashing on rare patient profiles.

## 4. Results & Verification

As this is currently our primary predictive model, we evaluated its performance to determine if it meets the safety requirements for a medical tool.

| Metric | Value | Analysis |
| :--- | :--- | :--- |
| **Recall (Sensitivity)** | **92.29%** | **Excellent.** The model successfully identifies over 92% of actual mortality cases. This is our most critical success metric. It means the model acts as a strong safety net and rarely misses a dying patient. |
| **Precision** | **39.10%** | **Suboptimal.** To achieve such high Recall, the model casts a "wide net," flagging many survivors as "At Risk." Currently, only ~39% of the patients flagged by the model actually die. |
| **Accuracy** | **88.89%** | **Strong.** The model correctly classifies the vast majority of the 1 million+ instances. |

### Conclusion
This Naive Bayes implementation represents the **safest possible configuration** for our current needs.

While the **Precision (39.10%)** is lower than ideal—meaning we have a high rate of "False Alarms"—this is a necessary trade-off to achieve the **High Recall (92.29%)**. In a medical context, missing a dying patient (False Negative) is far worse than incorrectly flagging a healthy one (False Positive). Therefore, despite the precision bottleneck, this model succeeds in its primary goal: ensuring critical cases are identified.