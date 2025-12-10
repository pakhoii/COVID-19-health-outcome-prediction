# Implementation of Cascade Tunnel Architecture (Stage 2)

## 1. Description

This module implements the **Cascade Tunnel Architecture**, a two-step machine learning pipeline designed to refine the predictions of the initial Naive Bayes model.

While the single-stage Naive Bayes model achieved high sensitivity, it suffered from a "Precision Bottleneck," generating a high volume of false alarms. The objective of this task is to introduce a secondary "Specialist" model (Random Forest) to re-evaluate high-risk cases, thereby improving overall accuracy and precision without completely sacrificing the safety net established in Stage 1.

**Task Owner:** Pham Hoang Phuong

## 2. The Problem: The Precision Bottleneck

As analyzed in the previous phase, the standalone Naive Bayes model presented a critical operational flaw:

*   **High Recall (92.29%)**: It successfully caught most critical cases.
*   **Low Precision (39.10%)**: To catch those cases, it flagged a massive number of survivors as "At Risk."

In a real-world hospital scenario, this creates a **"False Alarm" crisis**. Relying solely on Stage 1 would force medical staff to allocate scarce resources (ICU beds, ventilators) to patients who don't actually need them. We needed a way to filter these false positives.

## 3. The Solution: Cascade Architecture

To resolve this, we implemented a **Cascade Tunnel** consisting of two distinct stages:

### Stage 1: Naive Bayes (The Screener)
*   **Role:** The "General Practitioner."
*   **Function:** We keep the model from the previous task exactly as is. Its job is to cast a wide net and rapidly filter out the obviously safe patients. If it has any doubt, it flags the patient.

### Stage 2: Random Forest (The Specialist)
*   **Role:** The "Critical Care Specialist."
*   **Function:** This model only looks at the "Suspects" flagged by Stage 1.
*   **Why Random Forest?** Unlike Naive Bayes, Random Forest is excellent at understanding complex, non-linear relationships (e.g., how Age interacts specifically with Diabetes or Hypertension). It is used to distinguish between a "False Alarm" and a "True Critical" case.

## 4. Implementation Strategy

The logic follows the `CascadeTunnelCrossValidation` workflow described in the report:

1.  **The Broad Screen (Stage 1):** The Naive Bayes model screens every patient.
2.  **The Handoff:**
    *   If Stage 1 predicts **0 (Safe)**: The patient is deemed low-risk. We stop here (Efficiency gain).
    *   If Stage 1 predicts **1 (Risk)**: The patient is marked as a "Suspect" and passed to Stage 2.
3.  **Specialist Review (Stage 2):** The Random Forest evaluates the suspect.
    *   *Configuration:* Limited to 20 trees and depth of 10 to prevent overfitting.
    *   *Weighting:* Instead of undersampling, we use **Class Weighting**. We tell the model that missing a death (Weight 2.0) is twice as bad as a false alarm (Weight 1.0). This forces the Specialist to be careful before dismissing a risk.
4.  **Merging Results:** The final prediction combines both stages. For the "Suspects," the Random Forest has the final say, effectively "overruling" the Naive Bayes to remove false alarms.

## 5. Results & Verification

The transition to the Cascade Tunnel significantly altered the performance profile, as shown in the comparative analysis:

| Metric | Stage 1 (Naive Bayes) | Cascade Tunnel (Final) | The Shift | Analysis |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 88.89% | **93.58%** | **+4.7%** | **Improved.** We significantly reduced the number of misclassified cases (from ~116k to ~67k). |
| **Precision** | 39.10% | **54.97%** | **+15.8%** | **Major Win.** The Random Forest successfully filtered out the "false alarms," distinguishing true critical cases from survivors with similar symptoms. |
| **Recall** | 92.29% | **69.04%** | **-23.2%** | **The Trade-off.** The stricter filtering made the model more conservative. While we gained precision, we lost some sensitivity. |

### Conclusion

The Cascade Tunnel successfully solved the **Precision Bottleneck**.

By allowing the Random Forest to re-evaluate risk, we created a model that is far more efficient for resource allocation. While the drop in Recall means the final model is less of a "safety net" than the raw Naive Bayes, the substantial increase in Precision (54.97%) and Accuracy (93.58%) makes it a much more practical tool for triage decisions in an overwhelmed healthcare system.