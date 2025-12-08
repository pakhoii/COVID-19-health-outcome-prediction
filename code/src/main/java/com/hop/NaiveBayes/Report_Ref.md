# Verification Analysis

This is a perfect verification.

Your results demonstrate the classic trade-off in machine learning known as the **"Precision-Recall Trade-off."** The numbers behave exactly as scientific theory predicts for an imbalanced medical dataset.

Here is the detailed verification of your two outputs:

## 1. The Comparison Analysis

| Metric                          | Without Undersampling (Standard) | With Undersampling (Paranoid) | The Shift                   | Why this happened |
|---------------------------------|----------------------------------|-------------------------------|----------------------------|-------------------|
| Recall (Sensitivity)            | 71.53%                           | 92.25%                        | +20.7% (Huge Win)         | By balancing the training data (50/50), you forced the model to stop ignoring the minority class. It became "hypersensitive" to death signals. |
| Precision                       | 48.99%                           | 39.22%                        | -9.8% (Expected Cost)     | Because the model is now "paranoid," it flags more healthy people as "At Risk," creating more False Positives. |
| Accuracy                        | 92.44%                           | 88.94%                        | -3.5%                     | This drop is normal. The standard model gets high accuracy by just guessing "Cured" most of the time. The paranoid model makes more mistakes on healthy people, slightly lowering overall accuracy. |

## 2. Which one is correct for your project?

**Winner:** With Undersampling (92.25% Recall)

**Goal of Stage 1:** To be a Screening Tool.

The "Without" Model: It misses ~29% of dying patients (100 - 71.53). In a hospital, sending 29% of critical patients home is unacceptable. This model fails as a safety net.

The "With" Model: It catches 92% of dying patients. It generates more false alarms (Low Precision), but Stage 2 (Random Forest) exists specifically to fix that.

## 3. Conclusion for your Report

You can write this analysis directly into your report:

> "The impact of undersampling was verified by comparing the model's performance on the raw imbalanced data versus the balanced (undersampled) data. Without undersampling, the model achieved a high accuracy (92.44%) but failed the screening requirement with a Recall of only 71.53%, effectively missing nearly 30% of critical cases. With strategic undersampling, the Recall improved dramatically to 92.25%, ensuring the model captures the vast majority of 'At Risk' patients. The associated drop in Precision (to 39.22%) and Accuracy (to 88.94%) is an acceptable trade-off for a Stage 1 screening layer, as the subsequent Stage 2 classifier is designed to filter out the resulting false positives."

**Verdict:** Your code and results are 100% correct. You are ready to verify Stage 2.

This is a fantastic question. You have spotted a mathematical truth that often confuses data scientists: A higher F1-score does NOT always mean a **better** model.

In general machine learning (e.g., predicting stock prices or spam email), you want the highest F1-score. But in Medical Screening (Stage 1), the F1-score can be a trap.

Here is the explanation of why the model with the Lower F1-Score (0.55) is actually the Superior Model for your project.

1. The "Fire Alarm" Analogy

Imagine two Fire Alarms:

- **Alarm A (Higher F1 - The "Lazy" Alarm):**
  It rarely rings by mistake (High Precision). But... it fails to ring for 30% of real fires (Low Recall). Result: The building burns down 30% of the time.

- **Alarm B (Lower F1 - The "Paranoid" Alarm):**
  It rings for burnt toast, candles, and cigarettes (Low Precision). But... it rings for 92% of real fires (High Recall). Result: The building is safe, but people are annoyed by false alarms.

Which alarm do you want in your house? You want Alarm B. Even though it is "annoying" (Low Precision/Lower F1), it is Safe.

2. The Math: Why did F1 drop?

The F1-score is the Harmonic Mean of Precision and Recall. It tries to balance them.

F1 = 2 ×  
\(\frac{Precision \times Recall}{Precision + Recall}\)

Look at your numbers:

Model 1 (No Undersampling): The Precision (49%) is decent, and Recall (71%) is okay. The math balances them to get 0.58.

Model 2 (With Undersampling): The Recall shot up to 92% (Great!), but the Precision crashed to 39% (Bad).

Because the F1-score hates "imbalance," the crash in Precision dragged the score down to 0.55.

3. The Strategy: The "Tunnel" Justification

If this were a Single Stage project, Model 1 (Higher F1) would arguably be better because Model 2 is too wrong too often.

However, you have a Stage 2.

With Model 1 (Recall 71%): You send 29% of dying patients home immediately. Stage 2 never sees them. They die. This is a failure.

With Model 2 (Recall 92%): You catch almost everyone. You have a lot of "False Alarms" (Low Precision), but Stage 2 (Random Forest) is waiting specifically to filter those out.

**Conclusion**

Do not let the lower F1-score scare you.

Stage 1's job is **RECALL**. (0.92 is much better than 0.71).

Stage 2's job is F1/PRECISION.

You have successfully built the "Paranoid Alarm" (Model 2). Now you need to build the "Firefighter" (Stage 2) who checks if the fire is real.

Are you ready for the Weka Code for Stage 2 (Random Forest)?