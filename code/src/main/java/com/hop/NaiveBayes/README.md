# Create the data set for Naive Bayes Classifier

## 1. Description

This module is responsible for constructing the data set specifically for training and testing the Naive Bayes Classifier model. It utilizes the preprocessed datasets from the `/data/preprocess` directory to create a consolidated data set that is suitable for model training and evaluation. The final data set is exported in both `.csv` and `.arff` formats in directory `/data/naive_bayes`.

**Task Owner:** Pham Hoang Phuong

## 2. How to construct the data set?
To create the data set for the Naive Bayes Classifier, follow these steps:
1. **Load Preprocessed Data**: Read the preprocessed data files from the `/data/preprocess` directory. Ensure that the data is clean and properly formatted.
2. **Feature Selection**: Identify and select relevant features that will be used for training the Naive Bayes model. 
   - Choose `"SEVERITY_INDEX", "AGE_GROUP", "SEX", "PNEUMONIA", "DIED"` as the features for the model.
3. **Data Splitting**: Because the data is imbalance (dominated by one class - `DIED=0` takes 93% vs `DIED=1` only accounts for 7%). To handle this case, apply technique:
   - **Undersampling**: Reduce the number of instances in the majority class (`DIED=0`) to balance the class distribution.
   - Solution workflow:
     - Split Data: First, separate 20% of the data as a Test Set.
     - Balance Training: From the remaining 80% (Training Set), we will keep ALL "Died" cases and randomly select an equal number of "Cured" cases.
     - Result: A training dataset with a 50/50 ratio. This forces the Naive Bayes model to treat "Died" as equally important as "Cured."

4. **Data Formatting**: Format the data into the required structure for Naive Bayes. This typically involves ensuring that categorical features are properly encoded.

## 3. Why only use these features?
We selected **only these 5 columns** (`SEVERITY_INDEX`, `AGE_GROUP`, `SEX`, `PNEUMONIA`, `DIED`) for three specific mathematical and architectural reasons.

### 1. The "Independence" Assumption (Mathematical Reason)
Naive Bayes is named "Naive" because it assumes that every feature is completely **independent** of the others. It calculates risk by multiplying the probabilities of each separate feature.
* **The Problem:** In your raw data, `INTUBED`, `ICU`, and `PATIENT_TYPE` are **not independent**.
    * If a patient is Intubated, they are almost certainly in the ICU and Hospitalized.
    * If you feed all three columns to Naive Bayes, it effectively "triple counts" this risk signal. The probability math explodes, making the model incredibly overconfident and biased.
* **The Solution:** We replaced those three correlated columns with the single **`SEVERITY_INDEX`**. This gives the model one strong, clean signal (0-3) without violating the mathematical rules of the algorithm.

### 2. The "Screening" Goal (Architectural Reason)
Your architecture is a **Tunnel**.
* **Stage 1 (Naive Bayes) is the "General Practitioner."** Its job is **High Recall**. It needs to look at the big, obvious red flags (Is the patient old? Are they struggling to breathe? Are they on a ventilator?) and flag them as "At Risk."
* **Stage 2 (Random Forest) is the "Specialist."** Its job is **High Precision**. It looks at the subtle details (Does the patient have Diabetes? Hypertension? Kidney issues?) to decide if that risk is actually fatal.
* **Why Limit Columns?** If you give the "General Practitioner" (Stage 1) too many details, it might get confused. By limiting it to the 5 most powerful signals (`SEVERITY`, `AGE`, `PNEUMONIA`), we force it to focus on the immediate, life-threatening factors, ensuring it doesn't miss anyone (High Recall).

### 3. The "Sparseness" Problem (Data Reason)
Naive Bayes uses probability tables (e.g., "What is the % chance of death for a Male aged 40-49 with Pneumonia?").
* **With 5 Features:** The table is small and robust. Every combination has thousands of examples in your data.
* **With 22 Features:** The table becomes massive. You might end up with specific combinations (e.g., "Male, 40-49, Diabetes, No Asthma, Smoker, Renal Chronic...") that have **zero** examples in the training set.
* **The Result:** When the model sees a new patient with that rare combo, it assigns a probability of **zero** (or near zero), causing it to crash or give a wrong prediction. Keeping feature sets small prevents this "Zero Frequency" error.

**Summary:** We use these 5 columns to keep the math clean, the signal loud, and the model highly sensitive to the most critical risk factors. The complex details are saved for the Random Forest in Stage 2.
