
# Data Preprocessing Pipeline

## 1. Description

This module is responsible for preprocessing the raw datasets located in the `/data` directory. The primary goal is to clean, standardize, and transform the data into a usable format for analysis and model training. The final cleaned datasets are then exported in both `.csv` (for general use) and `.arff` (for Weka) formats.

**Task Owner:** Pham Anh Khoi

---

## 2. Implementation Details

The preprocessing logic is tailored for three distinct datasets: `covid.csv`, `symptoms.csv`, and `comorbidity.csv`.

### 2.1. `covid.csv`

This is a large, complex dataset requiring multiple cleaning steps.

#### **Initial Data Profile:**

<details>
<summary>Click to see unique values before preprocessing</summary>

-   **USMER** (2 unique): `[2, 1]`
-   **MEDICAL_UNIT** (13 unique): `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]`
-   **SEX** (2 unique): `[1, 2]`
-   **PATIENT_TYPE** (2 unique): `[1, 2]`
-   **DATE_DIED** (401 unique): `['03/05/2020', '03/06/2020', '9999-99-99', ...]`
-   **INTUBED** (4 unique): `[97, 1, 2, 99]`
-   **PNEUMONIA** (3 unique): `[1, 2, 99]`
-   **AGE** (121 unique): `[65, 72, 55, ...]`
-   **PREGNANT** (4 unique): `[2, 97, 98, 1]`
-   **DIABETES** (3 unique): `[2, 1, 98]`
-   **COPD** (3 unique): `[2, 1, 98]`
-   **ASTHMA** (3 unique): `[2, 1, 98]`
-   **INMSUPR** (3 unique): `[2, 1, 98]`
-   **HIPERTENSION** (3 unique): `[1, 2, 98]`
-   **OTHER_DISEASE** (3 unique): `[2, 1, 98]`
-   **CARDIOVASCULAR** (3 unique): `[2, 1, 98]`
-   **OBESITY** (3 unique): `[2, 1, 98]`
-   **RENAL_CHRONIC** (3 unique): `[2, 1, 98]`
-   **TOBACCO** (3 unique): `[2, 1, 98]`
-   **CLASIFFICATION_FINAL** (7 unique): `[3, 5, 7, 6, 1, 2, 4]`
-   **ICU** (4 unique): `[97, 2, 1, 99]`

</details>

#### **Preprocessing Steps:**

1.  **Filter Non-COVID Patients:** Remove any records where `CLASIFFICATION_FINAL` indicates the patient was not confirmed to have COVID-19.
2.  **Feature Extraction:** Create a new binary feature `DIED` from the `DATE_DIED` column. If `DATE_DIED` is `9999-99-99`, `DIED` is `0` (alive); otherwise, it is `1` (died).
3.  **Remove Redundant Columns:** Drop the original `DATE_DIED` and `CLASIFFICATION_FINAL` columns as they are no longer needed.
4.  **Standardize Binary Values:** The dataset uses `1` for YES and `2` for NO. This step converts this encoding to the more standard `1` (YES) and `0` (NO).
5.  **Mark Missing Values:** Values such as `97`, `98`, and `99` are used to represent missing or unknown information. These are converted to standard missing value markers.
6.  **Impute Missing Values:** A sophisticated imputation method is used:
    -   For each column with missing values, a Chi-Square test is performed against all other columns to find the one with the strongest correlation.
        $$X^2=\sum{\frac{(\text{Observed}-\text{Expected})^2}{\text{Expected}}}$$
    -   The data is then grouped by the values of this most correlated column.
    -   Missing values are imputed using the **mode** (most frequent value) of each respective group. Any remaining missing values are filled using a global mode.
1.  **Finalize and Export:** The fully cleaned `Instances` object is returned and saved.

#### **Final Data Profile:**

<details>
<summary>Click to see unique values after preprocessing</summary>

-   **USMER** (2 unique): `[0, 1]`
-   **MEDICAL_UNIT** (13 unique): `[1, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]`
-   **SEX** (2 unique): `[1, 0]`
-   **PATIENT_TYPE** (2 unique): `[1, 0]`
-   **INTUBED** (2 unique): `[0, 1]`
-   **PNEUMONIA** (2 unique): `[1, 0]`
-   **AGE** (118 unique): `[65.0, 72.0, 55.0, ...]`
-   **PREGNANT** (2 unique): `[0, 1]`
-   **DIABETES** (2 unique): `[0, 1]`
-   **COPD** (2 unique): `[0, 1]`
-   **ASTHMA** (2 unique): `[0, 1]`
-   **INMSUPR** (2 unique): `[0, 1]`
-   **HIPERTENSION** (2 unique): `[1, 0]`
-   **OTHER_DISEASE** (2 unique): `[0, 1]`
-   **CARDIOVASCULAR** (2 unique): `[0, 1]`
-   **OBESITY** (2 unique): `[0, 1]`
-   **RENAL_CHRONIC** (2 unique): `[0, 1]`
-   **TOBACCO** (2 unique): `[0, 1]`
-   **ICU** (2 unique): `[0, 1]`
-   **DIED** (2 unique): `[1, 0]`

</details>

---

### 2.2. `symptoms.csv` & `comorbidity.csv`

These datasets are relatively clean and require fewer preprocessing steps. The main tasks are to handle file encoding issues and standardize feature representation for consistency.

> **Note:** The source CSV files for these datasets are encoded in **UTF-8 with BOM**. The preprocessing script handles the Byte Order Mark to ensure attribute names are parsed correctly.

#### **`comorbidity.csv`**

This dataset contains information about patient comorbidities.

-   **Initial State:** The data is complete (no missing values). The `sex` attribute uses `1` for female and `2` for male.
-   **Preprocessing Steps:**
    1.  Standardize the `sex` attribute: `1` (female) remains `1`, while `2` (male) is converted to `0`.
    2.  Convert all relevant attributes from numeric to nominal type.

<details>
<summary>See Data Profile Changes</summary>

| Attribute         | Before (Unique Values) | After (Unique Values) |
| ----------------- | ---------------------- | --------------------- |
| **sex**           | `[1, 2]`               | `[1, 0]`              |
| **age**           | `[85.0, 63.0, ...]`    | `[85.0, 63.0, ...]`   |
| **hypertension**  | `[0, 1]`               | `[0, 1]`              |
| **cardiovascular**| `[0, 1]`               | `[0, 1]`              |
| ...               | ...                    | ...                   |
| **outcomes**      | `[1, 0]`               | `[1, 0]`              |

</details>

#### **`symptoms.csv`**

This dataset contains information about patient symptoms.

-   **Initial State:** The data is complete (no missing values). It uses a `gender` attribute with `1` for female and `2` for male.
-   **Preprocessing Steps:**
    1.  Standardize the `gender` attribute values: `1` (female) remains `1`, while `2` (male) is converted to `0`.
    2.  **Harmonize Attribute Name:** Rename the `gender` attribute to `sex` for consistency across all datasets.
    3.  Convert all relevant attributes from numeric to nominal type.

<details>
<summary>See Data Profile Changes</summary>

| Attribute         | Before (Unique Values) | After (Unique Values) |
| ----------------- | ---------------------- | --------------------- |
| headache          | `[0, 1]`               | `[0, 1]`              |
| fever             | `[0, 1]`               | `[0, 1]`              |
| ...               | ...                    | ...                   |
| **gender**        | `[1, 2]`               | (Renamed to `sex`)    |
| **sex**           | (N/A)                  | `[1, 0]`              |
| outcomes          | `[1, 0]`               | `[1, 0]`              |

</details>