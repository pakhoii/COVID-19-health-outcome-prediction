# Preprocess

## Description
Perform preprocessing on the raw dataset located in `/data`, then generate and export the cleaned dataset in both `.csv` and `.arff` formats.

**Task Owner:** Pham Anh Khoi


## Implementation
### `covid.csv`
Task list:
1. Drop out any patient that are not getting COVID-19
2. Extract feature `DIED` from `DATE_DIED` to indicate whether the patient is dead or not.
3. Remove unnessary columns
4. The dataset using `1` for YES and `2` for NO. We should standardize these binary value to the familiar set (`1` for YES and `0` for NO)
5. Marking the missing values
6. For each column, use the chi-square test to identify the column with the strongest correlation. Then, group the current column’s values according to the classes of that most correlated column, and impute any missing values using the mode within each corresponding group.
$$X^2=\sum{\frac{(\text{Observed}-\text{Expected})^2}{\text{Expected}}}$$
7. Return the cleaned data
8. Export the cleaned data


### `symptoms.csv`

### `comorbidity.csv`
