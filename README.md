# fall-2025-patient-risk-identification
Team project: fall-2025-patient-risk-identification

# Project README

## 1. Project Overview

This project focuses on predicting 30-day readmissions for inpatient encounters using a comprehensive set of patient demographics, encounter-level features, and historical utilization data. The goal is to develop a predictive model that can identify patients at high risk of readmission, enabling proactive interventions and improving patient outcomes. The project explores both Logistic Regression and Random Forest classifiers, with a particular emphasis on hyperparameter tuning for the latter to optimize performance.

## 2. Data Sources

The analysis utilizes synthetic healthcare data generated for 20,000 patients using SyntheaTM, an open-source patient population simulation made available by The MITRE Corporation. The data files that we generated from Syhthea data generator API can be accessed on this link: https://drive.google.com/drive/folders/1-d8zTI60hQ3ZJFp9j2BC-M-Nw5ojywzq?usp=sharing.  The following modules were integrated:

*   **`encounters.csv`**: Core encounter information, filtered for inpatient stays.
*   **`patients.csv`**: Patient demographic data, including age, gender, race, and ethnicity.
*   **`observations.csv`**: Clinical observations (e.g., vital signs, lab results).
*   **`procedures.csv`**: Medical procedures performed during encounters.
*   **`conditions.csv`**: Patient diagnoses and health conditions.
*   **`immunizations.csv`**: Immunization records.
*   **`medications.csv`**: Medication prescriptions and administrations.

## 3. Methodology

### 3.1. Data Loading and Initial Filtering

All relevant CSV files were loaded into pandas DataFrames. Inpatient encounters were isolated from the `encounters.csv` dataset, and a left join was performed with `patients.csv` to enrich encounter records with demographic details.

### 3.2. Feature Engineering

Extensive feature engineering was performed, including:

*   **Encounter-level features**: Length of stay (`LOS_days`), admission hour (`admit_hour`), day of week (`admit_dow`), weekend/night admissions (`is_weekend`, `is_night`), and calendar context (`admit_month`, `admit_quarter`, `admit_season`).
*   **Readmission target**: A binary `READMIT_30D` flag indicating readmission within 30 days of discharge for inpatient encounters.
*   **Prior encounter gap**: Time since the last discharge (`gap_since_prev_discharge_days`).
*   **Rolling prior-year utilization**: Counts of total, ER, inpatient, and outpatient encounters in the past 365 days (`prior365_total_enc`, `prior365_ER`, `prior365_inpatient`, `prior365_outpatient`), along with mean LOS and total cost.
*   **Previous encounter class**: The type of the immediately preceding encounter (`prev_class`).
*   **Module-specific features**: Counts, binary flags, and proportions of categories for observations, procedures, conditions, immunizations (e.g., flu vaccine within 1 year), and medications (number active at admission).

### 3.3. Preprocessing for Modeling

*   **Missing Value Imputation**: Numerical columns were imputed with their median, and categorical columns with a 'Missing' category or False for booleans.
*   **Age Calculation**: Patient age was calculated from `BIRTHDATE`.
*   **Train/Test Split**: Data was split using `GroupShuffleSplit` (80% train, 20% test) to ensure that all encounters from a single patient appear in either the training or testing set, preventing data leakage across patients.
*   **Categorical Encoding**: One-hot encoding was applied to categorical features (`GENDER`, `RACE`, `ETHNICITY`).
*   **Feature Scaling**: Numerical features were scaled using `StandardScaler`.

### 3.4. Model Training and Evaluation

Two classification models were implemented and evaluated:

*   **Logistic Regression**: A baseline model trained with `class_weight='balanced'` to address class imbalance.
*   **Random Forest Classifier**: An ensemble model known for its robustness, also trained with `class_weight='balanced'`.

Both models were evaluated using 5-fold `GroupKFold` cross-validation. Key metrics included AUC, Precision, Recall, F1-score, and Accuracy, with a focus on the readmission (Class 1) metrics due to class imbalance.

### 3.5. Hyperparameter Tuning (Random Forest)

`GridSearchCV` was employed to fine-tune the Random Forest Classifier's hyperparameters using `GroupKFold` for robust evaluation. The parameter grid explored `n_estimators`, `max_depth`, `min_samples_split`, and `class_weight`. `roc_auc` was used as the primary refit metric.

## 4. Model Comparison: Random Forest vs. Logistic Regression

| Metric                | Logistic Regression | Random Forest (Untuned) |
|:----------------------|:--------------------|:------------------------|
| **Average AUC**       | 0.866               | 0.898                   |
| **Precision (Class 1)** | 0.453               | 0.455                  |
| **Recall (Class 1)**    | 0.822               | 0.936                   |
| **F1-score (Class 1)**  | 0.584               | 0.612                   |
| **Accuracy**          | 0.812               | 0.810                   |

The Random Forest model generally performed better than Logistic Regression, exhibiting a higher AUC and significantly improved recall for identifying readmission cases.

## 5. Tuned Random Forest Model Summary

After hyperparameter tuning, the best Random Forest model achieved the following:

*   **Best Hyperparameters**: `n_estimators=200`, `max_depth=None`, `min_samples_split=10`, `class_weight='balanced'`.
*   **Performance on Test Set**:
    *   **AUC Score**: `0.895`
    *   **Precision (Class 1 - Readmission)**: `0.487`
    *   **Recall (Class 1 - Readmission)**: `0.603`
    *   **F1-score (Class 1 - Readmission)**: `0.539`
    *   **Overall Accuracy**: `0.837`

The Tuned RF improves overall calibration and efficiency (higher accuracy and precision), but misses 40% of true readmissions, which could mean missed opportunities for timely care.

## 6. Key Findings and Conclusions

*   Both untuned and tuned Random Forest classifiers achieved higher discriminative power (AUC ≈ 0.90) than the Logistic Regression baseline (AUC = 0.866), demonstrating stronger capability to identify patients at risk of 30-day hospital readmission.
*   Logistic Regression remained valuable for understanding variable effects through odds ratios but underperformed compared to ensemble models in predictive accuracy and recall.
*   Hyperparameter tuning introduced a precision–recall trade-off. Untuned Random Forest: High recall (0.936) — captured over 93% of true readmissions, ideal for early intervention -- vs. Tuned Random Forest: Improved precision (0.487) and accuracy (0.837) but reduced recall (0.603) — better suited for resource-efficient deployment. This illustrates the trade-off between maximizing sensitivity and controlling false positives.
*   **Key predictors driving readmission risk:**
*    **Prior inpatient encounters**
*    **Gap since previous discharge**
*    **Length of stay (LOS)**
*    **Procedure intensity (especially chemotherapy-related)**
*    **Patient demographics and prior utilization patterns**
 
*   **Model recommendation**:
For intervention-focused use cases, deploy the Untuned Random Forest with calibrated probability thresholds to preserve high recall while managing alert volume.
For efficiency-focused programs, the Tuned Random Forest offers improved precision and accuracy.

### Next Steps

*   Threshold calibration using precision–recall analysis.
*   SHAP value integration for transparent, patient-level feature explanations.
*   External validation on new cohorts to assess generalizability.
*   Deployment integration with EHR dashboards or clinical workflows.

## Reference
*   Jason Walonoski, Mark Kramer, Joseph Nichols, Andre Quina, Chris Moesel, Dylan Hall, Carlton Duffett, Kudakwashe Dube, Thomas Gallagher, Scott McLachlan, Synthea: An approach, method, and software mechanism for generating synthetic patients and the synthetic electronic health care record, Journal of the American Medical Informatics Association, Volume 25, Issue 3, March 2018, Pages 230–238, https://doi.org/10.1093/jamia/ocx079
