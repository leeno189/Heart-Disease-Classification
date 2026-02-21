# Predictive Modeling for Heart Disease Classification

## Overview
This project develops a robust machine learning pipeline to predict whether a patient has heart disease based on clinical features using the Heart Disease UCI dataset. The primary focus was handling missing data efficiently and evaluating the impact of different feature selection techniques on model accuracy and sensitivity. 

## Methodology & Architecture
* **Advanced Data Imputation:** Handled missing cholesterol data by treating it as a secondary prediction task, utilizing a LightGBM Regressor to accurately impute missing values and preserve the overall data distribution.
* **Data Scaling & Balancing:** Applied Min-Max normalization and utilized the Synthetic Minority Over-sampling Technique (SMOTE). While the dataset was naturally well-proportioned (55/45 split), SMOTE was applied to enforce strict 50/50 class parity, ensuring absolutely zero majority-class bias during clinical prediction.
* **Feature Selection Analysis:** Conducted a comparative analysis of two feature selection techniques to optimize computational efficiency and model interpretability:
  * **Univariate Filter (SelectKBest):** Utilized Chi-Square statistics to isolate the top 10 most significant features.
  * **Wrapper Method (Recursive Feature Elimination - RFE):** Deployed a Decision Tree estimator to recursively rank and select the top 10 predictors.
* **Classification Models:** Trained and comparatively evaluated four distinct classifiers: Logistic Regression, Support Vector Machine (SVM), eXtreme Gradient Boosting (XGBoost), and a Multi-layer Perceptron (MLP).

## Key Findings & Results
* **Best Performing Model:** The MLP neural network trained on RFE-selected features achieved the highest overall performance.
* **Metrics:** The final model reached an accuracy of 88% and an F1-score of 89.68%.
* **Clinical Impact:** The MLP model successfully minimized false negatives (producing only 12), which is highly critical in a medical context to avoid misdiagnosing patients who actually have heart disease.

## Tech Stack
* **Language:** Python
* **Libraries:** Scikit-Learn, LightGBM, Pandas, NumPy, Matplotlib/Seaborn, Imbalanced-Learn (SMOTE)
