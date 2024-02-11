# Parkinson's Disease Detection using Advanced Machine Learning Techniques

Parkinson's disease is a progressive neurodegenerative disorder that significantly affects people's quality of life, mainly targeting motor functions. It is characterized by tremor, rigidity, bradykinesia, and postural instability, caused by the loss of dopamine-producing neurons in the brain. Early detection is crucial for starting treatment before the disease progresses too far, improving patients' quality of life.

## Overview

An existing analysis for Parkinson's disease detection using machine learning techniques like XGBoost and Naive Bayes classifiers has been developed to improve model accuracy. However, this solution falls short in achieving the desired accuracy and effectively handling non-linearity in the data. 

In the proposed system, the primary objective is to enhance the detection accuracy significantly. The proposed system introduces advanced machine learning techniques such as Random Forest and Support Vector Machine (SVM). Random Forest's capability to handle nonlinear relationships makes it well-suited for capturing complex patterns in Parkinson's disease data. SVM classifiers are renowned for their accuracy. This model detects Parkinson’s disease with higher accuracy and handles non-linearity compared to existing systems.

## Objectives of Proposed System

- Leverage Random Forest and Support Vector Machine algorithms to significantly improve accuracy compared to the existing system.
- Utilize these algorithms to model intricate interactions among features, addressing non-linearity challenges present in the data.
- Reduce model complexity and enhance performance by focusing on key feature selection and mitigating issues from irrelevant or redundant data.
- Utilize advanced techniques and optimized feature selection to empower healthcare providers for informed decisions and tailored treatments to improve patient care.

## Software Requirements

- Python with libraries: NumPy, Pandas, Matplotlib, Seaborn, scikit-learn.
- Google Colab for machine learning tasks.

## Hardware Requirements

- Basic modern processor (e.g., Intel Core i3 or equivalent).
- At least 4 GB RAM (more for larger datasets).
- Adequate storage for code and datasets.

## Libraries Used

- `pandas` (aliased as `pd`): Used for data manipulation and handling.
- `numpy` (aliased as `np`): Provides support for numerical operations and arrays.
- `sklearn.model_selection`: Includes functions for model selection and evaluation, such as `train_test_split`.
- `sklearn.preprocessing`: Provides data preprocessing tools, including `StandardScaler` for feature scaling.
- `sklearn.ensemble`: Contains machine learning ensemble methods like `RandomForestClassifier`.
- `sklearn.metrics`: Offers various metrics for evaluating model performance, such as `accuracy_score`, `precision_score`, `recall_score`, and `f1_score`.
- `matplotlib.pyplot` (aliased as `plt`): Used for creating data visualizations and plots.
- `seaborn` (aliased as `sns`): A data visualization library that complements Matplotlib.
- `sklearn.svm`: Provides Support Vector Machine (SVM) algorithms, including `SVC`.

## Testing

- **Unit Testing**: Ensures individual components or functions work correctly.
- **System Testing**: Verifies that the entire system works as a whole.
- **Integration Testing**: Checks that different components or modules within the system interact correctly.
- **Acceptance Testing**: Verifies that the system meets requirements and performs as expected from a user's perspective.

## Usage

1. **Data Import and Examination**: Start by importing necessary libraries, loading a Parkinson's disease dataset, and inspecting its structure and missing values.
2. **Data Visualization**: Visualize data distributions for specific features using histograms to gain insights.
3. **Data Imputation**: Impute missing values in specific columns with median values to prepare the data for modeling.
4. **Data Splitting**: Split the dataset into features (X) and the target variable (y), further dividing them into training and testing sets using `train_test_split`. Perform feature scaling on the numeric features using `StandardScaler`.
5. **Random Forest Classifier**: Train a Random Forest classifier on the training data. Calculate and print evaluation metrics such as precision, recall, F1-score, and accuracy.
6. **Support Vector Machine (SVM) Classifier**: Initialize, train, and evaluate an SVM classifier using the same evaluation metrics as the Random Forest model.
7. **Prediction for New Data**: Use the Random Forest model to predict the likelihood of a person having Parkinson's disease based on a set of input features.

---

Feel free to contribute, report issues, or provide feedback to enhance this project!
