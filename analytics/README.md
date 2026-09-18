Module 2 - Analytics & Predictive Modeling
Setup & Execution:
Install all the required libraries using pip install -r requirements.txt.
Run 01_eda.ipynb first. This notebook performs Exploratory Data Analysis (EDA) and creates the offline fallback dataset titanic.csv.
Run 02_modeling.ipynb next. It handles preprocessing, model training, class imbalance handling, grid search, and the regression task.
Design Decisions & Methodology

Missing Values:
Missing values were handled based on the type of column. Numerical columns such as age and fare were filled using the median value. Categorical columns such as embarked were filled using the most frequently occurring value.

Train/Test Split:
The dataset was divided into training and testing sets using a stratified split based on the survived column. This helps maintain a similar proportion of survivors and non-survivors in both sets.

Preventing Data Leakage:
All preprocessing steps, including imputation, scaling, and encoding, were placed inside a scikit-learn Pipeline and ColumnTransformer. These steps were fitted only on the training data to prevent information from the test set from affecting the model.

Model Performance:
Logistic Regression:
Accuracy: ~0.80
Precision: ~0.76
Recall: ~0.72
F1 Score: ~0.74
ROC-AUC: ~0.85

Decision Tree:
Accuracy: ~0.78
Precision: ~0.73
Recall: ~0.74
F1 Score: ~0.73
ROC-AUC: ~0.77

Random Forest:
Accuracy: ~0.81
Precision: ~0.78
Recall: ~0.75
F1 Score: ~0.76
ROC-AUC: ~0.84

Linear Regression (Side-Task):
MAE: ~8.45
RMSE: ~13.20
R²: ~0.41
Adjusted R²: ~0.40

Final Recommendation:
The Random Forest Classifier was selected as the recommended model based on the results. It achieved an accuracy of around 0.81 and an F1 score of around 0.76, which were the highest among the classification models tested.
It can also capture non-linear relationships between features such as passenger class, age, and fare. Since it combines multiple decision trees instead of relying on a single tree, it can provide more stable predictions and reduce the risk of overfitting.