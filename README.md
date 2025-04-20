# 📧 Email Campaign Click Prediction using Machine Learning

## I. Executive Summary

This project addresses the optimization of email marketing campaigns through the development of a machine learning model. The core objective is to predict the likelihood of users clicking on email links, enabling targeted campaigns that maximize user engagement and minimize unnecessary email distribution.

---

## II. Project Goals

- To accurately predict the probability of users clicking on links within emails.  
- To provide actionable insights for optimizing email campaign strategies.  
- To quantify the potential improvement in click-through rate (CTR) resulting from model implementation.  
- To analyze email campaign performance across different user segments.

---

## III. Data Sources

The analysis is based on three structured datasets:

- `email_table`: Contains detailed information about individual emails.  
- `email_opened_table`: Records instances of emails being opened by users.  
- `link_clicked_table`: Tracks instances of users clicking on links within emails.

---

## IV. Data Dictionary

### `email_table`
- `email_id`: Unique identifier for each email.  
- `email_text`: Categorical variable indicating email text length (`"long text"` or `"short text"`).  
- `email_version`: Categorical variable representing email personalization (`"personalized"` or `"generic"`).  
- `hour`: Numerical variable indicating the hour the email was sent (local time).  
- `weekday`: Categorical variable representing the day of the week the email was sent.  
- `user_country`: Categorical variable indicating the user's country.  
- `user_past_purchases`: Numerical variable representing the number of past purchases made by the user.  

### `email_opened_table`
- `email_id`: Foreign key referencing `email_table`, indicating an opened email.  

### `link_clicked_table`
- `email_id`: Foreign key referencing `email_table`, indicating an email with a clicked link.

---

## V. Data Analysis Workflow

The project follows a structured data analysis workflow:

1. **Data Loading**: Loading data from CSV files into Pandas DataFrames.  
2. **Data Cleaning**: Handling missing values, inconsistencies, and data type conversions.  
3. **Exploratory Data Analysis (EDA)**: Understanding data distributions, relationships, and patterns.  
4. **Data Analysis**: In-depth examination of variables and their impact on email campaign performance.  
5. **Data Visualization**: Creating informative visualizations to support analysis and recommendations.  
6. **Reporting**: Summarizing key findings, insights, and actionable recommendations.  

### Jupyter Notebooks:
- `analytics1.ipynb`: Initial data exploration and descriptive statistics.  
- `table_joins.ipynb`: Data integration and feature engineering.  
- `visualisation1.ipynb`: Creation of data visualizations.  
- `model_selection.ipynb`: Comparative evaluation of machine learning models.  
- `knn_model.ipynb`: Implementation of the selected machine learning model (K-Nearest Neighbors).  
- `questions_and_answers.ipynb`: Addressing specific project questions and providing detailed answers.

---

## VI. Data Preparation

- **Data Loading**:  
  Load datasets (`email_table.csv`, `email_opened_table.csv`, `link_clicked_table.csv`) using `pd.read_csv()`.

- **Data Understanding**:  
  Use `.head()`, `.info()`, `.describe()`, and `.isnull().sum()` for inspection.  
  Analyze categorical features with `.value_counts()`.

- **Target Variable Creation**:  
  - Left join `email_df` with `opened_df` on `email_id`.  
  - Join result with `clicked_df` on `email_id`.  
  - Assign `1` to `clicked` if the email exists in `clicked_df`, otherwise `0`.

- **Feature Engineering**:
  - **Categorical Encoding**: One-hot encoding for `email_text`, `email_version`, `user_country`, `weekday`.  
  - **Datetime Feature Extraction** from `hour`:
    - `is_morning`: 6AM–12PM  
    - `is_afternoon`: 12PM–6PM  
    - `is_evening`: 6PM–10PM  
    - `is_night`: 10PM–6AM  
  - **Feature Scaling**: Apply `StandardScaler` or `MinMaxScaler` to `user_past_purchases`.

---

## VII. Model Development

- **Data Splitting**:  
  Train-test split (80%-20%) for model evaluation.

- **Model Selection**:  
  Evaluate multiple models:
  - Logistic Regression  
  - Random Forest  
  - Gradient Boosting (XGBoost, LightGBM)  
  - Naive Bayes  
  - K-Nearest Neighbors (KNN) – *Selected Model*

- **Model Training**:  
  Train KNN using the training set.

- **Hyperparameter Optimization**:  
  Use cross-validation and grid search for tuning (e.g., `C` for Logistic Regression, `n_estimators` and `max_depth` for Random Forest).

---

## VIII. Model Evaluation

- **Prediction Generation**:  
  Use `.predict()` and `.predict_proba()` for test set predictions.

- **Evaluation Metrics**:
  - **Precision**: Correctly predicted click events.  
  - **Recall**: Actual click events correctly identified.  
  - **F1-score**: Harmonic mean of precision and recall.  
  - **AUC-ROC**: Discriminatory power of the model.

- **Result Interpretation**:  
  Analyze overfitting and feature importance (e.g., coefficients in Logistic Regression).

---

## IX. Optimization and Recommendations

- **Probability Threshold Adjustment**:  
  Fine-tune decision threshold to optimize business KPIs using precision-recall curves.

- **Campaign Optimization Recommendations**:
  - Target users with predicted click probability above threshold `X`.
  - Use short email formats during morning hours for specific demographics.

---

## X. Dependencies

Python libraries used in this project:

- **Data Manipulation and Analysis**:  
  `pandas`, `numpy`

- **Machine Learning**:  
  `scikit-learn`

- **Data Visualization**:  
  `matplotlib`, `seaborn`
