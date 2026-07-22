# Employee Salary Prediction Project 💼

This repository contains a complete, end-to-end Machine Learning pipeline to predict employee salary categories (`<=50K` vs `>50K`) based on demographic and professional features. It includes data generation, extensive Exploratory Data Analysis (EDA), modular preprocessing, class imbalance handling, multi-model evaluation, hyperparameter tuning, model serialization, and an interactive Streamlit web deployment.

## 🚀 Quick Start Guide

### 1. Set Up Environment & Install Dependencies
First, ensure you have Python 3.10+ installed. Clone or copy this project folder, open a terminal inside the project directory, and install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset
If `employee_salary_data.csv` is not present, generate it by running:
```bash
python generate_data.py
```
This creates a realistic dataset of 2,500 samples with a naturally imbalanced distribution (~76% <=50K, ~24% >50K) and typical missing values in columns like `workclass` and `occupation`.

### 3. Run Exploratory Data Analysis (EDA)
Execute the EDA script to analyze dataset characteristics, print summary statistics, check class distribution, and save plots to the `plots/` folder:
```bash
python eda_analysis.py
```

### 4. Train, Tune, & Serialize the Model
Run the main training script. It loads the dataset, splits it (stratified 80/20 train/test), handles class imbalance using SMOTE on the training set, trains multiple classifiers, compares performance, tunes the best-performing model using Grid Search, and serializes the complete pipeline (preprocessor + model) to disk:
```bash
python train_model.py
```

### 5. Launch the Streamlit Web Application
Run the interactive deployment app:
```bash
streamlit run app.py
```
This will spin up a local development server and open a premium web interface in your default browser.

---

## 📂 Project Directory Structure

```text
├── employee_salary_data.csv     # Generated dataset (demographics + salary)
├── generate_data.py             # Script to generate synthetic data
├── eda_analysis.py              # Exploratory Data Analysis and plotting
├── data_preprocessing.py        # Modular preprocessing functions & SMOTE logic
├── train_model.py               # Model training, comparison, tuning & serialization
├── app.py                       # Streamlit interactive web deployment code
├── requirements.txt             # Project library requirements and versions
├── best_model.joblib            # Serialized pipeline containing preprocessing + classifier
├── model_comparison_results.csv # Summary table of evaluation metrics
├── plots/                       # Generated EDA and model evaluation charts
└── README.md                    # Project documentation (this file)
```

---

## ⚙️ ML Pipeline Details

### 1. Data Preprocessing & Leakage Prevention
Preprocessing is wrapped inside an scikit-learn `ColumnTransformer` to avoid data leakage:
- **Numerical columns** (`age`, `experience`, `hours_per_week`): Imputed with the `median` value and normalized using `StandardScaler`.
- **Categorical columns** (`workclass`, `education`, `occupation`): Imputed with the `most_frequent` (mode) category and encoded using `OneHotEncoder(handle_unknown='ignore')`.

### 2. Class Imbalance (SMOTE)
Because the dataset is imbalanced (76% <=50K vs 24% >50K), models trained on it would default to predicting the majority class. **SMOTE** is applied **strictly to the training split** to balance the classes 50/50, ensuring the test set remains clean and realistic.

### 3. Model Building & Comparison
We train six classifiers:
- **Logistic Regression**: High interpretability, strong baseline.
- **Decision Tree**: Tree-based model showing basic split criteria.
- **Random Forest**: Ensemble bagging classifier to reduce variance.
- **K-Nearest Neighbors**: Instance-based distance classifier.
- **Support Vector Machine (SVC)**: Robust kernel-based classification.
- **Gradient Boosting**: Ensemble boosting classifier to minimize bias.

Performance metrics evaluated on the test set include: **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **ROC-AUC**.

### 4. Hyperparameter Tuning & Deployment Pipeline
The best classifier (e.g. Random Forest) is tuned using `GridSearchCV`. The final model is saved as a complete pipeline including the preprocessor, meaning the Streamlit application can feed raw input DataFrames directly into the model without needing separate preprocessing code.

---

## 🎨 Streamlit Web Application Features
- **Interactive UI**: Elegant dark mode design with custom typography, clean layout sections, and hover-active buttons.
- **Robust Input Bounds**: Dynamically constrains work experience inputs based on the selected age to ensure logical entries.
- **Result visualization**: Renders the salary class prediction with colored confidence progress bars.
- **Benchmarking Charts**: Embeds interactive Seaborn bar charts comparing the user's attributes (age, hours, experience) to the average values of workers inside both salary classes in the database.
