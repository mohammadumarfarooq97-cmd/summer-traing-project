import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def get_preprocessor(numeric_features, ordinal_features, categorical_features):
    """
    Creates and returns a ColumnTransformer for numerical, ordinal, and categorical preprocessing.
    """
    # 1. Numerical pipeline: Median Imputation + StandardScaler
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 2. Ordinal pipeline: Most Frequent Imputation + OrdinalEncoder
    ord_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(
            categories=[['High School', 'Associate', 'Bachelors', 'Masters', 'Doctorate']], 
            handle_unknown='use_encoded_value', 
            unknown_value=-1
        ))
    ])
    
    # 3. Categorical pipeline: Most Frequent Imputation + OneHotEncoder
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine all preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numeric_features),
            ('ord', ord_pipeline, ordinal_features),
            ('cat', cat_pipeline, categorical_features)
        ]
    )
    
    return preprocessor

def prepare_data(filepath='employee_salary_data.csv', test_size=0.2, random_state=42):
    """
    Loads data, performs feature engineering, splits into train/test, 
    applies ColumnTransformer preprocessing, and applies SMOTE.
    """
    print("--- 3. DATA PREPROCESSING ---")
    if not os.path.exists(filepath):
        # Fallback in case of local execution issues
        filepath = os.path.basename(filepath)
        
    df = pd.read_csv(filepath)
    
    # Target and features separation
    target_col = 'salary'
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col]
    
    # Map target classes: <=25K -> 0, 25K-50K -> 1, 50K-75K -> 2, 75K-100K -> 3, >100K -> 4
    y_encoded = y.map({
        '<=25K': 0,
        '25K-50K': 1,
        '50K-75K': 2,
        '75K-100K': 3,
        '>100K': 4
    })
    print("Target mapped: '<=25K'->0, '25K-50K'->1, '50K-75K'->2, '75K-100K'->3, '>100K'->4")
    
    # --- Feature Engineering ---
    # Interaction term: experience per age
    X['experience_ratio'] = X['experience'] / (X['age'] + 1e-5)
    print("Engineered feature added: 'experience_ratio' = experience / age")
    
    # Identify feature types
    numeric_features = ['age', 'experience', 'hours_per_week', 'experience_ratio']
    ordinal_features = ['education']
    categorical_features = ['workclass', 'occupation']
    
    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, stratify=y_encoded, random_state=random_state
    )
    print(f"Train set shape: {X_train.shape[0]} rows")
    print(f"Test set shape: {X_test.shape[0]} rows")
    
    # Define and fit the preprocessor on the training data only
    preprocessor = get_preprocessor(numeric_features, ordinal_features, categorical_features)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after encoding for transparency/interpretability
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    encoded_cat_features = cat_encoder.get_feature_names_out(categorical_features).tolist()
    feature_names = numeric_features + ordinal_features + encoded_cat_features
    
    print("\n--- 4. HANDLING CLASS IMBALANCE (SMOTE) ---")
    # Check class distribution before SMOTE
    y_train_counts = y_train.value_counts().sort_index()
    print("Class distribution in training set before SMOTE:")
    class_labels = {0: '<=25K', 1: '25K-50K', 2: '50K-75K', 3: '75K-100K', 4: '>100K'}
    for val, label in class_labels.items():
        count = y_train_counts.get(val, 0)
        pct = count / len(y_train) * 100
        print(f"  Class {val} ({label}): {count} ({pct:.2f}%)")
    
    # Apply SMOTE to the training set only
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
    
    # Check class distribution after SMOTE
    y_resampled_counts = pd.Series(y_train_resampled).value_counts().sort_index()
    print("\nClass distribution in training set after SMOTE:")
    for val, label in class_labels.items():
        count = y_resampled_counts.get(val, 0)
        pct = count / len(y_train_resampled) * 100
        print(f"  Class {val} ({label}): {count} ({pct:.2f}%)")
    print(f"Resampled training set shape: {X_train_resampled.shape[0]} rows")
    
    # Return split datasets, the fitted preprocessor, and clean feature names
    return {
        'X_train_orig': X_train,
        'y_train_orig': y_train,
        'X_train_resampled': X_train_resampled,
        'y_train_resampled': y_train_resampled,
        'X_test_processed': X_test_processed,
        'X_test_orig': X_test,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'feature_names': feature_names
    }

if __name__ == '__main__':
    data_dict = prepare_data()
