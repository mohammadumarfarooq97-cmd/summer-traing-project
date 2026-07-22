import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import GridSearchCV

# Import data preparation function
from data_preprocessing import prepare_data

def train_and_evaluate_models(data, plots_dir='plots'):
    """
    Trains multiple classification models, compares their performance,
    and returns evaluation metrics.
    """
    print("\n--- 5. MODEL BUILDING & TRAINING ---")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract data
    X_train_res = data['X_train_resampled']
    y_train_res = data['y_train_resampled']
    X_test_proc = data['X_test_processed']
    y_test = data['y_test']
    
    # Initialize models
    # Note: SVM is wrapped in CalibratedClassifierCV to avoid SVC probability deprecations
    # and provide calibrated prediction probabilities
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Machine': CalibratedClassifierCV(SVC(random_state=42)),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    # Dictionary to hold results
    results = {}
    trained_models = {}
    
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        print(f"Training {name}...")
        # Fit model on resampled training data
        model.fit(X_train_res, y_train_res)
        trained_models[name] = model
        
        # Predict on processed test data
        y_pred = model.predict(X_test_proc)
        y_prob = model.predict_proba(X_test_proc)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
        
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-score': f1,
            'ROC-AUC': auc
        }
        
        # Plot micro-average ROC curve
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
    # Finalize ROC plot
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'roc_curves_comparison.png'), dpi=150)
    plt.close()
    print("- Saved roc_curves_comparison.png")
    
    # Convert results to DataFrame
    df_results = pd.DataFrame(results).T
    print("\nModel Comparison Table:")
    print(df_results.to_markdown())
    
    # Save comparison table as CSV
    df_results.to_csv('model_comparison_results.csv', index=True)
    
    return trained_models, df_results

def plot_confusion_matrices(trained_models, data, plots_dir='plots'):
    """
    Generates and saves confusion matrices for all models.
    """
    print("\n--- 6. MODEL EVALUATION: CONFUSION MATRICES ---")
    X_test_proc = data['X_test_processed']
    y_test = data['y_test']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    for idx, (name, model) in enumerate(trained_models.items()):
        y_pred = model.predict(X_test_proc)
        cm = confusion_matrix(y_test, y_pred)
        
        labels = ['<=25K', '25K-50K', '50K-75K', '75K-100K', '>100K']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                    xticklabels=labels, yticklabels=labels)
        axes[idx].set_title(f'Confusion Matrix - {name}')
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_ylabel('True Label')
        
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'confusion_matrices.png'), dpi=150)
    plt.close()
    print("- Saved confusion_matrices.png")

def plot_feature_importances(trained_models, data, plots_dir='plots'):
    """
    Plots feature importance (for tree models) or coefficients (for linear models) if available.
    """
    print("\n--- 6. FEATURE IMPORTANCE / COEFFICIENTS ---")
    feature_names = data['feature_names']
    
    for name, model in trained_models.items():
        importances = None
        title = ""
        
        # Check for tree-based feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            title = f'Feature Importance - {name}'
        # Check for linear model coefficients (1D array for binary classification)
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
            title = f'Feature Coefficients (Absolute) - {name}'
            
        if importances is not None:
            indices = np.argsort(importances)[::-1]
            top_n = min(12, len(feature_names))
            top_indices = indices[:top_n]
            
            plt.figure(figsize=(10, 6))
            y_labels = [feature_names[i] for i in top_indices]
            sns.barplot(
                x=importances[top_indices], 
                y=y_labels, 
                hue=y_labels, 
                palette='viridis', 
                legend=False
            )
            plt.title(title)
            plt.xlabel('Relative Importance / Weight')
            plt.ylabel('Feature')
            plt.tight_layout()
            filename = f'feature_importance_{name.lower().replace(" ", "_")}.png'
            plt.savefig(os.path.join(plots_dir, filename), dpi=150)
            plt.close()
            print(f"- Saved {filename}")

def tune_best_model(trained_models, data, df_results):
    """
    Finds the best-performing model based on F1-score, performs hyperparameter tuning
    using GridSearchCV, and returns the tuned model.
    """
    print("\n--- 7. HYPERPARAMETER TUNING ---")
    # Identify the best model based on F1-score
    best_model_name = df_results['F1-score'].idxmax()
    print(f"Best baseline model based on F1-score: {best_model_name}")
    
    X_train_res = data['X_train_resampled']
    y_train_res = data['y_train_resampled']
    X_test_proc = data['X_test_processed']
    y_test = data['y_test']
    
    # Configure parameter grids for each model type dynamically
    grids = {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['lbfgs']
        },
        'Decision Tree': {
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        },
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [2, 5]
        },
        'K-Nearest Neighbors': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        },
        'Support Vector Machine': {
            'estimator__C': [0.1, 1, 10]  # SVM wrapped in CalibratedClassifierCV
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    }
    
    # Fresh base models for clean fitting
    base_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Machine': CalibratedClassifierCV(SVC(random_state=42)),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    base_model = base_models[best_model_name]
    param_grid = grids[best_model_name]
    
    print(f"Grid Search Parameter Grid for {best_model_name}: {param_grid}")
    
    # Perform Grid Search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train_res, y_train_res)
    
    print(f"\nBest Parameters found: {grid_search.best_params_}")
    best_estimator = grid_search.best_estimator_
    
    # Evaluate performance before and after tuning
    y_pred_before = trained_models[best_model_name].predict(X_test_proc)
    f1_before = f1_score(y_test, y_pred_before, average='weighted')
    
    y_pred_after = best_estimator.predict(X_test_proc)
    f1_after = f1_score(y_test, y_pred_after, average='weighted')
    
    print(f"F1-score before tuning: {f1_before:.4f}")
    print(f"F1-score after tuning:  {f1_after:.4f}")
    
    return best_estimator, best_model_name, f1_after

def serialize_pipeline(best_model, preprocessor, filename='best_model.joblib'):
    """
    Wraps the preprocessing and trained classifier in a single sklearn Pipeline and saves it.
    """
    print("\n--- 8. MODEL SERIALIZATION ---")
    
    # Create final deployment pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', best_model)
    ])
    
    # Save the pipeline
    joblib.dump(pipeline, filename)
    print(f"Successfully saved preprocessing + model pipeline as '{filename}'")
    
    # Verification check: Reload and predict on a sample
    reloaded_pipeline = joblib.load(filename)
    print("Reloaded model verification check passed.")
    return reloaded_pipeline

def main():
    # 1. Load preprocessed data
    data = prepare_data()
    
    # 2. Train models and compare
    trained_models, df_results = train_and_evaluate_models(data)
    
    # 3. Generate confusion matrices
    plot_confusion_matrices(trained_models, data)
    
    # 4. Generate feature importance / coefficient plots
    plot_feature_importances(trained_models, data)
    
    # 5. Tune the best baseline model
    best_tuned_estimator, model_name, f1_tuned = tune_best_model(trained_models, data, df_results)
    
    # 6. Overall Comparison: Select the absolute highest scoring model to save
    # This prevents deploying a tuned model if a baseline model still performs better
    all_f1s = df_results['F1-score'].to_dict()
    all_f1s[f"Tuned {model_name}"] = f1_tuned
    
    overall_best_name = max(all_f1s, key=all_f1s.get)
    print(f"\nOverall Best Model Comparison:")
    for name, f1 in all_f1s.items():
        print(f"  {name}: F1={f1:.4f}")
        
    print(f"\nSelected Model for Deployment: '{overall_best_name}'")
    
    if overall_best_name.startswith("Tuned "):
        final_classifier = best_tuned_estimator
    else:
        final_classifier = trained_models[overall_best_name]
        
    # 7. Save the final pipeline
    serialize_pipeline(final_classifier, data['preprocessor'])

if __name__ == '__main__':
    main()
