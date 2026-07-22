import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath='employee_salary_data.csv'):
    """
    Loads the employee salary dataset and prints its basic properties.
    """
    print("--- 1. DATA LOADING & UNDERSTANDING ---")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    print("Data Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nSummary Statistics (Numerical):")
    print(df.describe().T)
    
    print("\nSummary Statistics (Categorical):")
    print(df.describe(exclude=[np.number]).T)
    
    target_col = 'salary'
    feature_cols = [col for col in df.columns if col != target_col]
    print(f"\nTarget Column: {target_col}")
    print(f"Feature Columns: {feature_cols}\n")
    
    return df, target_col, feature_cols

def perform_eda(df, target_col, feature_cols, output_dir='plots'):
    """
    Performs univariate, bivariate, correlation, and outlier analysis, saving plots to output_dir.
    """
    print("--- 2. EXPLORATORY DATA ANALYSIS (EDA) ---")
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Target Class Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_col, data=df, hue=target_col, palette='viridis', legend=False)
    plt.title('Target Class (Salary) Distribution')
    plt.xlabel('Salary Class')
    plt.ylabel('Count')
    class_dist = df[target_col].value_counts()
    class_dist_pct = df[target_col].value_counts(normalize=True) * 100
    for idx, (val, pct) in enumerate(zip(class_dist, class_dist_pct)):
        plt.text(idx, val + 20, f"{val} ({pct:.1f}%)", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'target_distribution.png'), dpi=150)
    plt.close()
    print("- Saved target_distribution.png")
    
    # 2. Univariate Analysis (Numerical features)
    num_features = df.select_dtypes(include=[np.number]).columns.tolist()
    fig, axes = plt.subplots(1, len(num_features), figsize=(15, 4))
    for idx, col in enumerate(num_features):
        sns.histplot(df[col], kde=True, ax=axes[idx], color='skyblue')
        axes[idx].set_title(f'Distribution of {col.capitalize()}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Density')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'univariate_numerical.png'), dpi=150)
    plt.close()
    print("- Saved univariate_numerical.png")
    
    # 3. Univariate Analysis (Categorical features)
    cat_features = [col for col in df.select_dtypes(exclude=[np.number]).columns if col != target_col]
    for col in cat_features:
        plt.figure(figsize=(10, 5))
        order = df[col].value_counts().index
        sns.countplot(y=col, data=df, order=order, hue=col, palette='Set2', legend=False)
        plt.title(f'Count Plot of {col.capitalize()}')
        plt.xlabel('Count')
        plt.ylabel(col.capitalize())
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'univariate_{col}.png'), dpi=150)
        plt.close()
        print(f"- Saved univariate_{col}.png")
        
    # 4. Bivariate Analysis: Salary vs Numerical Features
    fig, axes = plt.subplots(1, len(num_features), figsize=(15, 5))
    for idx, col in enumerate(num_features):
        sns.boxplot(x=target_col, y=col, data=df, ax=axes[idx], hue=target_col, palette='muted', legend=False)
        axes[idx].set_title(f'{col.capitalize()} vs {target_col.capitalize()}')
        axes[idx].set_xlabel(target_col)
        axes[idx].set_ylabel(col)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bivariate_numerical.png'), dpi=150)
    plt.close()
    print("- Saved bivariate_numerical.png")

    # 5. Bivariate Analysis: Salary vs Categorical Features (Education and Occupation)
    for col in ['education', 'occupation', 'workclass']:
        plt.figure(figsize=(10, 6))
        # Plot stacked bar chart of percentage distribution
        cross_tab = pd.crosstab(df[col], df[target_col], normalize='index') * 100
        cross_tab.plot(kind='bar', stacked=True, color=['#2b5c8f', '#d95f02'], figsize=(10, 6))
        plt.title(f'Salary Distribution by {col.capitalize()}')
        plt.ylabel('Percentage')
        plt.xlabel(col.capitalize())
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Salary')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'bivariate_{col}.png'), dpi=150)
        plt.close()
        print(f"- Saved bivariate_{col}.png")
        
    # 6. Correlation Heatmap for Numerical Features
    # Note: Categorical features are encoded using pandas category codes for heatmap visualization
    corr_df = df.copy()
    for col in cat_features + [target_col]:
        corr_df[col] = corr_df[col].astype('category').cat.codes
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap (All Features)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150)
    plt.close()
    print("- Saved correlation_heatmap.png")
    
    # 7. Outliers Detection (Boxplots)
    plt.figure(figsize=(10, 4))
    for idx, col in enumerate(num_features):
        plt.subplot(1, 3, idx+1)
        sns.boxplot(y=df[col], color='lightcoral')
        plt.title(f'Outliers in {col.capitalize()}')
        plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'outliers_boxplots.png'), dpi=150)
    plt.close()
    print("- Saved outliers_boxplots.png")
    
    print("\nEDA Completed successfully. All plots saved to the 'plots' folder.")

if __name__ == '__main__':
    df, target, features = load_data()
    perform_eda(df, target, features)
