import numpy as np
import pandas as pd

def generate_synthetic_data(num_samples=10000, random_state=42):
    np.random.seed(random_state)
    
    # 1. Age (ranging from 18 to 65)
    age = np.random.randint(18, 66, size=num_samples)
    
    # 2. Education levels and corresponding probabilities
    education_options = ['High School', 'Associate', 'Bachelors', 'Masters', 'Doctorate']
    education_probs = [0.35, 0.15, 0.35, 0.12, 0.03]
    education = np.random.choice(education_options, size=num_samples, p=education_probs)
    
    # Map education to years of education / base level
    edu_map = {'High School': 12, 'Associate': 14, 'Bachelors': 16, 'Masters': 18, 'Doctorate': 21}
    edu_years = np.array([edu_map[e] for e in education])
    
    # 3. Experience (should correlate with age and education)
    # Experience ~ Age - Edu_years - 6, but with some randomness and minimum 0
    experience = age - edu_years - 6
    experience = np.clip(experience + np.random.randint(-3, 4, size=num_samples), 0, None)
    
    # 4. Work-class
    workclass_options = ['Private', 'Self-Emp', 'State-Gov', 'Federal-Gov', 'Local-Gov', 'Without-Pay']
    workclass_probs = [0.70, 0.12, 0.06, 0.04, 0.07, 0.01]
    workclass = np.random.choice(workclass_options, size=num_samples, p=workclass_probs)
    
    # 5. Occupation
    occupation_options = ['Admin', 'Sales', 'Tech-Support', 'Exec-Managerial', 'Prof-Specialty', 
                          'Craft-Repair', 'Other-Service', 'Farming-Fishing']
    occupation_probs = [0.15, 0.15, 0.10, 0.15, 0.20, 0.10, 0.10, 0.05]
    occupation = np.random.choice(occupation_options, size=num_samples, p=occupation_probs)
    
    # 6. Hours per week (centered around 40 hours)
    hours_per_week = np.random.normal(40, 8, size=num_samples).astype(int)
    hours_per_week = np.clip(hours_per_week, 10, 80)
    
    # 7. Salary Class Calculation (<=50K vs >50K)
    # We want a realistic model where:
    # - Higher education increases probability of >50K
    # - Higher experience/age increases probability of >50K
    # - Certain occupations (Exec-Managerial, Prof-Specialty) have higher probability
    # - Hours-per-week increases probability
    
    base_salary = 10.0 # baseline in thousands
    
    # Calculate a continuous salary value (in thousands)
    salary_num = (
        base_salary +
        0.5 * age +
        1.2 * (edu_years - 12) +
        0.8 * experience +
        0.2 * (hours_per_week - 40)
    )
    
    # Add occupation multiplier
    occ_salary_effect = {
        'Admin': 0.0, 'Sales': 2.0, 'Tech-Support': 6.0, 'Exec-Managerial': 15.0, 
        'Prof-Specialty': 12.0, 'Craft-Repair': 2.0, 'Other-Service': -5.0, 'Farming-Fishing': -4.0
    }
    salary_num += np.array([occ_salary_effect[o] for o in occupation])
    
    # Add workclass multiplier
    wc_salary_effect = {
        'Private': 0.0, 'Self-Emp': 4.0, 'State-Gov': 1.0, 'Federal-Gov': 5.0, 
        'Local-Gov': 1.0, 'Without-Pay': -10.0
    }
    salary_num += np.array([wc_salary_effect[wc] for wc in workclass])
    
    # Add noise
    salary_num += np.random.normal(0, 6, size=num_samples)
    salary_num = np.clip(salary_num, 5, None)
    
    # Partition into 5 groups
    conditions = [
        (salary_num <= 25),
        (salary_num > 25) & (salary_num <= 50),
        (salary_num > 50) & (salary_num <= 75),
        (salary_num > 75) & (salary_num <= 100),
        (salary_num > 100)
    ]
    choices = ['<=25K', '25K-50K', '50K-75K', '75K-100K', '>100K']
    salary_class = np.select(conditions, choices, default='<=25K')
    
    # Let's introduce some missing values (e.g., 2% missing in workclass and occupation)
    workclass = workclass.astype(object)
    occupation = occupation.astype(object)
    
    wc_nan_idx = np.random.choice(num_samples, size=int(num_samples * 0.02), replace=False)
    occ_nan_idx = np.random.choice(num_samples, size=int(num_samples * 0.025), replace=False)
    
    workclass[wc_nan_idx] = np.nan
    occupation[occ_nan_idx] = np.nan
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'workclass': workclass,
        'education': education,
        'experience': experience,
        'occupation': occupation,
        'hours_per_week': hours_per_week,
        'salary': salary_class
    })
    
    # Save to CSV
    filename = 'employee_salary_data.csv'
    df.to_csv(filename, index=False)
    print(f"Dataset generated successfully and saved as '{filename}'")
    print(f"Shape: {df.shape}")
    print(f"Class Distribution:\n{df['salary'].value_counts(normalize=True) * 100}")

if __name__ == '__main__':
    generate_synthetic_data()
