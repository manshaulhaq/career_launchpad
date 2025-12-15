import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Data
df = sns.load_dataset('titanic')

# 2. Initial Inspection & Handling Missing Values

print("--- Initial Data Information (First 5 Rows) ---")
print(df.head())
print("\nMissing Values Count:")
print(df.isnull().sum())

# Impute 'Age' with the median
df['age'].fillna(df['age'].median(), inplace=True)
# Impute 'Embarked' with the mode (most frequent value)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
# Drop 'deck' column due to excessive missing values (~77%)
df.drop('deck', axis=1, inplace=True)

# 3. Feature Engineering/Encoding (for correlation/analysis)
# Convert 'sex' to numerical
df['sex_encoded'] = df['sex'].map({'male': 0, 'female': 1})
# Convert 'embarked' to numerical (using simple label encoding for correlation)
df['embarked_encoded'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# 4. Generate Visualizations and Correlation Heatmap

# Define the numerical columns subset for correlation (including encoded features)
numerical_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_encoded', 'embarked_encoded']
corr_df = df[numerical_cols]

# A. Univariate Analysis: Survival Count (Target Variable)
plt.figure(figsize=(6, 4))
sns.countplot(x='survived', data=df)
plt.title('Survival Count (0=No, 1=Yes)')
plt.show()

# B. Bivariate Analysis: Survival Rate by Gender (Strongest Predictor)
plt.figure(figsize=(6, 4))
sns.barplot(x='sex', y='survived', data=df)
plt.title('Survival Rate by Gender')
plt.show()

# C. Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = corr_df.corr()
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=.5,
    cbar_kws={'label': 'Pearson Correlation Coefficient'}
)
plt.title('Correlation Matrix of Titanic Features')
plt.tight_layout()
plt.show()
