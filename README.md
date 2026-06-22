# Iris Species Data Exploration and Classification

## 🎯 Introduction & Problem Statement
This project demonstrates a foundational Data Science workflow using the famous Iris dataset. The primary objective is to analyze the biological characteristics of three iris flower species, explore hidden feature relationships through data visualization, and structure the data for machine learning classification. 

By building clear visual profiles of the features, we can understand which physical attributes (sepal/petal lengths and widths) are the most deterministic in predicting the correct flower species.

---

## 📁 Dataset Understanding & Preparation
The Iris dataset is structurally well-balanced and inherently clean, consisting of **150 samples** across **4 numerical features** and **1 categorical target variable** (`species`).

* **Features:** `sepal_length`, `sepal_width`, `petal_length`, `petal_width`
* **Target Categories:** *Setosa*, *Versicolor*, *Virginica*

### Core Dependencies & Data Setup
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Apply aesthetic baseline grid
sns.set_theme(style="whitegrid")

# Load built-in iris repository
iris = sns.load_dataset('iris')

# Inspect dimensions and descriptive distributions
print(f"Dataset Shape: {iris.shape}")
print("\nStatistical Overview:")
print(iris.describe())

# Isolate feature matrix (X) from class labels (y)
X = iris.drop('species', axis=1)
y = iris['species']
