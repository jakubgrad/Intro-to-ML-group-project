import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train_data = pd.read_csv("kaggle/train.csv")

missing_values = train_data.isnull().sum()
train_data = train_data.dropna()
numerical_features = train_data.select_dtypes(include=[np.number])

correlation_matrix = numerical_features.corr()
target_corr = correlation_matrix["log_pSat_Pa"].sort_values(ascending=False)

print("\nCorrelation of features with log_pSat_Pa:")
print(target_corr)

plt.figure(figsize=(12, 15))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

top_features = target_corr.index[1:11]
plt.figure(figsize=(14, 8))
sns.barplot(x=target_corr[top_features], y=top_features, palette='viridis')
plt.title("Top Correlations with log_pSat_Pa")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
