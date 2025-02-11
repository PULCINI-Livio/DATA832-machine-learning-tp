import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data_df = pd.read_csv("data\Country-data.csv")
print(data_df.head())
#print(data_df)
"""print(data_df.mean()) # Moyenne
print(data_df.std()) # Variance
print(data_df.corr()) # Matrice de corrélation
"""

sns.barplot(data_df, x="country", y="child_mort")
plt.show()