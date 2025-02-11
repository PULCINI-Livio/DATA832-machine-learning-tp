import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data_df = pd.read_csv("data\Country-data.csv")
print(data_df)
print(data_df.mean()) # Moyenne
print(data_df.std()) # Variance
print(data_df.corr()) # Matrice de corrélation


for feature in data_df.columns[1:]:
    sorted_df = data_df.sort_values(by=feature, ascending=False)
    sorted_df = pd.concat([sorted_df.head(3), sorted_df.tail(3)])

    plt.figure(figsize=(20, 6))  # Plus large
    sns.barplot(data=sorted_df, x="country", y=feature)
    
    plt.xticks(rotation=45, ha="right", fontsize=8)  # Rotation et réduction de taille
    plt.title(f"Barplot of {feature} by Country")

    plt.tight_layout()  # Ajuste les marges
    plt.show()