from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
from lecture_visu import df
import geopandas as gpd
import matplotlib.pyplot as plt

###################################
#        Pipeline Scikit-learn    #
###################################

# Génération de données fictives
#df, _ = make_blobs(n_samples=300, n_features=5, centers=4, random_state=42)

df_bis = df.drop(columns=['country'])


# Définition du pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Normalisation des données
    ('minmax', MinMaxScaler()),    # Mise à l'échelle entre 0 et 1
    ('pca', PCA(n_components=2)),  # Réduction de dimension avec PCA
    ('clustering', KMeans(n_clusters=3, random_state=42)) # Clustering
])

# Exécution du pipeline
transformed_data = pipeline.fit_transform(df_bis)

# Application de t-SNE après le pipeline
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(transformed_data)

# Récupération des labels de clustering
labels = pipeline.named_steps['clustering'].labels_
print(labels)
print(len(labels))

# Associer chaque pays à son cluster
df['cluster'] = labels

# Affichage des pays par cluster
for cluster in sorted(df['cluster'].unique()):
    print(f"Cluster {cluster}:")
    print(df[df['cluster'] == cluster]['country'].tolist())
    print("\n")
    

# Affichage des clusters sur une carte
world = gpd.read_file("data/ne_110m_admin_0_countries.shp")  # Remplace par le bon chemin

# Fusion des données avec les coordonnées géographiques
df_geo = world.merge(df, how="left", left_on="ADMIN", right_on="country")

# Création de la carte
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
df_geo.plot(column='cluster', cmap='viridis', linewidth=0.8, edgecolor='black', legend=True, ax=ax)
ax.set_title("Clusters des Pays", fontsize=15)
plt.show()