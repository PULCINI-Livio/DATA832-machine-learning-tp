from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from lecture_visu import df

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
world = gpd.read_file("data/ne_110m_admin_0_countries.shp")  

# Fusion des données avec les coordonnées géographiques
df_geo = world.merge(df, how="left", left_on="ADMIN", right_on="country")
print(world["ADMIN"].unique())
df_geo["cluster"] = df_geo["cluster"].fillna(-1)  # -1 peut représenter un "cluster inconnu"

# Définir une palette de couleurs distinctes (assurez-vous d'en avoir assez pour tous les clusters)
clusters = df_geo["cluster"].unique()
clusters = sorted(clusters)  # Trier pour correspondre aux couleurs
nb_clusters = len(clusters)

# Générer une palette avec des couleurs distinctes
color_list = plt.cm.get_cmap("tab10", nb_clusters).colors  # 'tab10' offre 10 couleurs distinctes
color_dict = {
    -1: "#9c9fa1",  # Gris pour les pays sans cluster
    0: "#72E953",   # Rouge
    1: "#5372E9",   # Vert
    2: "#E95372",   # Bleu
}

# Créer un colormap basé sur ces couleurs
cmap = mcolors.ListedColormap([color_dict[cluster] for cluster in clusters])

# Création de la carte
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
df_geo.plot(column="cluster", cmap=cmap, linewidth=0.8, edgecolor="black", legend=False, ax=ax)

# Création d'une légende personnalisée
from matplotlib.patches import Patch
legend_labels = [Patch(facecolor=color_dict[cluster], edgecolor='black', label=f"Cluster {cluster}") for cluster in clusters]
ax.legend(handles=legend_labels, title="Clusters", loc="lower left", fontsize=12)

# Titre
ax.set_title("Clusters des Pays", fontsize=15)

# Affichage
plt.show()