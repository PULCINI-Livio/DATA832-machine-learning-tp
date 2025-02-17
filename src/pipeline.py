from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

###################################
#        Pipeline Scikit-learn    #
###################################

# Génération de données fictives
data, _ = make_blobs(n_samples=300, n_features=5, centers=4, random_state=42)

# Définition du pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Normalisation des données
    ('minmax', MinMaxScaler()),    # Mise à l'échelle entre 0 et 1
    ('pca', PCA(n_components=2)),  # Réduction de dimension
    ('clustering', KMeans(n_clusters=4, random_state=42)) # Clustering
])

# Exécution du pipeline
pipeline.fit(data)

# Récupération des labels de clustering
labels = pipeline.named_steps['clustering'].labels_
print(labels)