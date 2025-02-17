from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
from lecture_visu import df

###################################
#        Pipeline Scikit-learn    #
###################################

# Génération de données fictives
#df, _ = make_blobs(n_samples=300, n_features=5, centers=4, random_state=42)

df = df.drop(columns=['country'])

# Définition du pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Normalisation des données
    ('minmax', MinMaxScaler()),    # Mise à l'échelle entre 0 et 1
    ('tsne', TSNE(n_components=2, random_state=42)),  # Réduction de dimension avec t-SNE
    ('clustering', KMeans(n_clusters=3, random_state=42)) # Clustering
])

# Exécution du pipeline
transformed_data = pipeline.fit_transform(df)

# Récupération des labels de clustering
labels = pipeline.named_steps['clustering'].labels_
print(labels)
print(len(labels))