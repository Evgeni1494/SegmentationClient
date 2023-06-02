import pandas as pd
# from kmodes.kmodes import KModes
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


df=pd.read_csv('df_pr_kmean')

# Sélection des colonnes pour le clustering
columns = ['Total', 'gros_client', 'frequence_achat']
df_selected = df[columns]

# Application de l'algorithme K-means
kmeans = KMeans(n_clusters=7, random_state=0)
kmeans.fit(df_selected)

# Ajout des étiquettes de clusters au DataFrame
df['cluster'] = kmeans.labels_

# Affichage du résultat
print(df)