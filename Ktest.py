import pandas as pd
from kmodes.kmodes import KModes
from sklearn.cluster import KMeans
import numpy as np

# Charger le dataset dans un DataFrame
df = pd.read_csv('base_tri.csv')

# Sélectionner les colonnes numériques
num_cols = ['Quantity', 'Price']

# Sélectionner les colonnes catégorielles
cat_cols = ['Invoice', 'StockCode', 'Description', 'InvoiceDate', 'Customer ID', 'Country']

# Convertir les colonnes catégorielles en type 'str'
df[cat_cols] = df[cat_cols].astype(str)

# Créer une matrice de données pour les variables catégorielles
X_cat = df[cat_cols].values

# Créer un tableau de données pour les variables numériques
X_num = df[num_cols].values

# Créer un tableau combiné pour les deux types de variables
X = np.hstack((X_cat, X_num))

# Effectuer le clustering avec K-Prototype
kproto = KModes(n_clusters=3, init='Cao', verbose=2)
clusters_proto = kproto.fit_predict(X)

# Ajouter les labels de cluster au DataFrame
df['Cluster_KProto'] = clusters_proto

# Effectuer le clustering avec K-Means sur les variables numériques uniquement
kmeans = KMeans(n_clusters=3, random_state=42)
clusters_kmeans = kmeans.fit_predict(X_num)

# Ajouter les labels de cluster K-Means au DataFrame
df['Cluster_KMeans'] = clusters_kmeans

# Afficher le DataFrame avec les labels de cluster

df.to_csv('Clustered_df.csv',index=False)
print(df)
