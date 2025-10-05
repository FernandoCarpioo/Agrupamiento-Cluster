import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering

path = kagglehub.dataset_download("shwetabh123/mall-customers")
file_path = os.path.join(path, "Mall_Customers.csv")

datos = pd.read_csv(file_path)

X = datos[["Annual Income (k$)", "Spending Score (1-100)"]].values

SC = SpectralClustering(n_clusters=5)
y_pred = SC.fit_predict(X)

#Gráfica 
plt.figure(figsize=(6,5))
plt.scatter(X[:,0], X[:,1], c=y_pred, cmap="viridis", s=50)
plt.xlabel("Ingreso Anual (miles $)")
plt.ylabel("Puntaje de Gasto (1-100)")
plt.title("Spectral Clustering - Ingreso vs Gasto")
plt.show()
#-------------------------------------------------------
import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering

path = kagglehub.dataset_download("shwetabh123/mall-customers")
file_path = os.path.join(path, "Mall_Customers.csv")

datos = pd.read_csv(file_path)

X = datos[["Age", "Spending Score (1-100)"]].values

SC = SpectralClustering(n_clusters=5)
y_pred = SC.fit_predict(X)

#Gráfica 
plt.figure(figsize=(6,5))
plt.scatter(X[:,0], X[:,1], c=y_pred, cmap="viridis", s=50)
plt.xlabel("Edad")
plt.ylabel("Puntaje de Gasto (1-100)")
plt.title("Spectral Clustering - Edad vs Gasto")
plt.show()

#-----------------------------------------
import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering

path = kagglehub.dataset_download("shwetabh123/mall-customers")
file_path = os.path.join(path, "Mall_Customers.csv")

datos = pd.read_csv(file_path)

X = datos[["Age", "Annual Income (k$)"]].values

SC = SpectralClustering(n_clusters=5)
y_pred = SC.fit_predict(X)

#Gráfica 
plt.figure(figsize=(6,5))
plt.scatter(X[:,0], X[:,1], c=y_pred, cmap="viridis", s=50)
plt.xlabel("Edad")
plt.ylabel("Ingreso Anual (miles $)")
plt.title("Spectral Clustering - Edad vs Ingreso")
plt.show()