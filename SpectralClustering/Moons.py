import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering

X, y = make_moons(n_samples=300, noise=0.1, random_state=42)

plt.scatter(X[:,0], X[:,1], cmap='gray')
plt.title("Moons sin agrupar")
plt.show()

SC = SpectralClustering(n_clusters=3)
SC.fit(X)

plt.scatter(X[:,0], X[:,1], c=SC.labels_, cmap='viridis')
plt.title("Spectral Clustering - Moons")
plt.show()