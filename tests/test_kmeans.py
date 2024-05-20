import torch
import F2CGTLib as capi

n_samples = 5000
n_features = 50
n_clusters = 3
X = torch.randn((n_samples, n_features))
kmeans = capi.KMeans(n_clusters, n_features, "cosine")
kmeans.fit(X.cuda())
torch.cuda.synchronize()

labels = kmeans.predict(X.cuda())
print(labels)
