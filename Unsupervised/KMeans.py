from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
import numpy as np
from time import time

data, labels = load_digits(return_X_y=True)
data = scale(data)
n_samples, n_features = data.shape
n_digits = len(np.unique(labels))

print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\t\tAMI\t\tsilhouette')
print('----\t\t----\t-------\t----\t-----\t------\t---\t\t---\t\t----------')


def bench_k_means(estimator, name):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=300)))


bench_k_means(KMeans(init='k-means++', n_clusters=n_digits,
                     n_init=10), name='k-means++')
bench_k_means(KMeans(init='random', n_clusters=n_digits,
                     n_init=10), name='random')
pca = PCA(n_components=n_digits).fit(data)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits,
                     n_init=1), name='PCA-based')
