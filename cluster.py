import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

import plot_umap

def heirarchical_cluster(X, title, savetitle, figdir, embedding=None, cutoff_pct=99.5, method='ward', metric='euclidean'):
	'''
	inputs:
		X: numpy array, shape (N x n_features)
		title: title for plots
		savetite: name to save plots under
		figdir: directory to save plots in
		embedding: optional UMAP embedding to plot the points on. 
					If no embedding is given, it will calculate the UMAP embedding
		cutoff_pct: the distance cutoff (as a percent of the max distance) 
		method: see https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
		metric: see https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
	output:
		labels: numpy array of length N assigning each point to a cluster
	'''
	# do PCA on X and limit to number of PCs that explain at least 95% of the variance
	pca = PCA()
	pca.fit(X)
	n_pc = np.where(np.cumsum(pca.explained_variance_ratio_)>0.95)[0][0]
	print('Using', n_pc, 'PCs')
	pca = PCA(n_components=n_pc)
	X_pc = pca.fit_transform(X)

	# choose distance cutoff for clusters
	plt.clf()
	linkage_data = linkage(X_pc, method=method, metric=metric)
	cutoff = np.percentile(np.amin(linkage_data,axis=1),cutoff_pct)
	print('Using distance cutoff:', cutoff)
	# save dendrogram
	dendrogram(linkage_data, no_labels=True, color_threshold=cutoff)
	plt.axhline(y=cutoff, color='k', linestyle='-')
	plottitle='Dendrogram: '+title
	plt.title(plottitle)
	plotsavetitle='_'.join(['dendrogram',savetitle])
	plt.savefig(figdir+'/'+plotsavetitle+'.pdf')

	# plot on a umap
	agg = AgglomerativeClustering(distance_threshold=cutoff, n_clusters=None)
	labels = agg.fit_predict(X_pc)
	print('Found', np.amax(labels)+1,'clusters')
	plotsavetitle='_'.join(['heirarchical_cluster',savetitle])
	plottitle='Heirarchical Clustering: '+title
	plot_umap.plot_umap(X, y=labels, title=plottitle, figdir=figdir, savetitle=plotsavetitle, embedding=embedding)

	return labels

