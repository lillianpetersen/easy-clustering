import numpy as np
import matplotlib.pyplot as plt

import umap

def plot_umap(X, title, savetitle, figdir, y=None, y_labels=None, embedding=None):
	'''
	inputs:
		X: numpy array, shape (N x n_features)
		title: title for plots
		savetite: name to save plots under
		figdir: directory to save plots in
		y: optional, numpy array, shape (N x d).
			If y is None, all the points will have the same color
			If d = 1, the points will be colored by y on one plot
			If d > 1, the points will be colored by each dimension of y in separate subplots
			(current support is only for d<=4)
		y_labels: optional, list of strings, length d
			labels each subplot by its corresponding y_label
		embedding: optional UMAP embedding to plot the points on. 
					If no embedding is given, it will calculate the UMAP embedding
	'''
	if embedding is None:
		reducer = umap.UMAP()
		embedding = reducer.fit_transform(X)
	
	if len(embedding)>3000: s=0.5
	elif len(embedding)>1000: s=1
	elif len(embedding)>500: s=4
	else: s=8
	plt.clf()
	if y is None:
		plt.figure(figsize=(10,7))
		plt.scatter(embedding[:, 0], embedding[:, 1],
			marker='.', s=s)
		plt.gca().set_aspect('equal', 'datalim')
		plt.title(title, fontsize=12)
		plt.savefig(figdir+'/'+savetitle+'.png', dpi=300)
	else:
		if len(y.shape)==1 or y.shape[1]==1: # one y value per sample
			plt.figure(figsize=(6,5))
			if len(np.unique(y))>2:
				plt.scatter(embedding[:, 0], embedding[:, 1],
					c=y, cmap='turbo', vmin=np.percentile(y,1), vmax=np.percentile(y,99),
					marker='.', s=2*s)
			else:
				plt.scatter(embedding[:, 0], embedding[:, 1],
					c=y, cmap='coolwarm', 
					vmin=np.amin(y)-0.1, vmax=np.amax(y)+0.1,
					marker='.', s=2*s)
			plt.colorbar()
			plt.title(title, fontsize=12)
			plt.savefig(figdir+'/'+savetitle+'.png', dpi=300)
		elif y.shape[1]==2:
			fig, axs = plt.subplots(1,2, sharex=True, sharey=True, figsize=(12,5))
			axdict = {0:axs[0], 1:axs[1]}
			for i in range(y.shape[1]):
				if len(np.unique(y[:,i]))>2:
					sct = axdict[i].scatter(embedding[:, 0], embedding[:, 1],
						c=y[:,i], cmap='turbo', 
						vmin=np.percentile(y[:,i],1), vmax=np.percentile(y[:,i],99),
						marker='.', s=4*s)
				else:
					sct = axdict[i].scatter(embedding[:, 0], embedding[:, 1],
						c=y[:,i], cmap='coolwarm', 
						vmin=np.amin(y[:,i])-0.1, vmax=np.amax(y[:,i])+0.1,
						marker='.', s=4*s)
				axdict[i].set_title(y_labels[i])
				cbar = plt.colorbar(sct, ax=axdict[i])
				cbar.set_ticks([])
			fig.suptitle(title, fontsize=16)
			fig.tight_layout()
			plt.subplots_adjust(wspace=None, hspace=None)
			plt.savefig(figdir+'/'+savetitle+'.png', dpi=300)
		elif y.shape[1]==3 or y.shape[1]==4:
			fig, axs = plt.subplots(2,2, sharex=True, sharey=True, figsize=(12,10))
			axdict = {0:axs[0,0], 1:axs[0,1], 2:axs[1,0], 3:axs[1,1]}
			for i in range(y.shape[1]):
				if len(np.unique(y[:,i]))>2:
					sct = axdict[i].scatter(embedding[:, 0], embedding[:, 1],
						c=y[:,i], cmap='turbo', 
						vmin=np.percentile(y[:,i],1), vmax=np.percentile(y[:,i],99),
						marker='.', s=4*s)
				else:
					sct = axdict[i].scatter(embedding[:, 0], embedding[:, 1],
						c=y[:,i], cmap='coolwarm', 
						vmin=np.amin(y[:,i])-0.1, vmax=np.amax(y[:,i])+0.1,
						marker='.', s=4*s)
				axdict[i].set_title(y_labels[i])
				cbar = plt.colorbar(sct, ax=axdict[i])
				cbar.set_ticks([])
			fig.suptitle(title, fontsize=16)
			fig.tight_layout()
			plt.subplots_adjust(wspace=None, hspace=None)
			plt.savefig(figdir+'/'+savetitle+'.png', dpi=300)
		else:
			raise Exception("Current support is only for d<=4")
		plt.close()

def heirarchical_cluster(X, title, savetitle, figdir, embedding=None):
	'''
	inputs:
		X: numpy array, shape (N x n_features)
		title: title for plots
		savetite: name to save plots under
		figdir: directory to save plots in
		embedding: optional UMAP embedding to plot the points on. 
					If no embedding is given, it will calculate the UMAP embedding
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

	# choose distance cutoff for clusters (here, the 99.5th percentile of all the distances)
	cutoff_pct = 99.5
	plt.clf()
	linkage_data = linkage(X_pc, method='ward', metric='euclidean')
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
	plot_umap(X, y=labels, title=plottitle, figdir=figdir, savetitle=plotsavetitle, embedding=embedding)

	return labels

