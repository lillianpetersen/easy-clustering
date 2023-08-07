# easy-clustering

Have you ever looked at your data and thought "how many clusters should I even input into K-means??". I got tired of looking for an elbow in the elbow plot, so I created these functions that perform agglomerative clustering on X, automatically decide a distance cutoff for defining clusters, and plots the clusters on both a dendrogram and a UMAP so you can inspect the quality of them yourself. 

Additionally, the plot\_umap function accepts up to 4 y variables, which it will plot on up to 4 subplots, so you can visualize important features of your data and how they coincide with the clusters. The colorbars automatically change for better visualization of binary vs continuous features.
