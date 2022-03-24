# Clinical embedding bake-off

Comparing:
* t-SNE
* UMAP
* PHATE
* dbMap
* DenseMap
* TriMap
* PacMap
* PCA
* MDS
* Isomap


Measuring:
* kNN accuracy
* correlation between pair-wise distances; discriminate between short and long distances
* Jaccard index with k-Medoids
* Topology preservation
* ?


Based: 
* simulated data: varying (N,M), varying predictors/noise etc.
* Hematology data: 3+ million samples
* RNA expression data, TCGA: discrimination between cancers (long distance) and discrimination between specific cancer histologies (short distance).
