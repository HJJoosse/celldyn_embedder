This project is aimed towards embedding Abbott Celldyn Sapphire data in the UMC Utrecht, an academic hospital in Utrecht, The Netherlands.

This work is still in early stages of research, and it is therefore not likely to be useful for plug-and-play. 

Plan de campagne:
Compare different embedding methods:
* identify good transformations for variables
* identify good imputation strategy
* intra: with [flameplot](https://erdogant.github.io/flameplot/pages/html/index.html), expand flameplot with groundtruth
* inter: compared to the ground truth in terms of long/short distances
    * Clustering metrics
    * kNN-overlap
    * correlation of distances
    * self-consistency

Nice to have:
* library to evaluate embedders with sklearn-api
* heuristic for creating a method that maintains distances where it matters

