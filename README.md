# Country-Segmentation
 A case study using a socio-economic indicators dataset (167 countries) to cluster countries into homogeneous segments, then translating those segments into concrete aid-allocation and policy recommendations.

## Problem statement

Global development organizations, such as Global Progress Analytics, need to effectively and efficiently allocate development funds to maximize their impact. A one-size-fits-all approach is ineffective due to the wide diversity in socio-economic structures and inherent challenges across nations.

The core problem is to use unsupervised learning techniques on socio-economic indicator data to objectively discover and segment countries into homogeneous clusters. This segmentation allows policy analysts to move beyond general classifications and design targeted, group-specific global development programs and recommendations for each identified segment.

## The data

Nine indicators per country: `child_mort`, `exports`, `health`, `imports`, `income`, `inflation`, `life_expec`, `total_fer`, `gdpp`. No missing values, no type issues — the dataset is clean going in.

## Preprocessing

Outliers were checked per column (IQR method) but deliberately not removed — countries with unusually high `gdpp` or `income` aren't data errors, they're the signal the whole exercise is trying to capture. Removing them would bias the clusters toward the middle of the distribution and defeat the point.

`gdpp` and `income` are both heavily right-skewed, so both were log-transformed (`log1p`) before scaling. All nine features were then standardized with `StandardScaler`, since the clustering algorithms and PCA that follow are distance-based and would otherwise be dominated by whichever feature happens to have the largest raw range.

## Dimensionality reduction

PCA was run on the scaled features. A scree plot showed diminishing returns past the third component, so three components were kept, retaining a large majority of total variance while cutting the feature space from nine dimensions to three — mainly to make clustering more stable and the results plottable.

Loadings on PC1 were dominated by income, gdpp, and child mortality (in opposite directions), which is roughly "overall development level." PC2 and PC3 picked up more specific contrasts, e.g. trade intensity and inflation volatility, that don't collapse neatly onto the development axis.

## Clustering approaches tried

Three methods, run on the PCA-reduced data:

**K-Means** — Elbow method and silhouette score both pointed to k=3. Produced three balanced, well-separated clusters that mapped cleanly onto "underdeveloped / developing / developed."

**Hierarchical (Ward linkage)** — Dendrogram also supported a 3-cluster cut, but the shape came out differently: it merged the developing and developed groups into one large cluster, and split off a small third cluster of extreme trade-dependent wealthy outliers (Luxembourg, Singapore, Malta) rather than following the income/mortality axis K-Means used.

**DBSCAN** — Tuned eps via a k-distance plot (eps=0.85, min_samples=5). Found 3 clusters plus 33 noise points. The noise points weren't random — they clustered into recognizable categories: oil-rich Gulf states, small island/micro-economies, and countries with acute conflict or instability (Haiti, Yemen, Liberia). That's arguably useful information on its own, but it doesn't give every country a segment, which matters for a project whose output is meant to assign each country to a targeted program.

## Model comparison and choice

All three were scored with silhouette score, Davies-Bouldin index, and Calinski-Harabasz index on the same PCA-reduced data (DBSCAN scored only on non-noise points, which is a slightly easier comparison and worth keeping in mind).

K-Means came out ahead on Calinski-Harabasz and was chosen as the final model — not purely on the metric, but because it's the only one of the three that gives a clean, complete, roughly balanced partition of all 167 countries. That completeness matters more here than in a typical clustering task, since the deliverable is a policy assignment for every country, not just the well-separated ones.

## The three segments

**Underdeveloped (43 countries)** — e.g. Afghanistan, Angola, Burkina Faso, Chad. Highest child mortality and fertility, lowest income and gdpp, minimal health spending and exports. Recommended focus: maternal/child healthcare, family planning access, debt relief and foreign aid, and basic infrastructure (water, sanitation, electricity), largely through NGO partnerships given weak existing institutions.

**Developing (64 countries)** — e.g. Bangladesh, Bolivia, Argentina, Albania. Middle-range income and gdpp, moderate child mortality, some inflation volatility. This is the largest segment and the one with the most room to move — recommended focus is export diversification, higher health spend as a share of GDP, monetary stability to attract investment, and vocational/higher education.

**Developed (60 countries)** — e.g. Australia, Austria, Belgium. Highest income and gdpp, lowest child mortality, strong trade activity. Recommendations here shift from aid recipient to aid contributor: sustaining healthcare spend for an aging population, leading climate financing, and using trade/FDI to support the other two segments.

These groupings are directional guidance based on aggregate country-level averages, not a substitute for country-specific due diligence — a country sitting near a cluster boundary (Luxembourg, Singapore) is a reminder that the boundaries are statistical, not natural categories.

## Stack

pandas, numpy, scikit-learn (KMeans, AgglomerativeClustering, DBSCAN, PCA), scipy (dendrogram), matplotlib, seaborn

## Files

- `Country_segmentation.ipynb` — full notebook: EDA, outlier/skew handling, PCA, three clustering methods with parameter tuning, cross-method comparison, segment profiling, and policy recommendations
