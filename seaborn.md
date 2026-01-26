# Matplotlib & Seaborn EDA Cheatsheet (with “What to Look For” + “What to Do Next”)

This assumes:

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid", context="talk")  # nice defaults
tips = sns.load_dataset("tips")           # example dataset
```

## 1. Histogram – sns.histplot / plt.hist

What it shows:
Distribution (shape, skew, multimodality) of a single numeric variable.

```python
sns.histplot(data=tips, x="total_bill", kde=True, bins=30)
plt.title("Distribution of Total Bill")
plt.show()
```
**Look for**:
- Skewness: long tail to the right/left.
- Multimodal shape: multiple peaks → mixture of subpopulations.
- Heavy tails / outliers: bars far away from the bulk.
- Zero inflation: big spike at zero.

**What to do with it:**
- Right-skewed → consider log / Box-Cox transform, or robust models.
- Clear sub-peaks → consider segmenting by another feature (hue, col) to see groups.
- Many outliers → consider winsorizing, capping, or robust metrics (median/quantile loss).
- Non-bell-shaped when a linear model assumes normal errors → use non-linear models or transform.

## 2. Boxplot – sns.boxplot

**What it shows:**

Distribution summary (median, quartiles, potential outliers) and group comparisons.

**Example:**
```python
sns.boxplot(data=tips, x="day", y="total_bill")
plt.title("Total Bill by Day")
plt.show()
```

**Look for:**
- Differences in medians across categories.
- Spread (IQR) within groups.
- Outliers (points beyond whiskers).
- Asymmetry: median closer to one side of the box.

**What to do with it:**
- Large median differences → candidate predictors for classification/regression.
- Large variability → consider stabilizing variance with transforms or modeling heteroscedasticity.
- Many outliers in some groups → investigate data quality or use robust statistics.
- Overlapping boxes → groups may not be easily separable by this single feature.

## 3. Violin Plot – sns.violinplot

**What it shows:**

Full distribution shape per category (like a smoothed histogram + boxplot).

**Example:**
```python
sns.violinplot(data=tips, x="day", y="total_bill", inner="quartile")
plt.title("Violin Plot of Total Bill by Day")
plt.show()
```

**Look for:**
- Multimodality within groups (bumps within each violin).

- Skewness and tail behavior.

- Comparisons of distribution shape, not just central tendency.

**What to do with it:**

- Multimodal within a category → maybe another latent grouping (e.g., lunch vs dinner) is important.

- Highly skewed groups → use non-parametric tests or transforms.

- Very different shapes between groups → feature likely interacts with category → consider interaction terms.

## 4. Empirical CDF – sns.ecdfplot

**What it shows:**

Cumulative distribution; excellent for comparing distributions across groups.

**Example:**
```python
sns.ecdfplot(data=tips, x="total_bill", hue="sex")
plt.title("ECDF of Total Bill by Sex")
plt.show()
```

**Look for:**

- One ECDF curve consistently above another → that group tends to have smaller values.

- Crossing curves → distributions differ in complex ways (not just shift).

**What to do with it:**

- Non-crossing curves → stochastic dominance (useful in risk / decision contexts).

- Crossing curves → consider quantile-based features or non-linear models.

- Use ECDFs to validate distributional assumptions or to choose thresholds (e.g., 90th percentile).

## 5. Scatter Plot – sns.scatterplot

**What it shows:**

Relationship between two numeric variables; structure, clusters, and outliers.

**Example:**

```python
sns.scatterplot(
    data=tips, x="total_bill", y="tip",
    hue="time", style="smoker", alpha=0.7
)
plt.title("Tip vs Total Bill")
plt.show()
```

**Look for:**

- Trend: positive, negative, or no relationship.

- Non-linearity: curves, plateaus, saturation.

- Clusters: distinct clouds of points → possible segments.

- Heteroscedasticity: variance increasing with x.

- Outliers: extreme points far from the main cloud.

**What to do with it:**

- Clear trend → good candidate for regression (possibly non-linear).

- Curved pattern → try polynomials, splines, tree-based models, or transforms.

- Heteroscedasticity → consider log-transform, weighted regression, or models that handle changing variance.

- Clusters → consider clustering or using cluster ID as a feature.

## 6. Scatter with Regression Line – sns.regplot / sns.lmplot

**What it shows:**

Linear (or low-order polynomial) fit + confidence intervals.

**Example (simple):**
```python
sns.regplot(
    data=tips, x="total_bill", y="tip",
    scatter_kws={"alpha": 0.6}
)
plt.title("Linear Fit: Tip vs Total Bill")
plt.show()
```

**Look for:**

- Slope sign & magnitude.

- Fit quality: points tightly around line vs widely scattered.

- Evidence of non-linearity (systematic curvature in residuals).

**What to do with it:**

- Strong linear pattern → simple linear regression may be adequate.

- Systematic curvature → add non-linear terms or switch model class.

- Wide scatter → relationship is weak; don’t over-rely on this variable.

## 7. Pairwise Relationships – sns.pairplot

**What it shows:**

All pairwise scatter plots + univariate distributions; very useful early in EDA.

**Example:**
```python
sns.pairplot(
    data=tips,
    vars=["total_bill", "tip", "size"],
    hue="sex",
    diag_kind="kde"
)
plt.suptitle("Pairwise Relationships (Tips Dataset)", y=1.02)
plt.show()
```

**Look for:**

- Pairs with clear patterns → strong candidate predictors.

- Collinearity: two features almost linearly dependent.

- Different patterns by hue (e.g., sex) → interactions.

**What to do with it:**

- Highly collinear features → consider dropping one, or using regularization (L1/L2).

- Strong patterns → prioritize these in feature selection.

- Group-specific patterns → include interaction terms or segment models by group.

# 8. Bivariate Distributions – sns.jointplot

**What it shows:**

Joint distribution + marginals; can be scatter, hexbin, KDE, or regression.

**Example (hexbin):**
```python
sns.jointplot(
    data=tips,
    x="total_bill",
    y="tip",
    kind="hex"   # try "reg", "kde", "scatter"
)
plt.show()
```

**Look for:**

- Regions of high density in 2D space.

- Non-linear patterns not obvious in simple scatter.

- Outlier regions with isolated bins.

**What to do with it:**

- Use dense regions to understand typical operating ranges.

- If density concentrated in a narrow band → relationship is strong → feature important.

- Outlier regions → investigate data quality or special cases (VIP customers, failures, etc.).

## 9. Correlation Heatmap – sns.heatmap on .corr()

**What it shows:**

Correlation matrix across numeric features.

**Example:**
```python
num_cols = tips.select_dtypes(include="number")
corr = num_cols.corr()

plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.show()
```

**Look for:**

- High correlations (|r| close to 1).

- Blocks of correlated features.

- Features almost independent (values near 0).

**What to do with it:**

- High correlations → risk of multicollinearity in linear models; consider:

    - removing some,
    
    - using PCA,
    
    - or switching to models less sensitive to collinearity (trees).

- Very low correlations with target → may still be useful in non-linear models; don’t drop blindly.

- Use correlation structure for feature engineering (ratios, differences).

## 10. Count Plot – sns.countplot

**What it shows:**

Counts (frequencies) of categories.

**Example:**
```python
sns.countplot(data=tips, x="day", hue="sex")
plt.title("Number of Observations by Day and Sex")
plt.show()
```

**Look for:**

- Class imbalance (one category dominating).

- Rare categories with very few samples.

**What to do with it:**

- Severe imbalance in target → consider:
    - resampling (over/under-sampling),
    - using class weights,
    - using appropriate metrics (F1, ROC-AUC).

- Rare categories → may need:
    - grouping into “Other”,
    - or target encoding with regularization to avoid overfitting.

## 11. Bar Plot – sns.barplot

**What it shows:**

Mean (or other estimator) of a numeric variable per category, with error bars.

**Example (for seaborn ≥ 0.12):**
```python
sns.barplot(
    data=tips,
    x="day", y="tip",
    estimator=np.mean,
    errorbar="sd"  # for older seaborn use: ci="sd"
)
plt.title("Average Tip by Day (with SD)")
plt.show()
```

**Look for:**

- Ranking of categories by mean.

- Overlap / separation via error bars (rough sense of uncertainty).

**What to do with it:**

- Large differences with little overlap → feature likely important.

- Overlapping intervals → effect might be small/uncertain; confirm with statistical tests.

- Use insights to collapse similar categories or focus on the strongest ones.

## 12. Line Plot – sns.lineplot

**What it shows:**

Trends over an ordered variable (time, index, or ordered category).

**Example (fake “time” index):**
```python
tips_sorted = tips.sort_values("total_bill").reset_index(drop=True)
tips_sorted["idx"] = np.arange(len(tips_sorted))

sns.lineplot(data=tips_sorted, x="idx", y="tip")
plt.title("Tip Over Sorted Bills (as a pseudo-time)")
plt.show()
```

(For real time series, use a datetime index instead of idx.)

**Look for:**

- Trends (upward, downward).

- Seasonality / cycles.

- Structural breaks or sudden level shifts.

**What to do with it:**

- Clear trend/seasonality → use time-series models (ARIMA, Prophet, etc.) or add time features (month, day, hour).

- Structural breaks → model before/after separately or include event indicators.

- Non-stationarity → difference, detrend, or use models that handle non-stationarity.

## 13. Faceted Distribution – sns.displot / FacetGrid

**What it shows:**

How the distribution of a variable changes across subgroups (columns/rows).

**Example:**
```python
sns.displot(
    data=tips,
    x="total_bill",
    col="time",      # separate plots for Lunch/Dinner
    hue="sex",
    kde=True,
    bins=20
)
plt.suptitle("Total Bill Distribution by Time and Sex", y=1.05)
plt.show()
```

**Look for:**

- Shape differences across subgroups.

- Group-specific skewness, variance, or multimodality.

**What to do with it:**

- Large differences → consider segmenting the model (separate models per subgroup) or adding interaction terms.

- Very similar distributions → subgroup variable may be less informative for that feature.

## 14. Categorical Scatter – sns.stripplot / sns.swarmplot

**What it shows:**

Raw data points of a numeric variable across categories.

**Example:**
```python
sns.swarmplot(data=tips, x="day", y="tip", hue="sex", dodge=True)
plt.title("Tips by Day (Point-level)")
plt.show()
```

*Look for:*

- Overplotting / clusters around certain values.

- Gaps in support (values that never occur).

- Outliers relative to neighbors.

**What to do with it:**

- Gaps → possible discrete effects (price thresholds, business rules).

- Clusters → candidate binning thresholds or feature engineering (e.g., “small tip”, “medium tip”).

- Outliers → verify data, or cap / transform if they hurt model performance.

## 15. Missingness Heatmap – sns.heatmap on .isna()

**What it shows:**

Pattern of missing values across rows and columns.

(Using a modified dataset with some missingness is best, but this shows the pattern.)

**Example:**
```python
plt.figure(figsize=(8, 4))
sns.heatmap(tips.isna(), cbar=False)
plt.title("Missing Values Pattern")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.show()
```

**Look for:**

- Columns with many missing values.

- Patterns: missingness clustered in certain rows, or combinations of columns.
 
- Blocks of missingness associated with certain subgroups (if you facet).

**What to do with it:**

- Very high missingness in a feature → consider dropping or using specialized models (e.g., tree-based models can handle some missing).

- Structured missingness (MNAR, not at random) → be careful with naïve imputation; consider modeling missingness itself as a feature.
 
- Use appropriate imputation: mean/median for numeric, mode/“Unknown” for categories, or model-based imputation.

## 16. Clustered Heatmap – sns.clustermap

**What it shows:**

Hierarchically clustered heatmap, often used on correlation or distance matrices to reveal groups of similar features or samples.

**Example (feature clustering):**
```python
num_cols = tips.select_dtypes(include="number")
corr = num_cols.corr()

sns.clustermap(
    corr,
    cmap="coolwarm",
    center=0,
    annot=True
)
plt.suptitle("Clustered Correlation Heatmap", y=1.02)
plt.show()
```

**Look for:**

- Clusters of features that correlate strongly with each other.

- Features forming isolated branches → more independent.

**What to do with it:**

- Within clusters, consider:

    - dimensionality reduction (PCA over cluster),
    
    - picking a representative feature,
 
    - or using regularization to handle redundancy.
 
- Between-cluster independence → good candidates to combine in non-linear models (more diverse information).

## 17. Basic Matplotlib Subplots – plt.subplots

**What it shows:**

Multiple related views side-by-side; helps compare transformations, models, or subsets.

**Example:**
```python
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

sns.histplot(data=tips, x="total_bill", ax=axes[0], bins=30)
axes[0].set_title("Raw Total Bill")

sns.histplot(
    x=np.log1p(tips["total_bill"]),
    ax=axes[1], bins=30
)
axes[1].set_title("Log1p(Total Bill)")

plt.tight_layout()
plt.show()
```

**Look for:**

- How transformations change shape (skew → more symmetric).

- Whether transformation makes patterns more linear / homoscedastic.

**What to do with it:**

- If log/transform clearly improves shape → consider using transformed feature in modeling.

- Use subplots to justify preprocessing choices in reports or to teammates.