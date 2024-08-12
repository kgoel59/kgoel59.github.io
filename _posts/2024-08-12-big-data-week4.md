---

layout: post  
title: "CSCI946 - Big Data Analytics Week 3"  
date: 2024-08-12 15:09:00  
description: "Clustering"  
tags: projects learning uow
categories: learning  
giscus_comments: true  
featured: false  

---


### Big Data Analytics Week 3: Clustering and Hypothesis Testing

In Week 3 of CSCI946 - Big Data Analytics, we delved into two essential topics: **Hypothesis Testing** and **Clustering Analysis**. These are fundamental techniques in data science and analytics, providing the groundwork for data-driven decision-making and pattern recognition in large datasets.

#### Hypothesis Testing

Hypothesis testing is a statistical method that allows us to make inferences about a population based on sample data. The process involves testing an assumption, known as the **null hypothesis (H0)**, against an alternative hypothesis (HA). Depending on the nature of the data and the specific hypotheses being tested, different types of tests can be applied.

1. **Student’s t-test**: Used when comparing the means of two populations that are normally distributed with similar variances.
   
2. **Welch’s t-test**: Similar to the Student’s t-test but used when the populations have different variances.
   
3. **Wilcoxon Rank-Sum Test**: A non-parametric test used when the populations are not normally distributed.
   
4. **ANOVA (Analysis of Variance)**: Applied when comparing the means of more than two populations. ANOVA checks if at least one pair of population means differ.

   - **One-Way ANOVA**: Involves one independent factor.
   - **Two-Way ANOVA**: Involves two independent factors.

   **Limitations of ANOVA**:
   - Assumes data is normally distributed.
   - Sensitive to outliers.
   - Does not specify which pairs of means are different, requiring further analysis with tests like HSD (Honest Significant Difference).

#### Clustering Analysis

Clustering is a method of unsupervised learning that groups a set of objects in such a way that objects in the same group (or cluster) are more similar to each other than to those in other groups. We covered three major clustering algorithms:

1. **K-Means Clustering**:
   - A hard clustering algorithm where each object is a point in an n-dimensional space.
   - Objects are clustered based on their proximity to k randomly initialized centroids.
   - The algorithm iteratively assigns points to the nearest centroid and updates the centroids until convergence.
   - Commonly used in image processing, medical diagnosis, and data compression.

   **Challenges**:
   - Sensitive to noise and outliers.
   - May result in empty clusters or vary in density.
   - Dependent on the initial positioning of centroids, which may require multiple runs.

2. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**:
   - Locates regions of high density that are separated by regions of low density.
   - Classifies points as core points, border points, or noise points based on a density threshold (MinPts) and a radius (Eps).
   - Resistant to noise and outliers, and capable of handling clusters of varying shapes and sizes.
   
   **Limitations**:
   - Struggles with varying densities.
   - Not well-suited for sparse or high-dimensional data without additional techniques.

3. **Self-Organizing Maps (SOM)**:
   - A type of neural network used for clustering and visualization.
   - Projects high-dimensional data onto a lower-dimensional space while preserving the topology.
   - Useful for data visualization and exploration, especially when dealing with missing values or new data.

   **When Not to Use SOM**:
   - Data is very sparse.
   - The resolution of the map is limited.
   - Lack of multi-core computational resources.

### Conclusion

This week's focus on hypothesis testing and clustering provides a foundation for understanding and applying these methods in big data analytics. Each technique has its strengths and limitations, and the choice of method depends on the nature of the data and the specific problem at hand. Whether testing hypotheses or uncovering hidden patterns in data, these tools are essential for effective data analysis.


[Lecture 3](/assets/pdf/bigdata/w4-Clustering.pdf)