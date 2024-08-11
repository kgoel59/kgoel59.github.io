---

layout: post  
title: "Big Data Week 3"  
date: 2024-08-5 15:09:00  
description: "Methods for Data Exploration"  
tags: projects learning uow
categories: learning  
giscus_comments: true  
featured: false  

---

## Introduction

Data exploration is a critical stage in the data analytics lifecycle. It involves understanding and preparing data to facilitate effective analysis and model building. Here’s a streamlined overview of the key stages and techniques involved.

### Data Analytics Lifecycle

**1. Discovery**
   - **Learn Business Domain**: Understand the industry and context.
   - **Interview Sponsor & Identify Stakeholders**: Engage with key individuals to gather insights.
   - **Define Resources & Goals**: Establish objectives and available resources.
   - **Identify Potential Data Sources**: Locate relevant data sources.
   - **Frame the Problem & Develop Initial Hypotheses**: Formulate hypotheses, including Null Hypothesis (H0) and Alternative Hypothesis (HA or H1).

**2. Data Preparation**
   - **Prepare Sandbox**: Set up an environment for data preparation.
   - **Perform ETLT (Extract, Transform, Load, Transform)**: Process data for analysis.
   - **Understand Data Details**: Examine the data's structure and quality.
   - **Data Conditioning**: Address issues like missing values and outliers.
   - **Format Data**: Prepare data for analysis.
   - **Visualize Data**: Use plots to explore data patterns.

**3. Model Planning**
   - **Select Variables**: Based on relationships (e.g., correlation matrix) and domain knowledge.
   - **Identify Candidate Models**: Refer to hypotheses, translate into machine learning models, review literature, and document assumptions.

**4. Model Building**
   - **Create Datasets**: Prepare training, validation, and testing datasets.
   - **Train and Test Models**: Evaluate model performance.

**5. Communicating Results**
   - **Compare Results**: Assess against criteria.
   - **Articulate Findings**: Clearly present results.
   - **Discuss Limitations & Recommendations**: Provide insights on limitations and suggest improvements.

**6. Operationalize**
   - **Deliverables**: Finalize and deliver the project.
   - **Pilot Project**: Test the model in a real-world scenario.
   - **Performance & Constraints**: Monitor and address any constraints.
   - **Training**: Educate new users as needed.

### Key Objectives of Data Exploration

- **Understand Data Structure**: Analyze data types, distributions (population vs. sampling), and summary statistics.
- **Assess Data Quality**: Identify missing values, outliers, and duplicates.
- **Identify Patterns and Relationships**: Use correlation to find relationships between variables.
- **Formulate Hypotheses**: Develop hypotheses based on data patterns.

### Statistical Tools and Techniques

- **Basic Statistics**: Mean, median, variance, standard deviation, range, interquartile range (IQR).
- **Correlation and Covariance**: Measure relationships between variables (cor(x,y), cov(x,y)).
- **Hypothesis Testing**: Test hypotheses using p-values and confidence intervals.
  - **Two-Sample t-Test**: Compare means of two populations.
  - **Welch’s t-Test**: Used when equal variance assumption is not justified.
  - **Wilcoxon Rank-Sum Test**: Non-parametric test for non-normal distributions.

### Error Types in Hypothesis Testing

- **Type I Error (α)**: Incorrectly rejecting the null hypothesis when it is true.
- **Type II Error (β)**: Failing to reject the null hypothesis when the alternative hypothesis is true. Reduce by increasing sample size.

### Conclusion

Effective data exploration is essential for successful data analysis and model building. By understanding data structure, assessing quality, and using appropriate statistical tools, you can make informed decisions and derive actionable insights.

[Lecture 3](/assets/pdf/bigdata/w3_DataPrep.pdf)