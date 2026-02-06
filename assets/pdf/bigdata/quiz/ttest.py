import scipy.stats as stats

# Sample data
data1 = [85.28, 85.17, 85.94, 84.27, 83.27, 83.25, 84.35, 83.71]
data2 = [87.03, 83.52, 84.59, 83.79, 83.92, 85.91, 84.21, 84.42]

# Set the significance level
alpha = 0.05

# Perform t-test
t_statistic, p_value = stats.ttest_ind(data1, data2)

# Print the results
print(f"T-statistic: {round(t_statistic,4)}")
print(f"P-value: {round(p_value,4)}")


# Decision based on alpha
if p_value < alpha:
    print("Reject the null hypothesis (significant difference).")
else:
    print("Fail to reject the null hypothesis (no significant difference).")
