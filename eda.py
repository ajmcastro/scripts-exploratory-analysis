import pandas as pd
import matplotlib
import seaborn as sns
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

# Load the CSV data
data = pd.read_csv('data/iris.csv')

# Display the first few rows of the DataFrame
print(data.head())

# Get basic statistics for each column (like count, mean, std, min, 25%, 50%, 75%, max)
print(data.describe(include='all'))

# Get information about the data types,columns, null value counts, memory usage etc
print(data.info())

# univariate plots, that is, plots of each individual variable
# box and whisker plots
data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)

# Visualize the distribution of data for every feature
data.hist(figsize=(5, 5))
pyplot.show()

# Multivariate Plots - look at the interactions between the variables.
# scatter plot matrix
scatter_matrix(data)
pyplot.show()

# Let's also check the correlation between different variables
correlation = data.corr()
pyplot.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True)
pyplot.show()

# For categorical features you can use value_counts
# Replace 'column_name' with the name of a column
print(data['column_name'].value_counts())
