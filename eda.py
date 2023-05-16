import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV data
data = pd.read_csv('path_to_your_file/data.csv')

# Display the first few rows of the DataFrame
print(data.head())

# Get basic statistics for each column (like count, mean, std, min, 25%, 50%, 75%, max)
print(data.describe(include='all'))

# Get information about the data types,columns, null value counts, memory usage etc
print(data.info())

# Visualize the distribution of data for every feature
data.hist(figsize=(20,20))
plt.show()

# Let's also check the correlation between different variables
correlation = data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, cmap=plt.cm.Reds)
plt.show()

# For categorical features you can use value_counts
# Replace 'column_name' with the name of a column
print(data['column_name'].value_counts())