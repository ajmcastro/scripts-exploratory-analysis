from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd

# Load the CSV data
data = pd.read_csv('path_to_your_file/data.csv')

# Display the first few rows of the DataFrame
print(data.head())

# Get basic statistics for each column (like count, mean, std, min, 25%, 50%, 75%, max)
print(data.describe(include='all'))

# Get information about the data types,columns, null value counts, memory usage etc
print(data.info())

# Preprocessing: Assuming all columns except 'target' are features
X = data.drop(['target'], axis=1)
y = data['target']

# Standardize features by removing the mean and scaling to unit variance
sc = StandardScaler()
X = sc.fit_transform(X)

# Split the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a dictionary of different regressors we would like to test
regressors = {
    "Linear Regression": LinearRegression(),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "Support Vector Regression": SVR(),
    "Decision Tree Regression": DecisionTreeRegressor(),
    "Random Forest Regression": RandomForestRegressor()
}

# Evaluate the models
for name, reg in regressors.items():
    # Train the model
    reg.fit(X_train, y_train)

    # Predict the test set
    predictions = reg.predict(X_test)

    # Evaluate the model
    print(name)
    print("Mean Squared Error: ", mean_squared_error(y_test, predictions))
    print("Mean Absolute Error: ", mean_absolute_error(y_test, predictions))
    print("R^2 Score: ", r2_score(y_test, predictions))
    print()

