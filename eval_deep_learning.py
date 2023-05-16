from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Initialize the constructor
model = Sequential()

# Add an input layer and a hidden layer
model.add(Dense(12, activation='relu', input_shape=(X_train.shape[1],)))

# Add one hidden layer
model.add(Dense(8, activation='relu'))

# Add an output layer
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Fit model
model.fit(X_train, y_train,epochs=20, batch_size=1, verbose=1)

# Predict the test set
y_pred = model.predict_classes(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
