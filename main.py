import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt





# Preprocessing
names = ['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure', 'Rain']
data = pd.read_csv('weather_forecast_data.csv', names=names, header=0)

print("________________________________________")
# Display missing data before preprocessing
print("Missing Data for each feature (before preprocessing):")
print(data.isnull().sum())

print("________________________________________")
# Only handle missing data without modifying valid values
numeric_features = ['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']
for feature in numeric_features:
    missing_count = data[feature].isnull().sum()
    print(f"Missing values in {feature}: {missing_count}")

    # Calculate mean of non-missing values
    feature_mean = data[feature].dropna().mean()

    # Fill missing values with the calculated mean
    data[feature] = data[feature].fillna(feature_mean)

print("________________________________________")
# Save the preprocessed DataFrame to a new CSV file
output_file = 'cleaned_weather_forecast_data.csv'
data.to_csv(output_file, index=False)
print(f"\nPreprocessed data has been saved to '{output_file}'.")

# Display missing data after preprocessing
print("________________________________________")
print("\nMissing Data for each feature (after preprocessing):")
print(data.isnull().sum())

print("________________________________________")
print("\nRange of each numeric feature:")
for feature in numeric_features:
    value_range = data[feature].max() - data[feature].min()
    print(f"{feature}: {value_range}")

# Define features and target variable
X = data[numeric_features]  # Independent variables
y = data['Rain']            # Target variable

# Feature Scaling: Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaled data to a new DataFrame
scaled_data = pd.DataFrame(X_scaled, columns=numeric_features)
scaled_data['Rain'] = y
scaled_output_file = 'scaled_weather_forecast_data.csv'
scaled_data.to_csv(scaled_output_file, index=False)


# Splitting into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

print("________________________________________")
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")


# __________________________________________________________________________________________
#                              | Task 2 |
# __________________________________________________________________________________________

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
dt_predictions = dt_classifier.predict(X_test)
print("________________________________________")
print("Decision Tree Results")
print(f"Accuracy: {accuracy_score(y_test, dt_predictions):.2f}")
print(classification_report(y_test, dt_predictions))

# k-Nearest Neighbors Classifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)
print("\nk-Nearest Neighbors Results")
print(f"Accuracy: {accuracy_score(y_test, knn_predictions):.2f}")
print(classification_report(y_test, knn_predictions))


# Naïve Bayes Classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
nb_predictions = nb_classifier.predict(X_test)
print("\nNaïve Bayes Results")
print(f"Accuracy: {accuracy_score(y_test, nb_predictions):.2f}")
print(classification_report(y_test, nb_predictions))
print("________________________________________")


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        predictions = []

        for x in X:
            # Compute distances between x and all examples in the training set
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

            # Sort by distance and return indices of the first k neighbors
            k_indices = np.argsort(distances)[:self.k]

            # Extract the labels of the k nearest neighbor training samples
            k_nearest_labels = [self.y_train[i] for i in k_indices]

            # Find the most common class label
            label_counts = {}
            for label in k_nearest_labels:
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1

            # Get the label with the highest count
            most_common_label = max(label_counts, key=label_counts.get)

            # Append prediction to the list
            predictions.append(most_common_label)

        return np.array(predictions)





print("KNN Predictions: ")
knn = KNN(5000)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
accuracy = np.sum(predictions == y_test) / len(y_test)
y_pred = knn.predict(X_test)
classes = np.unique(y_test)
precision = {}
recall = {}

for cls in classes:
    tp = np.sum((y_pred == cls) & (y_test == cls))
    fp = np.sum((y_pred == cls) & (y_test != cls))
    fn = np.sum((y_pred != cls) & (y_test == cls))

    precision[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f'Accuracy of the implemented is {accuracy}')
print(f"Precision of the implementation for '{classes[0]}' is {precision[classes[0]]:.2f}")
print(f"Precision of the implementation for '{classes[1]}' is {precision[classes[1]]:.2f}")
print(f"Recall of the implementation for '{classes[0]}' is {recall[classes[0]]:.2f}")
print(f"Recall of the implementation for '{classes[1]}' is {recall[classes[1]]:.2f}")

print("________________________________________")


# __________________________________________________________________________________________
#                              | Task 3 |
# __________________________________________________________________________________________

# Handling the missing data by dropping rows with missing values
print("________________________________________")
newData = pd.read_csv('weather_forecast_data.csv', names=names, header=0)
data_dropped = newData.dropna()

output_file = 'dropped_weather_forecast_data.csv'  # Fixed variable name
data_dropped.to_csv(output_file, index=False)  # Save the dropped data instead of the original
print(f"\nPreprocessed data has been saved to '{output_file}'.")
print("________________________________________")

X = newData[numeric_features]  # Independent variables
y = newData['Rain']            # Target variable


# Splitting into training and testing datasets
Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_scaled, y, test_size=0.2)
scaled_data = pd.DataFrame(X_scaled, columns=numeric_features)
scaled_data['Rain'] = y


# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(Xs_train, ys_train)
dt_predictions = dt_classifier.predict(Xs_test)
print("________________________________________")
print("Decision Tree Results")
print(f"Accuracy: {accuracy_score(y_test, dt_predictions):.2f}")
print(classification_report(y_test, dt_predictions))

# k-Nearest Neighbors Classifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(Xs_train, ys_train)
knn_predictions = knn_classifier.predict(Xs_test)
print("\nk-Nearest Neighbors Results")
print(f"Accuracy: {accuracy_score(ys_test, knn_predictions):.2f}")
print(classification_report(ys_test, knn_predictions))


# Naïve Bayes Classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
nb_predictions = nb_classifier.predict(X_test)
print("\nNaïve Bayes Results")
print(f"Accuracy: {accuracy_score(y_test, nb_predictions):.2f}")
print(classification_report(y_test, nb_predictions))
print("________________________________________")

# Plot the Decision Tree
# Plot the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_classifier, filled=True, rounded=True, class_names=['No Rain', 'Rain'], feature_names=numeric_features)
plt.title("Decision Tree")
plt.savefig("decision_tree.png")
plt.show()
