import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')
# loading the data from csv file to a Pandas DataFrame
parkinsons_data = pd.read_csv('/content/Parkinsson disease-1.csv')
# printing the first 5 rows of the dataframe
parkinsons_data.head()
# number of rows and columns in the dataframe
parkinsons_data.shape
# getting more information about the dataset
parkinsons_data.info()
# checking for missing values in each column
parkinsons_data.isnull().sum()
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

# Plot the distributions
sns.histplot(parkinsons_data['MDVP:Jitter(Abs)'], ax=axes[0, 0], kde=True)
axes[0, 0].set_title('MDVP:Jitter(Abs) Distribution')

sns.histplot(parkinsons_data['MDVP:Shimmer'], ax=axes[0, 1], kde=True)
axes[0, 1].set_title('MDVP:Shimmer Distribution')

sns.histplot(parkinsons_data['Shimmer:APQ5'], ax=axes[1, 0], kde=True)
axes[1, 0].set_title('Shimmer:APQ5 Distribution')

sns.histplot(parkinsons_data['Shimmer:DDA'], ax=axes[1, 1], kde=True)
axes[1, 1].set_title('Shimmer:DDA Distribution')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()
median_values = parkinsons_data[['MDVP:Shimmer', 'MDVP:Jitter(Abs)', 'Shimmer:APQ5', 'Shimmer:DDA']].median()
parkinsons_data['MDVP:Shimmer'].fillna(median_values['MDVP:Shimmer'], inplace=True)
parkinsons_data['MDVP:Jitter(Abs)'].fillna(median_values['MDVP:Jitter(Abs)'], inplace=True)
parkinsons_data['Shimmer:APQ5'].fillna(median_values['Shimmer:APQ5'], inplace=True)
parkinsons_data['Shimmer:DDA'].fillna(median_values['Shimmer:DDA'], inplace=True)
# checking for missing values in each column
parkinsons_data.isnull().sum()
# getting some statistical measures about the data
parkinsons_data.describe()
# distribution of target Variable
parkinsons_data['status'].value_counts()
plt.figure(figsize=(6,5))
ax = sns.countplot(x = parkinsons_data['status'])
# grouping the data bas3ed on the target variable
parkinsons_data.groupby('status').mean()
X = parkinsons_data.drop(columns=['name','status'], axis=1)
y = parkinsons_data['status']
print(X)
print(y)
X_numeric = X.select_dtypes(include=['float64', 'int64'])
X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.1, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train)
best_rf_classifier = RandomForestClassifier(random_state=42)
best_rf_classifier.fit(X_train, y_train)
rf_predictions = best_rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
precision = precision_score(y_test, rf_predictions)
recall = recall_score(y_test, rf_predictions)
f1 = f1_score(y_test, rf_predictions)
print("Random Forest Accuracy",rf_accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy",rf_accuracy)
# Initialize and fit an SVM classifier
svm_classifier = SVC(probability=True, random_state=42)
svm_classifier.fit(X_train, y_train)
# Make predictions using the SVM model
svm_predictions = svm_classifier.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions)
svm_recall = recall_score(y_test, svm_predictions)
svm_f1 = f1_score(y_test, svm_predictions)

print("SVM Accuracy:", svm_accuracy)
print("SVM Precision:", svm_precision)
print("SVM Recall:", svm_recall)
print("SVM F1-Score:", svm_f1)
# Calculate accuracy for the SVM model
svm_accuracy = accuracy_score(y_test, svm_predictions)
print("SVM Accuracy:", svm_accuracy)
input_data = (119.99200, 157.30200, 74.99700, 0.00784, 0.00007, 0.00370, 0.00554, 0.01109, 0.04374, 0.42600,
              0.02182, 0.03130, 0.02971, 0.06545, 0.02211, 21.03300, 0.414783, 0.815285, -4.813031, 0.266482, 2.301442, 0.284654)
input_data_array = np.asarray(input_data)
input_data_reshaped = input_data_array.reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)
rf_prediction = best_rf_classifier.predict(std_data)

if rf_prediction[0] == 0:
    print("Random Forest Prediction: The Person does not have Parkinson's")
else:
    print("Random Forest Prediction: The Person has Parkinson's")


# Define the classifiers and their corresponding performance metrics
classifiers = ["Random Forest", "SVM"]
precisions = [precision, svm_precision]
recalls = [recall, svm_recall]
f1_scores = [f1, svm_f1]
accuracies = [rf_accuracy, svm_accuracy]

# Set up the subplots
fig, ax = plt.subplots(figsize=(10,6))

# Define the width of the bars
width = 0.1

# Set the x positions for the bars
x = range(len(classifiers))

# Create bars for precision, recall, and F1-score
plt.bar(x, precisions, width, label='Precision', align='center')
plt.bar([i + width for i in x], recalls, width, label='Recall', align='center')
plt.bar([i + 2 * width for i in x], f1_scores, width, label='F1-Score', align='center')

# Set labels and title
plt.xlabel('Classifiers')
plt.ylabel('Scores')
plt.title('Performance Metrics Comparison')
plt.xticks([i + width for i in x], classifiers)
plt.legend(loc='upper right')

# Show the bar chart
plt.tight_layout()
plt.show()

# Create a separate bar chart for accuracy
plt.figure(figsize=(6, 6))
plt.bar(classifiers, accuracies, color='skyblue')
plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.ylim(0, 1)  # Set the y-axis limit to 0-1
plt.show()
