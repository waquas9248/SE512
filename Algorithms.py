import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the data from the file
file_path = './online-payments-fraud-detection-dataset.csv' 
data = pd.read_csv(file_path)

# Separate features and target variable
data['type'] = data['type'].map({'CASH_OUT':1, 'PAYMENT':2, 'CASH_IN':3, 'TRANSFER':4, 'DEBIT':5})
X = data[['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
y = data['isFraud']

#map non-numeric columns

# Split the data into training (80%), validation (10%), and testing (10%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize the classifiers
nb_model = GaussianNB()
svm_model = SVC()
dt_model = DecisionTreeClassifier()




# Train the Naive Bayes model
start_time = time.time()
nb_model.fit(X_train, y_train)

# Make predictions on the validation set for Naive Bayes
y_val_pred_nb = nb_model.predict(X_val)

# Evaluate the Naive Bayes model on the validation set
accuracy_val_nb = accuracy_score(y_val, y_val_pred_nb)
conf_matrix_val_nb = confusion_matrix(y_val, y_val_pred_nb)
class_report_val_nb = classification_report(y_val, y_val_pred_nb)

# Record the run time
end_time = time.time()
elapsed_time = end_time - start_time
minutes, seconds = divmod(elapsed_time, 60)

print("Naive Bayes - Validation Set Results:")
print(f"Accuracy: {accuracy_val_nb:.2f}")
print(f"{int(minutes)} minutes and {seconds:.2f} seconds")
print("Confusion Matrix:")
print(conf_matrix_val_nb)
print("Classification Report:")
print(class_report_val_nb)




# Train the Decision Trees model
start_time = time.time()
dt_model.fit(X_train, y_train)

# Make predictions on the validation set for Decision Trees
y_val_pred_dt = dt_model.predict(X_val)

# Evaluate the Decision Trees model on the validation set
accuracy_val_dt = accuracy_score(y_val, y_val_pred_dt)
conf_matrix_val_dt = confusion_matrix(y_val, y_val_pred_dt)
class_report_val_dt = classification_report(y_val, y_val_pred_dt)

# Record the run time
end_time = time.time()
elapsed_time = end_time - start_time
minutes, seconds = divmod(elapsed_time, 60)

print("\nDecision Trees - Validation Set Results:")
print(f"Accuracy: {accuracy_val_dt:.2f}")
print(f"{int(minutes)} minutes and {seconds:.2f} seconds")
print("Confusion Matrix:")
print(conf_matrix_val_dt)
print("Classification Report:")
print(class_report_val_dt)




# Train the SVM model
start_time = time.time()
svm_model.fit(X_train, y_train)

# Make predictions on the validation set for SVM
y_val_pred_svm = svm_model.predict(X_val)

# Evaluate the SVM model on the validation set
accuracy_val_svm = accuracy_score(y_val, y_val_pred_svm)
conf_matrix_val_svm = confusion_matrix(y_val, y_val_pred_svm)
class_report_val_svm = classification_report(y_val, y_val_pred_svm)

# Record the run time
end_time = time.time()
elapsed_time = end_time - start_time
minutes, seconds = divmod(elapsed_time, 60)

print("\nSVM - Validation Set Results:")
print(f"Accuracy: {accuracy_val_svm:.2f}")
print(f"{int(minutes)} minutes and {seconds:.2f} seconds")
print("Confusion Matrix:")
print(conf_matrix_val_svm)
print("Classification Report:")
print(class_report_val_svm)


# Test the models on the test set
y_test_pred_nb = nb_model.predict(X_test)
y_test_pred_svm = svm_model.predict(X_test)
y_test_pred_dt = dt_model.predict(X_test)

# Evaluate the models on the test set
accuracy_test_nb = accuracy_score(y_test, y_test_pred_nb)
accuracy_test_svm = accuracy_score(y_test, y_test_pred_svm)
accuracy_test_dt = accuracy_score(y_test, y_test_pred_dt)

print("\nTest Set Results:")
print(f"Naive Bayes Accuracy: {accuracy_test_nb:.2f}")
print(f"SVM Accuracy: {accuracy_test_svm:.2f}")
print(f"Decision Trees Accuracy: {accuracy_test_dt:.2f}")