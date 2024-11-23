# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load your dataset
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
test_ids = test_data["ID"]

# Drop unnecessary columns
train_data = train_data.drop(columns=["ID", "Customer_ID", "Month", "Name", "Profession", "Number", "Loan_Type"])

# Fill missing values with median in train data
train_data.fillna(train_data.median(numeric_only=True), inplace=True)

# Convert relevant columns to numeric after removing underscores
numeric_columns = [
    "Total_Current_Loans", "Current_Debt_Outstanding", "Income_Annual", 
    "Credit_Limit", "Age", "Total_Credit_Cards", "Total_Bank_Accounts", 
    "Delay_from_due_date", "Monthly_Balance", "Monthly_Investment"
]

for col in numeric_columns:
    train_data[col] = pd.to_numeric(train_data[col].astype(str).str.replace("_", "", regex=False), errors="coerce")
    test_data[col] = pd.to_numeric(test_data[col].astype(str).str.replace("_", "", regex=False), errors="coerce")

# Convert "Credit_History_Age" to numeric
train_data["Credit_History_Age"] = train_data["Credit_History_Age"].str.extract("(\d+)").astype(float)
test_data["Credit_History_Age"] = test_data["Credit_History_Age"].str.extract("(\d+)").astype(float)

# Replace infinity and fill missing values in train data
train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
train_data.fillna(train_data.median(numeric_only=True), inplace=True)

# Label encode the target variable
label_encoder = LabelEncoder()
train_data["Credit_Score"] = label_encoder.fit_transform(train_data["Credit_Score"])

# Prepare training features and labels
X_train = train_data.drop(columns="Credit_Score")
y_train = train_data["Credit_Score"]

# Replace problematic values and fill missing data for categorical features
categorical_features = ["Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour", "Total_Delayed_Payments"]
X_train[categorical_features] = X_train[categorical_features].astype(str)
X_train.replace(["-", "NM"], np.nan, inplace=True)
test_data[categorical_features] = test_data[categorical_features].astype(str)
test_data.replace(["-", "NM"], np.nan, inplace=True)

# Define pipelines for preprocessing
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing pipelines
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_pipeline, X_train.select_dtypes(include=["float64", "int64"]).columns),
    ("cat", categorical_pipeline, categorical_features)
])

# Apply preprocessing
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(test_data)

# Convert preprocessed data to numpy arrays
X_train_np = X_train_preprocessed
y_train_np = np.array(y_train)
X_test_np = X_test_preprocessed

# Split training data into training and validation sets
X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(X_train_np, y_train_np, test_size=0.2, random_state=42, stratify=y_train_np)

# Define the SVM model
svm_model = SVC(kernel="rbf", C=1, gamma="scale", probability=True, random_state=42)

# Train the model
svm_model.fit(X_train_np, y_train_np)

# Evaluate the model on validation data
val_predictions = svm_model.predict(X_val_np)
val_accuracy = accuracy_score(y_val_np, val_predictions)
print(f"Validation Accuracy: {val_accuracy}")

# Make predictions on test data
test_predictions = svm_model.predict(X_test_np)
test_predictions_labels = label_encoder.inverse_transform(test_predictions)

# Prepare the submission file
submission = pd.DataFrame({"ID": test_ids, "Credit_Score": test_predictions_labels})
submission.to_csv("submission_SVM.csv", index=False)
print("Submission file 'submission_SVM.csv' created successfully!")
