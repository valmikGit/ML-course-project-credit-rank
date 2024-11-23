  
# Importing the necessary Python libraries

 
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB

 
# Load your dataset
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
test_ids = pd.read_csv("test.csv")["ID"]
# Analyze your dataset
# ID, Customer_ID, Month, Name, Age, Profession, Number are columns to be dropped
train_data = train_data.drop(columns=["ID", "Customer_ID", "Month", "Name", "Profession", "Number", "Loan_Type"])

print(train_data.head())

# Loan_Type, Credit_Mix, Payment_of_Min_Amount, Payment_Behaviour , and Credit_Score are categorical columns
# Credit_Score is the target column

 
# Fill missing values with median in train data
train_data.fillna(train_data.median(numeric_only=True), inplace=True)
print(train_data.head())
 
# Convert relevant columns to numeric after removing any underscores.This was done as I saw a value in Total_Current_Loans with an underscore.
for col in ["Total_Current_Loans", "Current_Debt_Outstanding", "Income_Annual", "Credit_Limit", "Age", "Total_Credit_Cards", "Total_Bank_Accounts", "Delay_from_due_date"]:
    train_data[col] = pd.to_numeric(
        train_data[col].astype(str).str.replace("_", "", regex=False), errors="coerce"
    )
    test_data[col] = pd.to_numeric(
        test_data[col].astype(str).str.replace("_", "", regex=False), errors="coerce"
    )
train_data
# Income_Annual, Base_Salary_PerMonth,Current_Debt_Outstanding,Ratio_Credit_Utilization, Per_Month_EMI, Monthly_Investment
 
#convert credit_history_age to numeric
# Format : 1 Years and 2 Months
train_data["Credit_History_Age"] = train_data["Credit_History_Age"].str.extract("(\d+)").astype(float)
test_data["Credit_History_Age"] = test_data["Credit_History_Age"].str.extract("(\d+)").astype(float)
train_data["Credit_History_Age"]
 
# Replace infinity values in train data
train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
train_data.fillna(train_data.median(numeric_only=True), inplace=True)

# Label encode the target variable in train data
label_encoder = LabelEncoder()
train_data["Credit_Score"] = label_encoder.fit_transform(train_data["Credit_Score"])
 
# Prepare training features and labels
X_train = train_data.drop(columns="Credit_Score")
y_train = train_data["Credit_Score"]

 
# Feature engineering in train data
train_data["Debt_Income_Ratio"] = (
    train_data["Current_Debt_Outstanding"] / train_data["Income_Annual"]
)
train_data["Income_Credit_Limit_Ratio"] = (
    train_data["Income_Annual"] / train_data["Credit_Limit"]
)
train_data["Debt_Credit_Limit_Ratio"] = (
    train_data["Current_Debt_Outstanding"] / train_data["Credit_Limit"]
)
 
# print Monthly_Balance column
# print(X_train["Monthly_Investment"])
# print(X_train.select_dtypes(include=["object"]).columns)
# Replace all str values in Monthly_Balance, Monthly_Investment

X_train["Monthly_Balance"] = pd.to_numeric(
    X_train["Monthly_Balance"].astype(str).str.replace("_", "", regex=False), errors="coerce"
)

X_train["Monthly_Investment"] = pd.to_numeric(
    X_train["Monthly_Investment"].astype(str).str.replace("_", "", regex=False), errors="coerce"
)

# Fill missing values with median for Monthly_Balance and Monthly_Investment
X_train.fillna(X_train.median(numeric_only=True), inplace=True)

# Same for test data
test_data["Monthly_Balance"] = pd.to_numeric(
    test_data["Monthly_Balance"].astype(str).str.replace("_", "", regex=False), errors="coerce"
)

test_data["Monthly_Investment"] = pd.to_numeric(
    test_data["Monthly_Investment"].astype(str).str.replace("_", "", regex=False), errors="coerce"
)

test_data.fillna(test_data.median(numeric_only=True), inplace=True)

# print Monthly_Balance column
print(X_train["Monthly_Investment"], X_train["Monthly_Balance"])
 
# Step 1: Identify numeric and categorical features


X_train.replace("-", np.nan, inplace=True)
X_train.replace("NM", np.nan, inplace=True)

categorical_features = [
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
    "Total_Delayed_Payments"
]
X_train[categorical_features] = X_train[categorical_features].astype(str)

numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X_train.select_dtypes(include=["object"]).columns

# Step 2: Handle Missing Values
# Use SimpleImputer to fill in missing values. Strategies: 'mean' for numerical, 'most_frequent' for categorical.

numerical_pipeline = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

# Define the preprocessing steps for numeric features
# numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])

# Define the preprocessing steps for categorical features
# Using OneHotEncoder to convert categorical variables into binary (dummy) variables.
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),  #! Can change this step for different encoding methods
    ]
)

# Step 3: Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)
 
#Print the columns
print(X_train.columns)
# Print Numerical columns
print(X_train.select_dtypes(include=["int64", "float64"]).columns)
# Print Categorical columns
print(X_train.select_dtypes(include=["object"]).columns)
 
# Feature engineering in test data
test_data["Debt_Income_Ratio"] = (
    test_data["Current_Debt_Outstanding"] / test_data["Income_Annual"]
)
test_data["Income_Credit_Limit_Ratio"] = (
    test_data["Income_Annual"] / test_data["Credit_Limit"]
)
test_data["Debt_Credit_Limit_Ratio"] = (
    test_data["Current_Debt_Outstanding"] / test_data["Credit_Limit"]
)
 
# Check for mixed data types in each column
for col in X_train.columns:
    unique_types = set(type(x) for x in X_train[col].dropna())
    if len(unique_types) > 1:
        print(f"Column '{col}' has mixed types: {unique_types}")
 
print(X_train.select_dtypes(include=["int64", "float64"]).columns)
 
# Step 4: Apply preprocessing

# Apply transformations to the training data
X_preprocessed = preprocessor.fit_transform(X_train)

# Convert to DataFrame and assign column names
X_train_df = pd.DataFrame.sparse.from_spmatrix(X_preprocessed)
X_train_df.columns = preprocessor.get_feature_names_out()

# Apply transformations to the test data and convert to DataFrame with column names
test_data_preprocessed = preprocessor.transform(test_data)
test_data_df = pd.DataFrame.sparse.from_spmatrix(test_data_preprocessed)
test_data_df.columns = preprocessor.get_feature_names_out()

# Step 5: Train-Test Split for the preprocessed data
X_train_1, X_val_1, y_train_1, y_val = train_test_split(
    X_train_df, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Verify the transformations
print("Transformed Training Data Shape:", X_train_1.shape)
print("Transformed Validation Data Shape:", X_val_1.shape)
print("Transformed Test Data Shape:", test_data_df.shape)
 
models = {}
# model = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
# models["Random forest"] = model
# model = XGBClassifier(
#     learning_rate=0.05,
#     max_depth=6,
#     n_estimators=300,
#     random_state=42,
#     eval_metric="mlogloss",
# )
model = XGBClassifier(
    learning_rate=0.17031088174537234,
    max_depth=9,
    n_estimators=177,
    random_state=42,
    eval_metric="mlogloss",
)
models["XGBoost"] = model
 
y_train
print(X_train_1.columns)
print(y_train)
# y_train should have 1 column of credit score
 
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score


# Objective function for Optuna
def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }

    # Initialize the classifier with trial parameters
    model = XGBClassifier(**params, random_state=42, eval_metric="mlogloss")

    # 5-fold cross-validation
    accuracy = cross_val_score(model, X_train_1, y_train_1, cv=5, scoring="accuracy").mean()
    return accuracy


# Run Bayesian Optimization with Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Best parameters and score
print("Best parameters found: ", study.best_params)
print("Best accuracy: ", study.best_value)
 
for key, value in models.items():
    try:  
        # Define model pipeline
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", value)])

        # Fit the pipeline on training data
        pipeline.fit(X_train, y_train)
        
        # Make predictions on test data
        test_predictions = pipeline.predict(test_data)

        # Convert predictions back to original labels
        test_predictions_labels = label_encoder.inverse_transform(test_predictions)

        # test_predictions_encoded = label_encoder.transform(test_predictions)
        # Prepare the submission file
        submission = pd.DataFrame(
            {"ID": test_ids, "Credit_Score": test_predictions_labels}
        )
        submission.to_csv(f"submission_{key}.csv", index=False)

        print(f"Submission file 'submission_{key}.csv' created successfully!")
    except Exception as e:
        print(f"Error : {e}")


