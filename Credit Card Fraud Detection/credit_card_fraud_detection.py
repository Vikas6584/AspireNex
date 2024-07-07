import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data (replace 'credit_card_data.csv' with your actual file path)
data = pd.read_csv('creditcard.csv')

# Separate features and target variable
features = data.drop('Class', axis=1)  # Assuming 'Class' is the target label
target = data['Class']

# Preprocessing
# Handle missing values (replace with your imputation strategy)
features.fillna(features.mean(), inplace=True)

# Scale numeric features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Encode categorical features (if any)
encoder = OneHotEncoder()
features_encoded = encoder.fit_transform(features_scaled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.2)

# Logistic Regression Model
# Train the model
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

# Predict on test data
y_pred_lr = model_lr.predict(X_test)

# Evaluate model performance (Logistic Regression)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

print("Logistic Regression Results:")
print(f"Accuracy: {accuracy_lr:.4f}")
print(f"Precision: {precision_lr:.4f}")
print(f"Recall: {recall_lr:.4f}")
print(f"F1 Score: {f1_lr:.4f}")

# Decision Tree Model
# Train the model (replace with desired hyperparameters)
model_dt = DecisionTreeClassifier(max_depth=5)
model_dt.fit(X_train, y_train)

# Predict on test data
y_pred_dt = model_dt.predict(X_test)

# Evaluate model performance (Decision Tree)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

print("\nDecision Tree Results:")
print(f"Accuracy: {accuracy_dt:.4f}")
print(f"Precision: {precision_dt:.4f}")
print(f"Recall: {recall_dt:.4f}")
print(f"F1 Score: {f1_dt:.4f}")