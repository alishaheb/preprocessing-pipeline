import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# --- 1️⃣ Load Titanic dataset ---
df = pd.read_csv("titanic.csv")  # <- make sure your file name is correct

print("✅ Data loaded successfully!")
print("Shape:", df.shape)
print(df.head())

# --- 2️⃣ Basic cleaning ---
# Drop irrelevant columns
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], errors='ignore')

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# --- 3️⃣ Define features and target ---
X = df.drop(columns=['Survived'])
y = df['Survived']

# --- 4️⃣ Split data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 5️⃣ Preprocessing ---
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# --- 6️⃣ Build pipeline ---
model = LogisticRegression(max_iter=1000)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# --- 7️⃣ Train model ---
pipeline.fit(X_train, y_train)

# --- 8️⃣ Evaluate model ---
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n📊 Evaluation Results:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nFull Classification Report:")
print(classification_report(y_test, y_pred))
