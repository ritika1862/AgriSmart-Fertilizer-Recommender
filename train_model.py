import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------------
# 1. LOAD DATASET
# -----------------------------------
BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "data" / "fertilizer_recommendation.csv"

df = pd.read_csv(file_path)
print("Dataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# -----------------------------------
# 2. SEPARATE INPUT AND TARGET
# -----------------------------------
target = "Recommended_Fertilizer"
X = df.drop(columns=[target])
y = df[target]

print("\nTarget column:", target)
print("Number of fertilizer classes:", y.nunique())
print("Fertilizer classes:", y.unique())

# -----------------------------------
# 3. IDENTIFY COLUMN TYPES
# -----------------------------------
categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
numerical_columns = X.select_dtypes(exclude=["object"]).columns.tolist()

print("\nCategorical columns:", categorical_columns)
print("Numerical columns:", numerical_columns)

# -----------------------------------
# 4. PREPROCESSING & PIPELINE
# -----------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ("numerical", "passthrough", numerical_columns)
    ]
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# -----------------------------------
# 5. TRAIN / TEST SPLIT & TRAINING
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)} | Testing samples: {len(X_test)}")
print("Training Random Forest classifier...")
pipeline.fit(X_train, y_train)
print("Model training completed!")

# -----------------------------------
# 6. EVALUATION
# -----------------------------------
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n-----------------------------------")
print("MODEL PERFORMANCE")
print("-----------------------------------")
print(f"Accuracy: {round(accuracy * 100, 2)}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------------
# 7. SAVE MODEL
# -----------------------------------
model_save_path = BASE_DIR / "fertilizer_model.pkl"
joblib.dump(pipeline, model_save_path)
print(f"\nModel saved successfully to {model_save_path}!")