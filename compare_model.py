import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score


# -------------------------------
# 1. LOAD DATA
# -------------------------------

df = pd.read_csv("data/fertilizer_recommendation.csv")

target = "Recommended_Fertilizer"

X = df.drop(columns=[target])
y = df[target]


# -------------------------------
# 2. IDENTIFY COLUMNS
# -------------------------------

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


# -------------------------------
# 3. PREPROCESSOR
# -------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)


# -------------------------------
# 4. TRAIN / TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -------------------------------
# 5. MODELS
# -------------------------------

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=2000
    )
}


# -------------------------------
# 6. TRAIN AND COMPARE
# -------------------------------

results = {}

for name, model in models.items():

    print("\nTraining:", name)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results[name] = accuracy

    print(
        name,
        "Accuracy:",
        round(accuracy * 100, 2),
        "%"
    )


# -------------------------------
# 7. FINAL COMPARISON
# -------------------------------

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

for name, accuracy in results.items():

    print(
        name,
        "->",
        round(accuracy * 100, 2),
        "%"
    )


best_model = max(
    results,
    key=results.get
)

print("\nBest Model:", best_model)
print(
    "Best Accuracy:",
    round(results[best_model] * 100, 2),
    "%"
)