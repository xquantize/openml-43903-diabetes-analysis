from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train, preprocessor):
    """Train a Random Forest classifier within a pipeline."""
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline
