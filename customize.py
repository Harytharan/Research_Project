
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def build_custom_model(use_tuning=False):
    base_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced"
    )

    if not use_tuning:
        print("model configuration.")
        return base_model

    print("GridSearchCV hyperparameter tuning")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [8, 10, 12, None],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 3]
    }

    grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
    return grid_search
