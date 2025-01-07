import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import joblib

def main():
    # Read and preprocess the dataset
    df = pd.read_csv("processed/facial_features_original.csv")
    valence = df["valence"]
    arousal = df["arousal"]
    inputs = df.drop(["subDirectory_filePath", "expression", "valence", "arousal"], axis=1)

    # Do a balanced split of the dataset for train/val/test: Split into 70/20/10
    # Training and Validation/Test: 90/10
    X_train_val, X_test, y_train_val_valence, y_test_valence, y_train_val_arousal, y_test_arousal = train_test_split(
        inputs,
        valence,
        arousal,
        test_size=0.1,
        random_state=42
    )

    # Training/Validation: 70/20
    X_train, X_val, y_train_valence, y_val_valence, y_train_arousal, y_val_arousal = train_test_split(
        X_train_val,
        y_train_val_valence,
        y_train_val_arousal,
        test_size=(0.2 / 0.9),  # 20% of original data
        random_state=42
    )

    print("\nSplit of the data:", len(X_train), len(X_val), len(X_test), "\n")

    # Define models and hyperparameter grids
    models_and_params = [
        ("KNeighborsRegressor", KNeighborsRegressor(), {
            "n_neighbors": [5, 10, 15, 20],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
        }),
        ("SVR", SVR(), {
            "C": [1, 5, 10, 20],
            "kernel": ["poly", "rbf", "linear", "sigmoid"],
            "degree": [1, 2, 3],
            "gamma": ["scale", "auto"]
        }),
        ("RandomForestRegressor", RandomForestRegressor(), {
            "n_estimators": [50, 100, 150],
            "max_features": ["sqrt", "log2"],
            "max_depth": [4, 5, 6, 7, 8],
            "criterion": ["squared_error", "absolute_error"],
        }),
        ("DecisionTreeRegressor", DecisionTreeRegressor(), {
            "criterion": ["squared_error", "absolute_error"],
            "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 10],
        }),
        ("LinearRegression", LinearRegression(), {}),
        ("GradientBoostingRegressor", GradientBoostingRegressor(), {
            "max_depth": [3, 5, 10],
            "max_features": ["sqrt", "log2"],
        })
    ]

    best_models = {"valence": {}, "arousal": {}}

    # Train and tune models for valence and arousal
    for target, y_train, y_test, y_train_val, y_test_val in [
        ("valence", y_train_valence, y_test_valence, y_train_val_valence, y_test_valence),
        ("arousal", y_train_arousal, y_test_arousal, y_train_val_arousal, y_test_arousal)
    ]:
        print(f"\nTraining models for {target}:")

        best_model = {}

        for model_name, model, param_grid in models_and_params:

            current_model = GridSearchCV(model, param_grid, scoring="r2")
            current_model.fit(X_train_val, y_train_val)

            # Evaluate on the test set
            y_pred = current_model.predict(X_test)
            test_r2 = r2_score(y_test, y_pred)
            test_mse = mean_squared_error(y_test, y_pred)

            print(
                f"{model_name} - Best parameters: {current_model.best_params_}, Test R²: {test_r2}, Test MSE: {test_mse}")

            # Save the best model
            if not best_model or test_mse < best_model["best_mse"]:
                best_model = {
                    "model_name": model_name,
                    "model": current_model.best_estimator_,
                    "best_params": current_model.best_params_,
                    "best_r2": test_r2,
                    "best_mse": test_mse
                }

        best_models[target] = best_model
        print(
            f"Best model for {target}: {best_model['model_name']} with parameters: {best_model['best_params']} and MSE: {best_model['best_mse']}")

    # Save the best models for valence and arousal
    joblib.dump(best_models["valence"], "scripts/subSystem1/models/valence_model.pkl")
    joblib.dump(best_models["arousal"], "scripts/subSystem1/models/arousal_model.pkl")

if __name__ == '__main__':
    main()
