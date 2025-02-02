import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import joblib

"""
This is a machine learning pipeline that produces different models 
for predicting emotions from Action Units, Arousal and Valence. The model with best accuracy will later 
be selected to predict new unseen information from faces

INPUT: Facial_features_original.csv (Arousal, Valence and Action Units)
OUTPUT: Model with configurations that produced the best accuracy, saved in models/emotion_model.pkl
"""

#Read and Preprocess the dataset 
df = pd.read_csv("processed/facial_features_original.csv")
emotion = df["expression"]
inputs = df.drop(["subDirectory_filePath", "expression"], axis=1)

#Do a balanced split of the dataset for train/val/test: We decided to split into 70/20/10:

#Training and Validation/Test: 90/10
X_train_val, X_test, y_train_val, y_test = train_test_split(
    inputs,
    emotion,
    test_size=0.1,
    stratify=emotion,
    random_state=42)

#Training/Validation: 70/20
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=(0.2/0.9), #20% of original data
    stratify=y_train_val,
    random_state=42)
print("\nSplit of the data: ", len(X_train), len(X_val),len(X_test), "\n")

#Our different models and configurations that will be tested
models_and_params = [
    ("KNeighborsClassifier", KNeighborsClassifier(), {
        "n_neighbors": [15,19,20,21,22,23,24],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
    }),
    ("SVC", SVC(), {
        "C": [5,10,15,20,25],
        "kernel": ["poly", "rbf", "linear", "sigmoid"],
        "degree": [1,5,10,15],
        "gamma": ["scale", "auto"]
    }),
    ("RandomForestClassifier", RandomForestClassifier(), {
        "n_estimators": [50,100,150,200],
        "max_features": ["sqrt", "log2"],
        "max_depth": [4, 5, 6, 7, 8],
        "criterion": ["gini", "entropy"],
    }),
    ("DecisionTreeClassifier", DecisionTreeClassifier(), {
        "criterion": ["gini", "entropy"],
        "max_depth": [1,2,3,4,5,6,7,8,10],
    }),
    ("LogisticRegression", LogisticRegression(), {
        "penalty": ["l1", "l2"],
        "C": [1,2,3,4,5,6,7,8,10],
        "solver": ["liblinear"]
    }),
    ("GradientBoostingClassifier", GradientBoostingClassifier(), {
        "max_depth": [2,3,4,5,10,20,25],
        "max_features": ["log2", "sqrt"],
    })
]

best_model = {}

#Hyperparameter tuning and model selection
#OBS: When using GridSearchCV, we split the training/validation data internally and can use all the data from the training/validation datasets
for model_name, model, param_grid in models_and_params:
    
    current_model = GridSearchCV(model, param_grid)
    current_model.fit(X_train_val, y_train_val) 

    test_accuracy = accuracy_score(y_test, current_model.predict(X_test))
    print(f"\n{model_name} - Best parameters: {current_model.best_params_}, Test accuracy: {test_accuracy}")
    
    #Save the best model into best_model
    if not best_model or test_accuracy > best_model["best_accuracy"]:
        best_model = {
            "model_name": model_name,
            "model": current_model.best_estimator_,
            "best_params": current_model.best_params_,
            "best_accuracy": test_accuracy
        }

#Print the best model, the best parameters and the accuracy
print(f"Best model: {best_model["model_name"]} with best parameters: {best_model["best_params"]} and accuracy: {best_model["best_accuracy"]}")

# Save the best model to a file
joblib.dump(best_model, "scripts/subSystem1/models/emotion_model.pkl")
