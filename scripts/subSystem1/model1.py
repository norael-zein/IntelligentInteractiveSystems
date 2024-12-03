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

#Read and Preprocess the dataset in a format that is appropriate for training
df = pd.read_csv("dataset/test/dataset.csv")
emotion = df["emotion"]
inputs = df.drop(labels="emotion", axis=1)

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

models_and_params = [
    ("KNN", KNeighborsClassifier(), {
        'n_neighbors': [15,19,20,21,22,23,24],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }),
    ("SVC", SVC(), {
        "kernel": ["poly", "rbf", "linear", "sigmoid"],
        "degree": [1,5,10,15]
    }),
    ("Random Forest", RandomForestClassifier(), {
        "n_estimators": [100,150,200,500],
        "max_features": ['sqrt', 'log2'],
        "max_depth": [4, 5, 6, 7, 8],
        "criterion": ['gini', 'entropy']
    }),
    ("Decision Tree", DecisionTreeClassifier(), {
        'criterion': ['gini', 'entropy'],
        'max_depth': [1,2,3,4,5,6,7,8,10,20,50,100]
    }),
    ("Logistic Regression", LogisticRegression(), {
        'penalty': ['l1', 'l2'],
        'C': [1,2,3,4,5,6,7,8,10,100,1000],
        'solver': ['liblinear']
    }),
    ("Gradient Boosting", GradientBoostingClassifier(), {
        "max_depth": [3,5,7,8,9,10,11,20,25],
        "max_features": ["log2", "sqrt"],
        "subsample": [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0]
    })
]

#When using GridSearchCV, we split the training/validation data internally and can use all the data from the training/validation datasets
for model_name, model, param_grid in models_and_params:
    
    current_model = GridSearchCV(model, param_grid)
    current_model.fit(X_train_val, y_train_val) 

    test_accuracy = accuracy_score(y_test, current_model.predict(X_test))
    print(f"\n{model_name} - Best parameters: {current_model.best_params_}, Test accuracy: {test_accuracy}")
    

#Classify samples on test_to_submit.csv
#test_to_submit = pd.read_csv("dataset/test/test_to_submit.csv")
#predictions = best_model.predict(test_to_submit)
#print("\nPredictions on test_to_submit.csv:\n", predictions, "\n")

#File containing the classifications of the model of the samples in test_to_submit.csv
#with open('outputs', 'w') as f:
#    for emotion in predictions:
#        f.write(f"{emotion}\n")