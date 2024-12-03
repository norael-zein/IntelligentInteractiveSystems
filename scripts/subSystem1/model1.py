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

#Do a balanced split of the dataset for train/val/test: I decided to split into 70/20/10:
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

#KNeighborsClassifier() as model
def KN_classifier():
    #Hyperparameter search/tuning using the training and validation set
    param_grid = {
    'n_neighbors': [3, 5, 7, 10, 13, 16, 19, 22, 25],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }

    best_model = GridSearchCV(KNeighborsClassifier(), param_grid)
    best_model.fit(X_train_val, y_train_val)

    #Analyze performance on selected model with selected hyperparameters on test set
    print("\n\nPrediction on kNN with best parameters on test set:",accuracy_score(y_test, best_model.predict(X_test)))
    print("\nBest parameters: ",best_model.best_params_)

    return best_model

#SVC classification
def SVC_classification():
    param_grid = [
        {"kernel": ["poly"], "degree": [3, 15, 25, 50]},
        {"kernel": ["rbf", "linear", "sigmoid"]}
    ]
    best_model = GridSearchCV(SVC(), param_grid)
    best_model.fit(X_train_val, y_train_val) 

    print("\n\nBest model with best parameters on test set: ",
          accuracy_score(y_test, best_model.predict(X_test)))
    print("Best parameters of best model: ",best_model.best_params_)

    return best_model

#Random forest
def random_forest():
    param_grid = [
        {"n_estimators": [200, 500]},
        {"max_features": ['auto', 'sqrt', 'log2']},
        {"max_depth": [4,5,6,7,8]},
        {"criterion":['gini', 'entropy']}
    ]

    best_model = GridSearchCV(SVC(), param_grid)
    best_model.fit(X_train_val, y_train_val) 

    print("\n\nBest model with best parameters on test set: ",
          accuracy_score(y_test, best_model.predict(X_test)))
    print("Best parameters of best model: ",best_model.best_params_)

    return best_model

#Classify samples on test_to_submit.csv
#test_to_submit = pd.read_csv("dataset/test/test_to_submit.csv")
#predictions = best_model.predict(test_to_submit)
#print("\nPredictions on test_to_submit.csv:\n", predictions, "\n")

#File containing the classifications of the model of the samples in test_to_submit.csv
#with open('outputs', 'w') as f:
#    for emotion in predictions:
#        f.write(f"{emotion}\n")