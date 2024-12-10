import joblib

#Load information about the best model (from model1.py)
best_model_data = joblib.load("scripts/subSystem1/best_model.pkl")
print(f"Model Name: {best_model_data['model_name']}")
print(f"Model: {best_model_data['model']}")
print(f"Best Parameters: {best_model_data['best_params']}")
print(f"Best Accuracy: {best_model_data['best_accuracy']}")

