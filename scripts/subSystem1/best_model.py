import joblib

def best_model():
    best_model_data = joblib.load("scripts/subSystem1/best_model.pkl")

    return best_model_data['model']
