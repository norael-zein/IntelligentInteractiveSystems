import joblib
import featureExtractor as fe

def best_model():
    best_model_data = joblib.load("scripts/subSystem1/best_model.pkl")
    action_units = fe.FeatureExtractor.extract_action_units
    return best_model_data["model"], best_model_data["best_params"], action_units

