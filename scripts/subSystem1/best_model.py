import joblib
import featureExtractor as fe

def best_model():
    best_model_data = joblib.load("scripts/subSystem1/best_model.pkl")
    features = fe.FeatureExtractor.extract_action_units
    return best_model_data["model"], features

best_model() 