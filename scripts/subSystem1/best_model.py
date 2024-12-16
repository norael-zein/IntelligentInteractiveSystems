import joblib
import featureExtractor as fe

def best_model():

    try:
        best_model_data = joblib.load("scripts/subSystem1/best_model.pkl")
        action_units = fe.FeatureExtractor().extract_action_units()
        emotions = best_model_data["model"].predict(action_units)
        return emotions
        
    except ValueError:
        print("No image detected")
        return None


