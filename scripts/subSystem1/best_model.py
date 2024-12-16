import joblib
import featureExtractor as fe

def best_model():
    """
    Input - Action Units (AUs): AU01,AU02,AU04,AU05,AU06,AU07,AU09,AU10,AU11,AU12,AU14,AU15,AU17,AU20,AU23,AU24,AU25,AU26,AU28,AU43
    Output - Emotional states: Angry, Disgust, Fear, Happy, Neutral, Sad, Suprise
    """
    try:
        best_model_data = joblib.load("scripts/subSystem1/best_model.pkl")
        action_units = fe.FeatureExtractor().extract_action_units()
        emotions = best_model_data["model"].predict(action_units)
        return emotions
        

    except ValueError:
        print("No image detected")
        return None


