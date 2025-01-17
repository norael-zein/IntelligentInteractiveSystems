import joblib

def best_model(extractor):
    """
    This model facilitates integration between the User Perception subsystem
    and the Interaction subsystem by analyzing real-time user data to generate emotional states.

    INPUT: Valence, Arousal, Action Units (AUs) from real-time captured frames
    OUTPUT: Emotional states: Angry, Disgust, Fear, Happy, Neutral, Sad, Suprise
    """
    try:
        action_units = extractor.get_action_units()

        # Load model for predicting emotions
        emotion_model_data = joblib.load("scripts/subSystem1/models/emotion_model.pkl")

        # Load model for predicting valence
        valence_model_data = joblib.load("scripts/subSystem1/models/valence_model.pkl")

        # Load model for predicting arousal
        arousal_model_data = joblib.load("scripts/subSystem1/models/arousal_model.pkl")

        # Predict valence and arousal
        valence = valence_model_data["model"].predict(action_units)
        arousal = arousal_model_data["model"].predict(action_units)

        # Combine valence, arousal, and action units
        extended_action_units = action_units.copy()
        extended_action_units.insert(0, "valence", valence)  # Add valence at the start
        extended_action_units.insert(1, "arousal", arousal)  # Add arousal after valence

        # Predict emotion from valence, arousal and action units
        emotions = emotion_model_data["model"].predict(extended_action_units)
        return emotions
        
    # If image is not detected
    except ValueError:
        print("Error (BestModel): No image detected")
        return None
