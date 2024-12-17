#imports
import exercise
from subSystem1.best_model import *

class States:
    
    def __init__(self):
        pass
    
    #States define both atomic parts of exercises, and useful helper functions to smooth out interaction.
    def ready_check(furhat, model, exercise, context_prompt):
        response = model.generate_content(context_prompt+"Introduce yourself")
        furhat.say(""+ exercise + ". ")
        #if furhat.  
    

    def wait_response(furhat):
        """
        not sure if I need this either
        Args:
            furhat (_type_): furhat instance at play.

        Returns:
            _type_: _description_
        """
        while True:
            response = furhat.listen().message
            if response != "": return response
    
    def monitor_emotion(model, feature_extractor, prompt, time_limit):
        #not sure what I would need this for yet if at all.
        while True:
            emotions = best_model()
        
        
    def body(model, feature_extractor):
        body_prompt = """
        For the current exercise, the user should first get comfortable in a sitting position.
        Their gaze should rest relaxed at a point in front of them.
        Their hands and body should be instructed to be relaxed.
        """
        return body_prompt
        #if monitor_emotion(model,feature_extractor,body_prep,time_limit) != 

    def awareness():
        awareness_prompt = """
        The user should bring non-judgemental awareness to their sensations, both within and outside the body.
        They should register the sensations, without holding onto any particular stimulus or emotion.
        """
        return awareness_prompt

    def breathing():
        breathing_prompt = """
        Now the user should bring attention to their breathing.
        Bring user attention to their breathing in their belly or chest.
        simply paying attention to the rise and fall as they breath in and out.
        """
        return breathing_prompt

    def reflection():
        reflection_prompt = """
        The user has now finished practicing their mindfulness.
        Try to bring the user's awareness back to their surroundings.
        Encourage the user to share their reflections on the exercise if they would like. How do they feel?
        """
        return reflection_prompt

    def end_state():
        end_prompt = """
        The current exercise is now over.
        Congratulate the user for a job well done, and encourage them to take this sense of presence with them throughout their day.
        Welcome them to practice with you again in the future.
        Ask if they would like to continue, or if they would like to stop for now.
        """
        return end_prompt