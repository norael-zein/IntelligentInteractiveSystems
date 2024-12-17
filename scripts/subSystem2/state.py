#imports
import exercise
from subSystem1.best_model import *

from datetime import datetime, time
from time import sleep

#States define both atomic parts of exercises, and useful helper functions to smooth out interaction.

def wait_response(furhat):
    """
    
    Args:
        furhat (_type_): furhat instance at play.

    Returns:
        _type_: _description_
    """
    while True:
        response = furhat.listen().message
        if response != "": return response
    
def monitor_emotion(model, feature_extractor):
    while True:
        emotions = best_model()
        
        
def body():
    body_prep = """
    For the current exercise, the user should first get comfortable in a sitting position.
    Their gaze should rest relaxed at a point in front of them.
    Their hands and body should be instructed to be relaxed.
    """

def awareness():
    awareness_prep = """
    The user should bring non-judgemental awareness to their sensations, both within and outside the body.
    They should register the sensations, without holding onto any
    """
    pass

def breathing():
    breathing_prep = """
    Now the user should bring attention to their breathing.
    Bring user attention to their breathing in their belly or chest.
    simply paying attention to the rise and fall as they breath in and out.
    """
    pass

def reflection():
    pass

def end_state():
    pass