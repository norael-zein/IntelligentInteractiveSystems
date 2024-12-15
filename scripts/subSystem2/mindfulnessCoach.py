""" Mindfulness coach app designed for Vetrans and service members on google play store
 has good info about mindfulness generally as well as a free seated practice
 which would be a good rubric to start with.
"""

#imports
#generic
import os
import sys
import pandas as pd
import pickle
#furhat
from furhat_remote_api import FurhatRemoteAPI
from furhat_remote_api import Gesture
#llm 
import google.generativeai as genai


# machine agnostic way of adding project folders to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),"..",".."))
script = os.path.abspath(os.path.join(os.path.dirname(__file__),"..")) #scripts
system1 = script + "/subSystem1"
system2 = script + "/subSystem2"
if project_root not in sys.path:
    sys.path.append(project_root)
if script not in sys.path:
    sys.path.append(script)
if system1 not in sys.path:
    sys.path.append(system1)
if system2 not in sys.path:
    sys.path.append(system2)

#package imports
import exercise, state
from subSystem1.best_model import *
import subSystem1.featureExtractor as fe

"""_summary_
main driver
"""
def main():
    emotions = best_model()
    print("emotions:", emotions)
    
    #furhat
    furhat = FurhatRemoteAPI("localhost")
    
    # Use the environment variable to access api key
    apiKey = get_key()
    
    #load model using apikey
    genai.configure(api_key=apiKey)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content("Introduce yourself")
    
    furhat.say(response, blocking = True)
    while True:
        #State progressions always start with introduction and preparation.
        request = state.introduction(furhat)
    
        #seated practice
        if request == "seated practice":
            exercise.seated_practice()
    


    
    
def get_emotion():
    """
    get model output prediction from current 
    """
    pass
    
    
def get_facialfeatures():
    pass



def get_key():
    """ 
    Gets api key from .bashrc if working in a virtual environment.
    first set up key in .bashrc following: https://ai.google.dev/gemini-api/docs/api-key
    """
    bashrc_path = os.path.expanduser('~/.bashrc')
    with open(bashrc_path) as f:
        for line in f:
            if 'export GEMINI_API_KEY' in line:
                _, value = line.split('=')
                os.environ['GEMINI_API_KEY'] = value.strip()
    return os.getenv('GEMINI_API_KEY')

if __name__== "__main__":
    main()
