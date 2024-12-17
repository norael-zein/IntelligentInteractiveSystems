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

"""
main driver
"""
def main():
    emotions = best_model()
    print("emotions:", emotions)
    
    #furhat
    furhat = FurhatRemoteAPI("localhost")
    
    # Use the environment variable to access api key
    apiKey = get_key()
    
    persona = """
    You are an experienced mental health professional.
    Your goal is to provide calming, understanding guidance to help users practice mindfulness meditation exercises.
    You are supposed to be good at helping people relax.
    Listen to and understand the users questions, writing answers with a calm understanding.
    limit each response to a minimum of 10 words, and a maximum of 100.
    """
    
    #load model using apikey, give prompt for persona as initial instructions
    genai.configure(api_key=apiKey)
    model = genai.GenerativeModel("gemini-1.5-flash",
                                  system_instruction=persona)
    
    #Initialize History
    history = []
    
    model_response = model.generate_content("introduce yourself").text
    furhat.say(text = model_response, blocking = True)

    #Interaction loop
    while True:
        #Read user input
        user_prompt = furhat.listen().message
  
        if user_prompt.lower() in ["quit", "exit"]:
            break
      
        #Build Prompt
        prompt_from_history = build_prompt_from_history(history, user_prompt)

        #Send to model and receive response
        response = model.generate_content(prompt_from_history)
  
        #Handle errors if there was no content
        if response.text == None or response.text == "":
            print("Model Error!")
            continue
      
        #output response
        furhat.say(text = response.text, blocking = True)
  
        #Add the message to history
        history.append({"role":"user", "parts": [user_prompt]})
        history.append({"role":"model", "parts": [response.text]})

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

def build_prompt_from_history(history, current_user_prompt):
      messages = history + [{"role": "user", "parts": [current_user_prompt]}]
      return messages # This now a prompt for Gemini.

if __name__== "__main__":
    main()
