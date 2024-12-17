""" Mindfulness coach app designed for Vetrans and service members on google play store
 has good info about mindfulness generally as well as a free seated practice
 which would be a good rubric to start with.
"""

#imports
#generic
import os, sys, time
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

def main():
    """
    main driver function
    """
    start_time = None
    last_action_start = None
    states = state.States()
    #state progression
    prog = iter(["start", "body", "awareness", "breathing", "reflection"])
    sys_state = next(prog)
    #furhat
    furhat = FurhatRemoteAPI("localhost")
    
    # Use the environment variable to access api key
    apiKey = get_key()
    
    #load model using apikey, give prompt for persona as initial instructions
    genai.configure(api_key=apiKey)
    persona = get_persona()
    
    model = genai.GenerativeModel("gemini-1.5-flash",
                                  system_instruction=persona)
    
    #Initialize chat history
    history = []
    
    #Introduction
    model_response = model.generate_content("introduce yourself").text
    furhat.say(text = model_response.text, blocking = True)
    history.append({"role":"model", "parts": [model_response.text]})

    #Interaction loop
    while True:
        #Read user input
        user_prompt = furhat.listen().message
  
        #Build Prompt
        messages = history.append({"role": "user", "parts": [user_prompt]})

        #Send to model and receive response
        response = model.generate_content(messages)
  
        #handle errors if they occur?
        if response.text == None or response.text == "":
            print("No response")
            continue
        
        #system start
        if sys_state is "start" and response.text is "[START]":
            sys_state = next(prog)
            start_time = last_action_start = time.time()
            system_instruction = state.body()
            history.append({"role": "system", "parts": [system_instruction]})
            response = model.generate_content(history)
        #state progression?  
        elif last_action_start - time.time() > 30:
            pass #continue here
            
            
        #Handle if user wants to exit.
        if response.text == "[EXIT]":
            system_instruction = state.end_state()
            history.append({"role": "system", "parts": [system_instruction]})
            response = model.generate_content(history)
            furhat.say(text = response.text, blocking = True)
            break
        
        #output response
        furhat.say(text = response.text, blocking = True)
  
        #Add the message to history
        #history.append({"role":"user", "parts": [user_prompt]})
        history.append({"role":"model", "parts": [response.text]})

def get_key():
    """ 
    Gets api key from .bashrc file if working in a virtual environment.
    first set up key in .bashrc following: https://ai.google.dev/gemini-api/docs/api-key
    """
    bashrc_path = os.path.expanduser('~/.bashrc')
    with open(bashrc_path) as f:
        for line in f:
            if 'export GEMINI_API_KEY' in line:
                _, value = line.split('=')
                os.environ['GEMINI_API_KEY'] = value.strip()
    return os.getenv('GEMINI_API_KEY')

def get_persona():
    return """
    You are an experienced mental health professional.
    Your goal is to provide calming, understanding guidance to help users 
    practice mindfulness meditation exercises.
    You are supposed to be good at helping people relax.
    Listen to and understand the users questions, writing answers with a calm understanding.
    limit each response to a minimum of 10 words, and a maximum of 100.
    
    Additionally, if the user indicates they want to exit the program (e.g., 
    "quit", "exit", "stop", "I want to leave", or similar), respond with the 
    *exact* text: [EXIT] and nothing else.
    
    If the user wishes to begin practice (e.g., "start", "begin", "I would 
    like to begin", or similar), respond with the *exact* text: [START] and nothing else.
    Wait for permission to begin the practice. 
    """

if __name__== "__main__":
    main()
