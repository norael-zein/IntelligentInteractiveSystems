""" Mindfulness coach app designed for Vetrans and service members on google play store
 has good info about mindfulness generally as well as a free seated practice
 which would be a good rubric to start with.
"""

#imports
#generic
import os, sys, time, random
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

def state_main():
    """
    updated driver function for parallel state tracks.
    """
    unhappy_exercises = ["body","awareness","breathing"]        # If unhappy, suggest calming exercises to ground the user.
    happy_exercises = ["gratitude", "visualization", "eating"]  # If happy, suggest "happy" exercises.
    
    furhat = FurhatRemoteAPI("localhost")
    
    apiKey = get_key()              # Use the environment variable to access api key
    genai.configure(api_key=apiKey) # Load model using apikey, 
    persona = get_persona()         # Give prompt for persona as initial instructions
    model = genai.GenerativeModel("gemini-1.5-flash",
                                  system_instruction=persona)
    
    history = []                    # Init history, check emotional state for first exercise
    emotion, history = state.introduction(model, furhat, history)
    
    for i in range(3):              # Proceed over 3 exercises
        #history = []
        if emotion in ["sad","angry","fear","disgust","surprise"]: # Of Angry, Disgust, Fear, Happy, Neutral, Sad, Suprise
            exercise = unhappy_exercises[i]
            next_state = getattr(state, exercise)
            emotion, history = next_state(model, furhat, history)
        else:
            exercise = happy_exercises[i]
            next_state = getattr(state, exercise)
            emotion, history = next_state(model, furhat, history)
    
    state.outro(model, furhat, history)

def main():
    """
    main driver function
    """
    start_time = None
    last_action_start = float("inf")
    #state progression
    prog = iter(["start", "body", "awareness", "breathing", "reflection"])
    # if unhappy, suggest calming exercises to ground the user.
    unhappy_exercises = {"body","awareness","breathing"}
    # If happy, try an exercise which focuses on positive feelings about things in their life
    # like visualizing a positive outcome, or gratitude
    happy_exercises = {"gratitude","visualization","eating"}
    
    sys_state = next(prog)
    #furhat
    furhat = FurhatRemoteAPI("localhost")
    furhat.attend("closest")
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
    model_response = model.generate_content("Introduce yourself and ask if the user would like to begin practice.")
    furhat.say(text = model_response.text, blocking = True)
    history.append({"role":"model", "parts": [model_response.text]})

    #Interaction loop
    while True:
        #Read user input
        user_prompt = furhat.listen().message
  
        #Build Prompt
        history.append({"role": "user", "parts": [user_prompt]})

        #Send to model and receive response
        response = model.generate_content(history)
  
        #handle errors if they occur?
        if response.text == None or response.text == "":
            print("No response")
            continue
        
        #Handle if user wants to exit.
        if response.text.strip() == "[EXIT]" or sys_state == "reflection":
            system_instruction = state.end_state()
            history.append({"role": "user", "parts": [system_instruction]})
            response = model.generate_content(history)
            furhat.say(text = response.text, blocking = True)
            break
        #system state handling block, start
        if sys_state == "start" and response.text.strip() == "[START]":
        #if state reaches reflection, trigger wind down sequence
            sys_state = next(prog)
            start_time = last_action_start = time.time()
            get_prompt = getattr(state, sys_state)
            system_instruction = get_prompt()
            history.append({"role": "user", "parts": [system_instruction]})
            response = model.generate_content(history)
        #state progression every 30th second
        elif last_action_start - time.time() > 30:
            last_action_start = time.time()
            sys_state = next(prog)
            get_prompt = getattr(state, sys_state)
            system_instruction = get_prompt()
            history.append({"role": "user", "parts": [system_instruction]})
            response = model.generate_content(history)
        
        #output response
        furhat.say(text = response.text, blocking = True)
  
        #Add the response to history
        #history.append({"role":"user", "parts": [user_prompt]})
        history.append({"role":"model", "parts": [response.text]})
        #small pause for more natural progression
        

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
    Your goal is to provide calming guidance to help users 
    practice mindfulness meditation exercises.
    You are supposed to be good at helping people relax.
    limit each response to a minimum of 10 words, and a maximum of 100.
    Keep to the instructions given you receive after 'prompt: ' and do not give instructions beyond these prompts
    as the exercise instructions are externally managed.
    
    Inform the user when we start an exercise. 
    Exercise names are preceded and followed by two stars (e.g. **mindful sleeping**).
    
    Additionally, if the user indicates they want to exit the program (e.g., 
    "quit", "goodbye", "stop", "I want to leave", or similar), respond with the 
    *exact* text: [EXIT] and nothing else.
    
    If the user wishes to begin practice (e.g., "start", "begin", "I would 
    like to begin", or similar), respond with the *exact* text: [START] and nothing else.
    Wait for permission to begin the practice. 
    """

if __name__== "__main__":
    #main()
    state_main() #for parallel tracks
