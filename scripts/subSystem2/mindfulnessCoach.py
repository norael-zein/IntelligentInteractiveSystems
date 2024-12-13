""" Mindfulness coach app designed for Vetrans and service members on google play store
 has good info about mindfulness generally as well as a free seated practice
 which would be a good rubric to start with.
"""
#imports
from furhat_remote_api import FurhatRemoteAPI
import google.generativeai as genai
import os

def main():
    
    #furhat
    furhat = FurhatRemoteAPI("localhost")
    
    # Use the environment variable to access api key
    apiKey = get_key()
    
    #load model using apikey
    genai.configure(api_key=apiKey)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content("Explain how AI works")
    
    print(response.text)
    
    #State progressions always start with introduction and preparation.
    request = introduction()
    
    #seated practice
    if request == "seated practice":
        seated_practice()
    
    #birb. maybe make gemini make chicken noises, idk.
    return "birb"

def seated_practice():
    
    #will test with more information about how to effectively engineer prompts.
    promptHeaders = []
    
    breathing()
    
    reflection()
    
    end_state()

def introduction():
    """
    
    """
    pass

def breathing():
    pass

def reflection():
    pass

def end_state():
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
