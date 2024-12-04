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
    
    # Use the environment variable
    apiKey = get_key()
    genai.configure(api_key=apiKey)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content("Explain how AI works")
    print(response.text)
    
    #will test with more information about how to effectively engineer prompts.
    promptHeaders = []
    
    #birb
    return "birb"

""" gets api key if working in a virtual environment.
first set up key in .bashrc following: https://ai.google.dev/gemini-api/docs/api-key
"""
def get_key():
    bashrc_path = os.path.expanduser('~/.bashrc')
    with open(bashrc_path) as f:
        for line in f:
            if 'export GEMINI_API_KEY' in line:
                _, value = line.split('=')
                os.environ['GEMINI_API_KEY'] = value.strip()
    return os.getenv('GEMINI_API_KEY')

if __name__== "__main__":
    main()
