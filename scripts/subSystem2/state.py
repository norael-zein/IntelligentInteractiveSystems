#imports
import time
import exercise, gesture
from subSystem1.bestModel import *
import random
from subSystem1.featureExtractor import FeatureExtractor

#Feature extractor used
extractor = FeatureExtractor()

def introduction(model, furhat, history):
    gesture.subtle_smile()
    intro_prompt = ["Introduce yourself and ask if the user would like to begin practice."]
    return state(model, furhat, history, prompt=intro_prompt, trig="[START]")

# raisin practice le mao
def eating(model, furhat, history):
    gesture.subtle_smile()
    eating_prompt = [
    'Now we will practice **mindful eating**. Begin by asking them to find a raisin or other small food item.',
    'Now ask them to observe its color, shape, texture and other details.',
    'Encourage them to feel the raisin with their hands. What texture do they feel? is it soft, hard, or sticky?',
    'They should then smell the raisin. What do they notice about the smell? is it strong, subtle?',
    'Encourage them to taste it. What do they notice about its flavors?',
    'Remind them to chew slowly, savoring its taste, and swallow.'
    ]
    return state(model, furhat, history, prompt=eating_prompt, dur=100)

def gratitude(model, furhat, history):
    gesture.close_eyes()
    gratitude_prompt = [
    'Now we will practice **gratitude**. For the current exercise, the user should take a moment to think of one thing that brings them joy.',
    'It could be a memory, a person, or even a small object.',
    'Encourage them to hold onto that thought for a moment and smile.'
    ]
    return state(model, furhat, history, prompt=gratitude_prompt)

def visualization(model, furhat, history):
    gesture.subtle_smile()
    visualization_prompt = [
    'Now we will practice our **visualization**. For the current exercise, the user should take a moment to imagine something they are looking forward to.',
    'Ask them to picture it vividly in their mind. What do they see, hear, and feel as it happens?',
    'Encourage them to hold onto that image and let it energize them for the day.'
    ]
    return state(model, furhat, history, prompt=visualization_prompt)

def body(model, furhat, history):
    gesture.close_eyes()
    body_prompt = [
    'Now we will practice **bodily relaxation**. For the current exercise, the user should first get comfortable in a sitting position.',
    'Their gaze should rest relaxed at a point in front of them.',
    'Their hands and body should be instructed to be relaxed.'
    ]
    return state(model, furhat, history, prompt=body_prompt)
  
def awareness(model, furhat, history):
    gesture.subtle_smile()
    awareness_prompt = [
    'Now we will practice **awareness of ones senses**. For the current practice, guide the user to take notice of the sensations around them.',
    'The user should bring non-judgemental awareness to their sensations, both within and outside the body.',
    'They should register the sensations, without holding onto any particular stimulus or emotion.'
    ]
    return state(model, furhat, history, awareness_prompt)

def breathing(model, furhat, history):
    gesture.deep_breath()
    breathing_prompt = [
    'Now we will practice **mindful breathing**. Now the user should bring attention to their breathing.',
    'Bring user attention to their breathing in their belly or chest.',
    'Simply paying attention to the rise and fall as they breath in and out.'
    ]
    return state(model, furhat, history, breathing_prompt)

def outro(model, furhat, history, early_stop = False):
    if early_stop == False:
        state(model, furhat, history, prompt=[reflection()])
    model_response = model.generate_content(end_state())
    furhat.say(text = model_response.text, blocking = True)
    
    
def state(model, furhat, history, prompt, dur = 30, trig = "[EXIT]"):
    """ loops over an interaction until the time is up
    based on a previous prompt appended to history input
    
    Args:
        model (_type_): llm model for response generation
        furhat (_type_): furhat instance for speaking and listening.
        history (list): conversation history.
        prompt (list): list of strings to be passed to gemini for guiding interactions.
        trigger (string): trigger word surrounded by brackets "[...]" for breaking out of interaction
        dur (int, optional): duration of state. Defaults to 30 (seconds).

    Returns:
        tuple: emotion (string) and history (list)
    """
    
    prompt_iter = iter(prompt)              # iterates through instructions to guide practice                             
    history.append({"role": "user", "parts": "prompt: "+next(prompt_iter, "")})
    model_response = model.generate_content(history).text
    furhat.say(text = model_response, blocking = True)
    history.append({"role": "model", "parts": [model_response]})
    
    start_time = time.time()
    while time.time()-start_time < dur:
        system_instruction = next(prompt_iter, "")
        
        user_prompt = furhat.listen().message               # Read user input, wait 5 seconds for response by default.
        random.choice([gesture.listen_nod_response, gesture.listen_smile_response])() #Randomize between two gestures every time
        history.append({"role": "user", "parts": [user_prompt]}) # Build Prompt
        if system_instruction != "":
            history.append({"role": "user", "parts": "prompt: "+system_instruction}) # If we have more instructions, add them
            
        response = model.generate_content(history).text          # Send to model and receive response
        if response.strip() == trig:                   # Break if request to exit triggered
            history.append({"role":"model", "parts": [response]})
            break
            
        furhat.say(text = response, blocking = True)   #output response
        history.append({"role":"model", "parts": [response]})

    emotion = best_model(extractor)  # Update emotion after each exercise
    print(f"Current feeling: {emotion[0]}")
    
    return emotion, history
    
def reflection():
    gesture.reflect()
    reflection_prompt = """
    The user has now finished practicing their mindfulness.
    Try to bring the user's awareness back to their surroundings.
    Encourage the user to share their reflections on the exercise if they would like. How do they feel?
    """
    return reflection_prompt

def end_state():
    gesture.big_smile()
    end_prompt = """
    The current exercise is now over.
    Congratulate the user for a job well done, and encourage them to take this sense of presence with them throughout their day.
    Welcome them to practice with you again in the future.
    Ask if they would like to continue, or if they would like to stop for now.
    """
    extractor.clean_up()  # Clean up after each exercise 
    return end_prompt


