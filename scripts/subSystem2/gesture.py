from furhat_remote_api import FurhatRemoteAPI
from furhat_remote_api import Gesture
import time

# Create an instance of the FurhatRemoteAPI class, providing the address of the robot or the SDK running the virtual robot
furhat = FurhatRemoteAPI("localhost")


"""
Gestures made by Furhat during the interaction with the user 
"""
def deep_breath():
        """
        Furhat is taking a deep breath
        """
        furhat.gesture(body={
        "name": "ClosedEyes",
        "frames": [
            {
                "time": [0, 1],
                "persist": True,
                "params": {
                    "BLINK_LEFT": 1.0,  
                    "BLINK_RIGHT": 1.0,  
                    "SMILE_CLOSED": 0.5,  
                    
                }
            },
            {
                  "time": [3,4],
                  "persist": True,
                  "params": {
                        "PHONE_OOH_Q": 1
                  }


            },
            {
                "time": [6,7],
                "persist": False,
                "params": {
                    "reset": True  
                }
            }
        ],
        "class": "furhatos.gestures.Gesture"
    })

def reflect():
        """
        Furhat is reflecting
        """
        furhat.gesture(body={
        "name": "ClosedEyes",
        "frames": [
            {
                "time": [0, 1],
                "persist": True,
                "params": {
                    "LOOK_UP": 0.4,
                    "BROW_UP_RIGHT": 0.5,  
                    "BROW_UP_LEFT": 0.5,
                    "PHONE_W": 0.4
                }
            },
            {
                "time": [6,7],
                "persist": False,
                "params": {
                    "reset": True  
                }
            }
        ],
        "class": "furhatos.gestures.Gesture"
    })

def close_eyes():
    """
    Closed eyes 
    """
    furhat.gesture(body={
        "name": "ClosedEyes",
        "frames": [
            {
                "time": [0, 1],
                "persist": True,
                "params": {
                    "BLINK_LEFT": 1.0,  
                    "BLINK_RIGHT": 1.0,  
                    "SMILE_CLOSED": 0.5,  
                }
            },
            {
                "time": [6,7],
                "persist": False,
                "params": {
                    "reset": True  
                }
            }
        ],
        "class": "furhatos.gestures.Gesture"
    })
        
def subtle_smile():
    """
    Furhat gives a subtle smile
    """
    furhat.gesture(body={
        "name":"BigSmile",
        "frames":[
            {
            "time":[0,1],
            "persist":False,
            "params":{
                "BROW_UP_LEFT":0.8,
                "BROW_UP_RIGHT":0.8,
                "SMILE_CLOSED": 1,
                "EYE_SQUINT_LEFT": 0.5,
                "EYE_SQUINT_RIGHT": 0.5
                }
            },
            {
            "time":[0,3],
            "persist":False,
            "params":{
                "reset":True
                }
            }],
        "class":"furhatos.gestures.Gesture"
        })

def big_smile():
    """
    Furhat smiles widely
    """
    furhat.gesture(body={
        "name":"BigSmile",
        "frames":[
            {
            "time":[0,1],
            "persist":False,
            "params":{
                "BROW_UP_LEFT":1,
                "BROW_UP_RIGHT":1,
                "SMILE_OPEN": 1,
                "EYE_SQUINT_LEFT": 0.5,
                "EYE_SQUINT_RIGHT": 0.5
                }
            },
            {
            "time":[0,3],
            "persist":False,
            "params":{
                "reset":True
                }
            }],
        "class":"furhatos.gestures.Gesture"
        })

def listen_smile_response():
        """
        Furhat listens and smiles during a conversation
        """
        furhat.gesture(body={
        "name":"Smile",
        "frames":[
            {
            "time":[0,1],
            "persist":False,
            "params":{
                "NECK_ROLL": 2,
                "BROW_UP_LEFT": 1,
                "BROW_UP_RIGHT": 1,
                "SMILE_CLOSED": 0.6
                }
            },
            {
            "time":[0,4],
            "persist":False,
            "params":{
                "reset":True
                }
            }],
        "class":"furhatos.gestures.Gesture"
        })

def listen_nod_response():
    """
    Furhat listens and nods during a conversation
    """
    furhat.gesture(body={
        "name":"Nod",
        "frames":[
            {
            "time":[0,1],
            "persist":False,
            "params":{
                "SMILE_CLOSED": 1,
                "NECK_TILT": 1
                }
            },
            {
            "time":[0,3],
            "persist":False,
            "params":{
                "reset":True
                }
            }],
        "class":"furhatos.gestures.Gesture"
        })

def surprise():
    """
    Furhat gets surprised
    """
    furhat.gesture(body={
        "name":"Surprise",
        "frames":[
            {
            "time":[0,2],
            "persist":False,
            "params":{
                "SURPRISE": 0.5,
                "PHONE_OH": 0.1,
                "BROW_UP_LEFT": 0.8,
                "BROW_UP_RIGHT": 0.8
                }
            },
            {
            "time":[0,3],
            "persist":False,
            "params":{
                "reset":True
                }
            }],
        "class":"furhatos.gestures.Gesture"
        })   

