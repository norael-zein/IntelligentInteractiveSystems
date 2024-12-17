from furhat_remote_api import FurhatRemoteAPI
from furhat_remote_api import Gesture

# Create an instance of the FurhatRemoteAPI class, providing the address of the robot or the SDK running the virtual robot
furhat = FurhatRemoteAPI("localhost")


"""
Gestures made by Furhat during the interaction with the user 
"""
def deep_breath():
    pass

def reflect():
    pass

def close_eyes():
        furhat.gesture(body={
        "name": "CalmBreathing",
        "frames": [
            {
                "time": [0.5, 20],
                "persist": True,
                "params": {
                    "BLINK_LEFT": 1.0,  
                    "BLINK_RIGHT": 1.0,  
                    "SMILE_CLOSED": 0.2,  
                }
            },
            {
                "time": [0, 20],
                "persist": False,
                "params": {
                    "SMILE_CLOSED": 1

                }
            },
            {
                "time": [19, 20],
                "persist": False,
                "params": {
                    "reset": True  
                }
            }
        ],
        "class": "furhatos.gestures.Gesture"
    })

def surprise():
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

def sad():
    furhat.gesture(body={
        "name":"ExpressSad",
        "frames":[
            {
            "time":[0,2],
            "persist":False,
            "params":{
                "PHONE_OOH_Q": 0.2,
                "EYES_CLOSED": 1
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

def angry():
    furhat.gesture(name="BrowFrown")
    furhat.gesture(body={
        "name":"ExpressAnger",
        "frames":[
            {
            "time":[0,0.5],
            "persist":False,
            "params":{
                "EXPR_ANGER":0.2,
                "EXPR_DISGUST":0.1,
                "BROW_IN_RIGHT":0.3,
                "BROW_IN_LEFT": 0.3,
                "NECK_PAN": -5
                
                }
            },
            {
            "time":[0,2],
            "persist":False,
            "params":{
                "reset":True
                }
            }],
        "class":"furhatos.gestures.Gesture"
        })

def judge():
    furhat.gesture(body={
        "name":"BrowFrown",
        "frames":[
            {
            "time":[0,1],
            "persist":False,
            "params":{
                "EXPR_ANGER":0.2,
                "EXPR_DISGUST":0.1,
                "BROW_IN_RIGHT":0.5,
                "BROW_IN_LEFT": 0.5,
                "NECK_PAN": -2
                
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
    
def happy():
    furhat.gesture(body={
        "name":"BigSmile",
        "frames":[
            {
            "time":[0,1],
            "persist":False,
            "params":{
                "BROW_UP_LEFT":1,
                "BROW_UP_RIGHT":1,
                "SMILE_OPEN":0.5,
                "SMILE_CLOSED":0.7,
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
                "SMILE_CLOSED": 0.5
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

def listen_nod_response():
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
            "time":[0,1.5],
            "persist":False,
            "params":{
                "reset":True
                }
            }],
        "class":"furhatos.gestures.Gesture"
        })
    
def test():
    furhat.gesture(body={
        "name": "CalmBreathing",
        "frames": [
            {
                "time": [0.5, 20],
                "persist": True,
                "params": {
                    "BLINK_LEFT": 1.0,  
                    "BLINK_RIGHT": 1.0,  
                    "SMILE_CLOSED": 0.2,  
                }
            },
            {
                "time": [0, 20],
                "persist": False,
                "params": {
                    "SMILE_CLOSED": 1

                }
            },
            {
                "time": [19, 20],
                "persist": False,
                "params": {
                    "reset": True  
                }
            }
        ],
        "class": "furhatos.gestures.Gesture"
    })


    
print(test())