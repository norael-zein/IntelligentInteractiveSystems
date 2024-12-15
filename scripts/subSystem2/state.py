def wait_response(furhat):
    """
    
    Args:
        furhat (_type_): _description_

    Returns:
        _type_: _description_
    """

    while True:
        response = furhat.listen().message
        if response != "": return response
        
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