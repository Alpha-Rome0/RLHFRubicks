def find_second_next_move(s):
    """
    Finds the second instance of the phrase "Next move" in the given string.
    
    Args:
        s (str): The input string to search.
    
    Returns:
        int: The index of the second instance of "Next move", or -1 if not found.
    """
    # Find the first instance of "Next move"
    first_index = s.find("Next move:")
    
    if first_index == -1:
        # "Next move" not found
        return -1
    
    # Find the second instance of "Next move"
    second_index = s.find("Next move:", first_index + 1)
    
    return second_index + 10

def reward_model_basic(correct_output, model_output):
    possible_rotations = ['R', 'L', 'U', 'D', "R'", "L'", "U'", "D'", "R2", "L2", "U2", "D2"]
    if any(possible_rotation in model_output for possible_rotation in possible_rotations):
        return 1
    return 0

def reward_model_strict(correct_output, model_output):
    possible_rotations = set(['R', 'L', 'U', 'D', "R'", "L'", "U'", "D'", "R2", "L2", "U2", "D2"])
    model_response = model_output[find_second_next_move(model_output):]
    print(model_response)
    # find the first rotation in the model response that is in the possible_rotations set
    selected_rotation = ""
    model_response = model_response.split()
    for rotation in model_response:
        if rotation in possible_rotations:
            selected_rotation = rotation
            break
    if selected_rotation == correct_output:
        return 1
    return -1
    
def reward_R(correct_output, model_output):
    model_response = model_output[find_second_next_move(model_output):]
    print(model_response)
    if 'R' in model_response:
        print('1')
        return 100    
    print('-1')
    return -1

def reward_U(correct_output, model_output):
    model_response = model_output[find_second_next_move(model_output):]
    print(model_response)
    if 'U' in model_response:
        print('1')
        return 1
    print('-1')
    return -1