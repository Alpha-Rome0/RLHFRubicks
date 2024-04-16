def find_second_next_move(s):
    """
    Finds the second instance of the phrase "Next move" in the given string.
    
    Args:
        s (str): The input string to search.
    
    Returns:
        int: The index of the second instance of "Next move", or -1 if not found.
    """
    # Find the first instance of "Next move"
    first_index = s.find("Next move")
    
    if first_index == -1:
        # "Next move" not found
        return -1
    
    # Find the second instance of "Next move"
    second_index = s.find("Next move", first_index + 1)
    
    return second_index

def reward_model_basic(correct_output, model_output):
    possible_rotations = ['R', 'L', 'U', 'D', "R'", "L'", "U'", "D'", "R2", "L2", "U2", "D2"]
    if any(possible_rotation in model_output for possible_rotation in possible_rotations):
        return 1
    return 0

def reward_model_strict(correct_output, model_output):
    # if correct_output == model_output.strip('<bos>')[0]:
    #     return 1
    return 0
    