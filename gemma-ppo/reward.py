from pycubescrambler import nxn,side,non

# Python >3.9 compatibility
import collections.abc
#hyper needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping

from rubik_solver.Cubie import Cube
from rubik_solver import utils
from rubik_solver.Move import Move

# Plot histogram
import matplotlib.pyplot as plt
import numpy as np

import re

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

def reward_model_distance(cube, optimalSolution, model_output, solver):
    model_response = model_output[find_second_next_move(model_output):].strip()
    pattern = re.compile("([UDFBLR][2']?\s*)+")
    print("model_reponse:", model_response)
    matched_sequences = re.findall(pattern, model_response)
    print(matched_sequences)
    #Penalize by -10 for invalid output
    if len(matched_sequences) == 0 or not bool(pattern.fullmatch(matched_sequences[0])):
        return -10

    #Get default output when applying solver to cube that is already solved
    solvedSolution = utils.solve("yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww", 'Kociemba')
    solvedSolution = np.array(solvedSolution)
    #Apply move to cube
    print("move:", matched_sequences[0])
    for move in matched_sequences[0]:
        cube.move(Move(move))
    naive = cube.to_naive_cube()
    cubestr = naive.get_cube()
    #get new solution given new cube configuration
    newSln = utils.solve(cubestr, solver)
    #if given cube is already solved
    optimalSolution = optimalSolution.split(" ")
    if np.array_equal(newSln, solvedSolution):
        newSln = []
    #if cube after action is already solved
    if np.array_equal(optimalSolution, solvedSolution):
        optimalSolution = []
    reward = len(optimalSolution) - len(newSln)
    print(optimalSolution, len(optimalSolution))
    print(newSln, len(newSln))
    print(reward)
    return reward

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
    