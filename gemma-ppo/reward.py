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
    first_index = s.find("Next moves:")
    
    if first_index == -1:
        # "Next move" not found
        return -1
    
    # Find the second instance of "Next move"
    second_index = s.find("Next moves:", first_index + 1)
    
    return second_index + 10

def filter_valid_moves(raw_moves):
    # Define a set of valid moves
    valid_moves = {"U", "U'", "U2", "D", "D'", "D2", "F", "F'", "F2", "B", "B'", "B2", "L", "L'", "L2", "R", "R'", "R2"}

    if raw_moves.endswith('<eos>'):
        raw_moves = raw_moves[:-5].strip().lstrip()
        with open("output.txt", "a+") as file:
            file.write(f"raw moves: {raw_moves}\n")

    # Split the raw_moves input into a list of potential moves
    moves = raw_moves.split()

    # Create a list to store filtered valid moves
    filtered_moves = []

    # Check each move, add to filtered_moves if valid, otherwise return False
    for move in moves:
        if move in valid_moves:
            filtered_moves.append(move)
        else:
            return False  # Return False immediately if an invalid move is found

    return filtered_moves

def reward_model_basic(correct_output, model_output):
    possible_rotations = ['R', 'L', 'U', 'D', "R'", "L'", "U'", "D'", "R2", "L2", "U2", "D2"]
    if any(possible_rotation in model_output for possible_rotation in possible_rotations):
        return 1
    return 0

def reward_function(model_output, optimalSolution):
    optimal_moves = optimalSolution.split(" ")
    model_response = model_output[find_second_next_move(model_output)+1:].strip()
    with open("output.txt", "a+") as file:
        file.write(f"model response: {model_response}\n")
    # pattern_full = re.compile("^([UDFBLR][2']?\s*)+$")
    # pattern_moves = re.compile("[UDFBLR][2']?")
    # actual_moves = re.findall(pattern_moves, model_response)
    actual_moves = filter_valid_moves(model_response)
    with open("output.txt", "a+") as file:
        file.write(f"actual_moves: {actual_moves}\n")
    with open("output.txt", "a+") as file:
        file.write(f"optimal_moves: {optimal_moves}\n")
    if not actual_moves:# or not bool(re.fullmatch(pattern_full, model_response)):
        reward = -100
        with open("output.txt", "a+") as file:
            file.write(f"reward: {reward}\n")
        return reward
    reward = 0
    # Iterate over the minimum length of both lists to avoid index errors
    for actual, optimal in zip(actual_moves, optimal_moves):
        if actual == optimal:
            reward += 1
        else:
            break  # Stop counting at the first mismatch
    reward *= 10
    with open("output.txt", "a+") as file:
        file.write(f"reward: {reward}\n")
    return reward

def reward_model_distance(cube, optimalSolution, model_output, solver):
    numMovesToConsider = 5

    model_response = model_output[find_second_next_move(model_output):].strip()
    pattern_full = re.compile("^([UDFBLR][2']?\s*)+$")
    pattern_moves = re.compile("[UDFBLR][2']?")
    matched_sequences = re.findall(pattern_moves, model_response)
    with open("output.txt", "a+") as file:
        file.write(f"model response: {model_response}\n")
    with open("output.txt", "a+") as file:
        file.write(f"matched_sequences: {matched_sequences}\n")
    #Penalize by -10 for invalid output
    if len(matched_sequences) == 0:# or not bool(re.fullmatch(pattern_full, model_response)):
        return -100

    #Get default output when applying solver to cube that is already solved
    solvedSolution = utils.solve("yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww", solver)
    solvedSolution = np.array(solvedSolution)
    #Apply move to cube
    with open("output.txt", "a+") as file:
        file.write(f"moves: {matched_sequences[:numMovesToConsider]}\n")
    for move in matched_sequences[:numMovesToConsider]:
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
    reward *= 10
    with open("output.txt", "a+") as file:
        file.write(f"optimalSolution: {optimalSolution} {len(optimalSolution)}\n")
    with open("output.txt", "a+") as file:    
        file.write(f"newSln: {newSln} {len(newSln)}\n")
    with open("output.txt", "a+") as file:    
        file.write(f"reward: {reward}\n")
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
    