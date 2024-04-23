#!/usr/bin/env python
#coding: utf-8

# Modified from https://github.com/MeepMoop/py222
# AL - 2024-04-20

from __future__ import print_function
import numpy as np
import py222

hO = np.ones(729, dtype=int) * 12
hP = np.ones(117649, dtype=int) * 12

slns = []
moveStrs = {0: "U", 1: "U'", 2: "U2", 3: "R", 4: "R'", 5: "R2", 6: "F", 7: "F'", 8: "F2"}

# generate pruning table for the piece orientation states
def genOTable(s, d, lm=-3):
  index = py222.indexO(py222.getOP(s))
  if d < hO[index]:
    hO[index] = d
    for m in range(9):
      if int(m / 3) == int(lm / 3):
        continue
      genOTable(py222.doMove(s, m), d + 1, m)

# generate pruning table for the piece permutation states
def genPTable(s, d, lm=-3):
  index = py222.indexP(py222.getOP(s))
  if d < hP[index]:
    hP[index] = d
    for m in range(9):
      if int(m / 3) == int(lm / 3):
        continue
      genPTable(py222.doMove(s, m), d + 1, m)

# IDA* which prints all optimal solutions
def IDAStar(s, d, moves, lm=-3, verbose=False):
  if py222.isSolved(s):
    printMoves(moves, verbose=verbose)
    return True
  else:
    sOP = py222.getOP(s)
    if d > 0 and d >= hO[py222.indexO(sOP)] and d >= hP[py222.indexP(sOP)]:
      dOptimal = False
      for m in range(9):
        if int(m / 3) == int(lm / 3):
          continue
        newMoves = moves[:]; newMoves.append(m)
        solved = IDAStar(py222.doMove(s, m), d - 1, newMoves, m)
        if solved and not dOptimal:
          dOptimal = True
      if dOptimal:
        return True
  return False

# print a move sequence from an array of move indices
def printMoves(moves, verbose=False):
  global slns
  moveStr = ""
  for m in moves:
    moveStr += moveStrs[m] + " "
  if verbose:
    print(moveStr)
  slns.append(moveStr)

# solve a cube state
def solveCube(s, depth_lim=11, verbose=False):
  global slns
  slns = []
  # print cube state
  if verbose:
    py222.printCube(s)

  # FC-normalize stickers
  if verbose:
    print("normalizing stickers...")
  s = py222.normFC(s)

  # generate pruning tables
  if verbose:
    print("generating pruning tables...")
  genOTable(py222.initState(), 0)
  genPTable(py222.initState(), 0)

  # run IDA*
  if verbose:
    print("searching...")
  solved = False
  depth = 1
  while depth <= depth_lim and not solved:
    if verbose:
      print("depth {}".format(depth))
    solved = IDAStar(s, depth, [], verbose=verbose)
    depth += 1

  return slns

if __name__ == "__main__":
  # input some scrambled state
  s = py222.doAlgStr(py222.initState(), "R U2 R2 F2 R' F2 R F R")
  # solve cube
  solveCube(s)

