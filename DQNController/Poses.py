import HParams
from Agent import Agent
from Environment import Environment
import time
import math
import cv2
import numpy as np
import random
from stable_baselines import DQN
import WebotsPeroperties
import Code


# Active vision parameters
ObservationsNumber = 12
ActionsNumber = 40
YawNumbers = 10
PitchNumbers = 4

YawStartPoint = -90
YawEndPoint = 90
PitchStartPoint = 5
PitchEndPoint = 65

YawStep = (YawEndPoint - YawStartPoint) / (YawNumbers - 1)
PitchStep = (PitchEndPoint - PitchStartPoint) / (PitchNumbers - 1)
# Active vision parameters


Actions = [
    'NoAction',
    'Up',
    'UpRight',
    'Right',
    'DownRight',
    'Down',
    'DownLeft',
    'Left',
    'UpLeft'
]

def mod_angle(a):
    if (a<0):
        a = a + 2*math.pi
    
    a = np.mod(a, (2*math.pi))
    
    if (a >= math.pi):
        a = a - 2*math.pi
    
    return a

def main():
    env = Environment()
    tStart = time.time()
    env.GenerateRandomPositions()

if (__name__ == '__main__'):
    main()
