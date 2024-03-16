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
    model = DQN.load("deepq_soccer")
    tStart = time.time()
    for error in np.arange(HParams.LEStart, HParams.LEEnd, HParams.LEAnnealingStep):
        print("Error:", np.round(error, 2))
        SuccessSum = 0
        for cnt in range(HParams.TotalExperimentEpisodesNumber):
            obs = env.reset()
            tra = WebotsPeroperties.AshkanTranslation.getSFVec3f()
            rot = WebotsPeroperties.AshkanRotation.getSFRotation()
            # print(-tra[0], tra[2], mod_angle((rot[3]*rot[1])+math.pi/2))
            met = error
            AE = error
            theta = random.uniform(-math.pi, math.pi)
            class_list = [1.0, -1.0]
            randomSide = random.choice(class_list)

            cT = math.sin(theta)
            sT = math.cos(theta)

            Code.Update(-tra[0] + met*cT, tra[2] + met*sT, mod_angle((rot[3]*rot[1])+math.pi/2 + randomSide*AE), -0.035, 0.0, 0.0, 0.0)
            BallPose = WebotsPeroperties.BallTranslation.getSFVec3f()
            Code.updateballFilters(-BallPose[0], BallPose[2])
            BestAction = Code.HeadUpdate()
            DesiredYaw = (YawEndPoint - ((BestAction) % YawNumbers) * YawStep) * math.pi / 180.0
            DesiredPitch = (PitchEndPoint - (int((BestAction) / YawNumbers)) * PitchStep) * math.pi / 180.0
            # print(BestAction, DesiredYaw*180.0/math.pi, DesiredPitch*180.0/math.pi)

            if (HParams.Active):
                for a in range(20):
                    action, _states = model.predict(obs)
                    obs, rewards, dones, info = env.step(action)
            else:
                WebotsPeroperties.NeckJoint.setPosition(-DesiredYaw)
                WebotsPeroperties.HeadJoint.setPosition(DesiredPitch)
                WebotsPeroperties.robot.step(10*WebotsPeroperties.timestep)

            NeckPose2 = WebotsPeroperties.NeckSensor.getValue()
            HeadPose2 = WebotsPeroperties.HeadSensor.getValue()
            Code.update_cam(NeckPose2, HeadPose2)
            Code.Update(-tra[0], tra[2], mod_angle((rot[3]*rot[1])+math.pi/2), -0.035, 0.0, 0.0, 0.0)
            Code.updateballFilters(-BallPose[0], BallPose[2])
            BestAction = Code.HeadUpdate()
            condition = Code.Status(BestAction, 1)
            CurrentERate = np.round(condition[2]/condition[3], 2) 
            SuccessSum += CurrentERate
            print(cnt, ":::", CurrentERate, np.round(SuccessSum/(cnt+1), 2))
            # f= open("./ExperimentActive.txt", "a+")
            f= open("./ExperimentPassive.txt", "a+")
            f.write(str(error))
            f.write(" , ")
            f.write(str(cnt))
            f.write(" , ")
            f.write(str(np.round(CurrentERate, 2)))
            f.write("\n")
            f.close()

            # action, _states = model.predict(obs)
            # obs, rewards, dones, info = env.step(action)
        tEnd = time.time()
        TotalTime = int(tEnd - tStart)
        print("Success rate :::", SuccessSum/HParams.TotalExperimentEpisodesNumber)
        print("------------------------------------------")
    print("total time :::", TotalTime, "seconds")
        


if (__name__ == '__main__'):
    main()
