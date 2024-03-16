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
import os


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
    
    # Load model
    model = DQN.load("LastModels/deepq_soccer")
    tStart = time.time()

    with open("Poses.txt", "r") as TheFile:
        ExpEpisodeNum = 0
        # For loop on experiment episodes, because we should repeat these episodes for many times and calculate their mean
        for line in TheFile:
            a = line.split(",")
            Poses = {
                "RandomX": np.round(float(a[0]), 2),
                "RandomY": np.round(float(a[1]), 2),
                "RandomA": np.round(float(a[2]), 2),
                "RandomYaw": np.round(float(a[3]), 2),
                "RandomPitch": np.round(float(a[4]), 2),
                "point_in_polyX": np.round(float(a[5]), 2),
                "point_in_polyY": np.round(float(a[6]), 2)
            }
            obs, RGBObs = env.ResetFromText(Poses)
            os.makedirs("./Directory/" + str(ExpEpisodeNum), exist_ok=True)
            
            # Read robot pose from Webots and set it in world model
            tra = WebotsPeroperties.AshkanTranslation.getSFVec3f()
            rot = WebotsPeroperties.AshkanRotation.getSFRotation()
            Code.Update(-tra[0], tra[2], mod_angle((rot[3]*rot[1])+math.pi/2), -0.035, 0.0, 0.0, 0.0)

            # Get ball pose from Webots and set it in world model
            BallPose = WebotsPeroperties.BallTranslation.getSFVec3f()
            Code.updateballFilters(-BallPose[0], BallPose[2])

            # Calculate best action from cpp code
            BestAction = Code.HeadUpdate()
            DesiredYaw = (YawEndPoint - ((BestAction) % YawNumbers) * YawStep) * math.pi / 180.0
            DesiredPitch = (PitchEndPoint - (int((BestAction) / YawNumbers)) * PitchStep) * math.pi / 180.0
            # print(BestAction, DesiredYaw*180.0/math.pi, DesiredPitch*180.0/math.pi)


            cv2.imwrite("./Directory/" + str(ExpEpisodeNum) + "/S" + str(0) + ".png", cv2.resize(RGBObs, (640, 480)))
            condition = Code.Status(BestAction, 1)
            Code.SaveLocalization(ExpEpisodeNum, 0)

            # Run the agent with current model for 20 timesteps
            for a in range(20):
                action, _states = model.predict(obs)
                # action = np.random.randint(9)
                obs, rewards, dones, info, RGBObs = env.step(action)
                
                f= open("./Directory/" + str(ExpEpisodeNum) + "/info" + str(a) + ".txt","w+")
                f.write(str(action))
                f.write("\n")
                f.close()

                # print("----------------------------------------------")
                # cv2.imshow("obs2", cv2.resize(RGBObs, (640, 480)))
                # cv2.waitKey(20)
                cv2.imwrite("./Directory/" + str(ExpEpisodeNum) + "/S" + str(a+1) + ".png", cv2.resize(RGBObs, (640, 480)))
                # print("----------------------------------------------2222")

                # print(obs.shape)
                
                # Recieve current status of the task. [0]: If ball is in FOV. [1]: If it is in the success condition. [2]: Number of observations currently in FOV. [3]: Number of observations that would be in FOV in best camera pose
                condition = Code.Status(BestAction, 1)
                Code.SaveLocalization(ExpEpisodeNum, a+1)
                # print(condition[2]/condition[3])

            # WebotsPeroperties.NeckJoint.setPosition(-DesiredYaw)
            # WebotsPeroperties.HeadJoint.setPosition(DesiredPitch)
            # WebotsPeroperties.robot.step(10*WebotsPeroperties.timestep)
            # NeckPose = WebotsPeroperties.NeckSensor.getValue()
            # HeadPose = WebotsPeroperties.HeadSensor.getValue()
            # Code.update_cam(NeckPose, HeadPose)
            # condition = Code.Status(BestAction, 1)
            # a = WebotsPeroperties.Camera.getImageArray()
            # b = np.array(a, dtype=np.uint8)
            # obs = cv2.rotate(cv2.flip(np.array(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), dtype=np.uint8), 1), cv2.ROTATE_90_COUNTERCLOCKWISE)
            # RGBObs = cv2.rotate(cv2.flip(np.array(cv2.cvtColor(b, cv2.COLOR_RGB2BGR), dtype=np.uint8), 1), cv2.ROTATE_90_COUNTERCLOCKWISE)
            # a = 1000
            # print("./Directory/" + str(ExpEpisodeNum) + "/S" + str(a+1) + ".png")
            # cv2.imwrite("./Directory/" + str(ExpEpisodeNum) + "/S" + str(a+1) + ".png", cv2.resize(RGBObs, (640, 480)))
            # Code.SaveLocalization(ExpEpisodeNum, a+1)

            # Read robot neck and head positions from Webots and update camera matrix with it
            NeckPose = WebotsPeroperties.NeckSensor.getValue()
            HeadPose = WebotsPeroperties.HeadSensor.getValue()
            Code.update_cam(NeckPose, HeadPose)

            # Recieve current status of the task. [0]: If ball is in FOV. [1]: If it is in the success condition. [2]: Number of observations currently in FOV. [3]: Number of observations that would be in FOV in best camera pose
            condition = Code.Status(BestAction, 1)
            # print(condition[2]/condition[3])

            CurrentORate = condition[2]/condition[3]
            
            print("Exp episode", ExpEpisodeNum, "is running", np.round(CurrentORate, 2))
            ExpEpisodeNum += 1
    
    print("------------------------------------------")
    
    tEnd = time.time()
    TotalTime = int(tEnd - tStart)
    print("total time :::", TotalTime, "seconds")

if (__name__ == '__main__'):
    main()
