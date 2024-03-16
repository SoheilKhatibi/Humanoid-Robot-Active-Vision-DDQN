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
    # For loop on the models, because we should repeat and save results for all saved models
    # os.makedirs("./Directory/" + str(self.Episode) + "-" + str(self.TimeStep), exist_ok=True)
    f= open("./TrainingCourseOutput.txt","a+")
    for ModelNum in np.arange(HParams.FirstSaveModelTimeFrame, HParams.LastSaveModelTimeFrame, HParams.SaveModelTimestep):
        # Load corresponding model
        model = DQN.load("LastModels/deepq_soccer" + str(ModelNum + HParams.SaveModelTimestep - 1))
        
        SuccessSum = 0.0
        BallBenchmarkSum = 0.0
        BallLossSum = 0.0

        # For loop on experiment episodes, because we should repeat these episodes for many times and calculate their mean
        for ExpEpisodeNum in range(HParams.TotalExperimentEpisodesNumber):
            obs = env.reset()
            
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

            # Run the agent with current model for 20 timesteps
            BallLost = False
            Success = False
            StepsTakenToLoseBall = -1
            StepsTakenToSucceed = -1
            for a in range(20):
                action, _states = model.predict(obs)
                # action = np.random.randint(9)
                obs, rewards, dones, info = env.step(action)
                # print(obs.shape)
                
                # Recieve current status of the task. [0]: If ball is in FOV. [1]: If it is in the success condition. [2]: Number of observations currently in FOV. [3]: Number of observations that would be in FOV in best camera pose
                condition = Code.Status(BestAction, 1)
                if (condition[0] == 0.0):
                    print("BallLost")
                    StepsTakenToLoseBall = a + 1
                    BallLost = True
                    break
                elif (condition[1] == 1.0):
                    StepsTakenToSucceed = a + 1
                    Success = True
                    break
                # print(condition[2]/condition[3])

            # Read robot neck and head positions from Webots and update camera matrix with it
            NeckPose = WebotsPeroperties.NeckSensor.getValue()
            HeadPose = WebotsPeroperties.HeadSensor.getValue()
            Code.update_cam(NeckPose, HeadPose)

            # Recieve current status of the task. [0]: If ball is in FOV. [1]: If it is in the success condition. [2]: Number of observations currently in FOV. [3]: Number of observations that would be in FOV in best camera pose
            condition = Code.Status(BestAction, 1)
            # print(condition[2]/condition[3])
            if (BallLost):
                EpisodeBallScaleNum = 20
                CurrentORate = 0.0
                BallLossNum = StepsTakenToLoseBall
            elif (Success):
                EpisodeBallScaleNum = StepsTakenToSucceed
                CurrentORate = 1.0
                BallLossNum = 20
            else:
                EpisodeBallScaleNum = 20
                CurrentORate = condition[2]/condition[3]
                BallLossNum = 20

            SuccessSum += CurrentORate
            BallBenchmarkSum += EpisodeBallScaleNum
            BallLossSum += BallLossNum
            print("Exp episode", ExpEpisodeNum, "is running", np.round(CurrentORate, 2), np.round(SuccessSum/(ExpEpisodeNum+1), 2))

            f= open("./TrainingCourseOutput.txt","a+")
            f.write(str(ModelNum + HParams.SaveModelTimestep - 1))
            f.write(" , ")
            f.write(str(ExpEpisodeNum))
            f.write(" , ")
            f.write(str(np.round(CurrentORate, 2)))
            f.write(" , ")
            f.write(str(StepsTakenToSucceed))
            f.write(" , ")
            f.write(str(StepsTakenToLoseBall))
            f.write("\n")
            f.close()
        
        print("ModelNum:", ModelNum + HParams.SaveModelTimestep - 1, "Success rate :::", SuccessSum/HParams.TotalExperimentEpisodesNumber)
        print("------------------------------------------")
        
        f= open("./ToPlot.txt","a+")
        f.write(str(ModelNum + HParams.SaveModelTimestep - 1))
        f.write(" , ")
        f.write(str(np.round(SuccessSum/HParams.TotalExperimentEpisodesNumber, 2)))
        f.write(" , ")
        f.write(str(np.round(BallBenchmarkSum/HParams.TotalExperimentEpisodesNumber, 2)))
        f.write(" , ")
        f.write(str(np.round(BallLossSum/HParams.TotalExperimentEpisodesNumber, 2)))
        f.write("\n")
        f.close()
        del model
    
    tEnd = time.time()
    TotalTime = int(tEnd - tStart)
    print("total time :::", TotalTime, "seconds")

if (__name__ == '__main__'):
    main()
