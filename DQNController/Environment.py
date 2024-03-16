import numpy as np
import os
from Utils import PreprocessFrame
import cv2
import math
import random
import time
import WebotsPeroperties
import Code
import HParams
from shapely.geometry import Polygon, Point
import gym
from gym import spaces

UpBanned = [1, 2, 8]
DownBanned = [4, 5, 6]
RightBanned = [2, 3, 4]
LeftBanned = [6, 7, 8]
FailureReward = -100
SuccessReward = 100
SuccessThreshold = 5
RegularStepReward = 0
ActionStep = 3 * math.pi / 180

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

BestAction = -1

def pose_global(pRelative, Pose):
    ca = math.cos(Pose[2])
    sa = math.sin(Pose[2])
    return [Pose[0] + ca*pRelative[0] - sa*pRelative[1], Pose[1] + sa*pRelative[0] + ca*pRelative[1], Pose[2] + pRelative[2]]

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]])
    
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                   [math.sin(theta[2]),    math.cos(theta[2]),     0],
                   [0,                     0,                      1]])
    
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

def get_random_point_in_polygon(poly, pose):
    field = Polygon([(4.5, 3.0), (4.5, -3.0), (-4.5, -3.0), (-4.5, 3.0)])
    minx, miny, maxx, maxy = poly.bounds
    while True:
        p = Point(random.uniform(minx, 4.5), random.uniform(miny, 4.5))
        Globalp = pose_global([p.x, p.y, 0], pose)
        # print(Globalp[0], Globalp[1])
        Globalp = Point(Globalp[0], Globalp[1])
        # print(poly.contains(p), field.contains(Globalp), (math.sqrt( (p.x)**2 + (p.y)**2 ) > 1.0))
        Distance = math.sqrt( (p.x)**2 + (p.y)**2 )
        if poly.contains(p) and field.contains(Globalp) and (Distance > 1.0) and (Distance < 3.0):
            return p

# RandomX = 0.0
# RandomY = 0.0
# RandomA = 0.0

class Environment(gym.Env):
    def __init__(self):
        super(Environment, self).__init__()
        self.action_space = spaces.Discrete(HParams.ActionSize)
        self.observation_space = spaces.Box(low=0, high=255, shape=(HParams.FrameHeight, HParams.FrameWidth, HParams.StackSize), dtype=np.uint8)
        WebotsPeroperties.Camera.enable(WebotsPeroperties.timestep)
        WebotsPeroperties.robot.step(WebotsPeroperties.timestep)
        Code.Entry()
        Code.set_camera_info(HParams.focalLength, HParams.focalBase, HParams.scaleA, HParams.scaleB, HParams.width, HParams.height, HParams.radialDK1, HParams.radialDK2, HParams.radialDK3, HParams.tangentialDP1, HParams.tangentialDP2, HParams.x_center, HParams.y_center)
        self.Horizon = 0
    
    def Reposition(self):
        # global RandomX, RandomY, RandomA
        # Randomly position the robot
        RandomX = random.uniform(-3, 3) # RandomX + 0.01
        RandomY = random.uniform(-1.5, 1.5) # RandomY + 0.01
        RandomA = random.uniform(-math.pi, math.pi) # RandomA + 0.01
        euler = [0.0 * math.pi/180.0, -20.0*math.pi/180.0, RandomA-math.pi]
        R = eulerAnglesToRotationMatrix(euler)
        theta = math.acos((R[0][0] + R[1][1] + R[2][2] - 1)/2)
        m = [(R[2][1]-R[1][2])*(1.0/2.0*math.sin(theta)), (R[0][2]-R[2][0])*(1.0/2.0*math.sin(theta)), (R[1][0]-R[0][1])*(1.0/2.0*math.sin(theta))]
        RandomRotation = []
        RandomRotation.append(m[0])
        RandomRotation.append(m[1])
        RandomRotation.append(m[2])
        RandomRotation.append(theta)
        Length = math.sqrt(math.pow(RandomRotation[0], 2) + math.pow(RandomRotation[1], 2) + math.pow(RandomRotation[2], 2))
        RandomRotation[0] = RandomRotation[0] / Length
        RandomRotation[1] = RandomRotation[1] / Length
        RandomRotation[2] = RandomRotation[2] / Length


        WebotsPeroperties.Ashkan.resetPhysics()
        WebotsPeroperties.robot.simulationResetPhysics()
        WebotsPeroperties.AshkanRotation.setSFRotation(RandomRotation)
        WebotsPeroperties.Ashkan.resetPhysics()
        WebotsPeroperties.robot.simulationResetPhysics()
        RandomPosition = []
        RandomPosition.append(RandomX)
        RandomPosition.append(RandomY)
        RandomPosition.append(0.411324)
        WebotsPeroperties.Ashkan.resetPhysics()
        WebotsPeroperties.robot.simulationResetPhysics()
        WebotsPeroperties.AshkanTranslation.setSFVec3f(RandomPosition)
        WebotsPeroperties.Ashkan.resetPhysics()
        WebotsPeroperties.robot.simulationResetPhysics()

        # RandomLocalBallX = random.uniform(0.5, 2)
        # RandomLocalBallY = random.uniform(-2, 2)
        # RobotPose = [RandomX, RandomY, RandomA]
        # BallLocal = [RandomLocalBallX, RandomLocalBallY, 0]
        # BallGlobal = pose_global(BallLocal, RobotPose)
        # ball = [-BallGlobal[0], 0.07, BallGlobal[1]]
        # WebotsPeroperties.Ball.resetPhysics()
        # WebotsPeroperties.robot.simulationResetPhysics()
        # WebotsPeroperties.BallTranslation.setSFVec3f(ball)
        # WebotsPeroperties.Ball.resetPhysics()
        # WebotsPeroperties.robot.simulationResetPhysics()

        Code.Update(RandomX, RandomY, RandomA, -0.035, 0.0, 0.0, 0.0)
        # Code.HeadUpdate(RandomX, RandomY, RandomA)

    def step(self, action):
        NeckPose = WebotsPeroperties.NeckSensor.getValue()
        HeadPose = WebotsPeroperties.HeadSensor.getValue()

        DesiredYaw = (YawEndPoint - ((BestAction) % YawNumbers) * YawStep) #* math.pi / 180.0
        DesiredPitch = (PitchEndPoint - (int((BestAction) / YawNumbers)) * PitchStep) #* math.pi / 180.0

        CurrentYaw = -(NeckPose * 180.0/math.pi)
        CurrentPitch = HeadPose * 180.0/math.pi
        Diff1 = math.sqrt( (abs(CurrentYaw-DesiredYaw))**2 + (abs(CurrentPitch-DesiredPitch))**2 )

        # print("Pose1:", np.round((NeckPose)*180/math.pi, 2), np.round((HeadPose)*180/math.pi, 2))

        if (action == 0):
            # print("No Action")
            yaw = NeckPose
            pitch = HeadPose
        elif (action == 1):
            # print("Up")
            yaw = NeckPose
            pitch = HeadPose - ActionStep
        elif (action == 2):
            # print("UpRight")
            yaw = NeckPose + (math.sqrt(2) / 2) * ActionStep
            pitch = HeadPose - (math.sqrt(2) / 2) * ActionStep
        elif (action == 3):
            # print("Right")
            yaw = NeckPose + ActionStep
            pitch = HeadPose
        elif (action == 4):
            # print("DownRight")
            yaw = NeckPose + (math.sqrt(2) / 2) * ActionStep
            pitch = HeadPose + (math.sqrt(2) / 2) * ActionStep
        elif (action == 5):  
            # print("Down")
            yaw = NeckPose
            pitch = HeadPose + ActionStep
        elif (action == 6):
            # print("DownLeft")
            yaw = NeckPose - (math.sqrt(2) / 2) * ActionStep
            pitch = HeadPose + (math.sqrt(2) / 2) * ActionStep
        elif (action == 7):
            # print("Left")
            yaw = NeckPose - ActionStep
            pitch = HeadPose
        elif (action == 8):
            # print("UpLeft")
            yaw = NeckPose - (math.sqrt(2) / 2) * ActionStep
            pitch = HeadPose - (math.sqrt(2) / 2) * ActionStep
        
        WebotsPeroperties.NeckJoint.setPosition(yaw)
        WebotsPeroperties.HeadJoint.setPosition(pitch)


        WebotsPeroperties.robot.step(10*WebotsPeroperties.timestep)
        self.Horizon += 1
        
        NeckPose2 = WebotsPeroperties.NeckSensor.getValue()
        HeadPose2 = WebotsPeroperties.HeadSensor.getValue()
        Code.update_cam(NeckPose2, HeadPose2)
        # print(BestAction)
        condition = Code.Status(BestAction, 1)
        BallPose = WebotsPeroperties.BallTranslation.getSFVec3f()
        Code.updateballFilters(BallPose[0], BallPose[1])
        # print("aaaaaa", BallIsInOnlineFOV)
        # print("Pose2:", np.round((NeckPose2)*180/math.pi, 2), np.round((HeadPose2)*180/math.pi, 2))
        # print("Pose3:", np.round((NeckPose2-NeckPose)*180/math.pi, 2), np.round((HeadPose2-HeadPose)*180/math.pi, 2))
        # print("---------------------------------------------------------------------------")

        # ts = time.time()
        # te = time.time()
        # print("timeimage:", te - ts)
        # print(b.shape)
        # a = 


        CurrentYaw = -(NeckPose2 * 180.0/math.pi)
        CurrentPitch = HeadPose2 * 180.0/math.pi
        Diff2 = math.sqrt( (abs(CurrentYaw-DesiredYaw))**2 + (abs(CurrentPitch-DesiredPitch))**2 )
        # print(DesiredYaw, DesiredPitch, np.round(CurrentYaw, 2), np.round(CurrentPitch, 2), np.round(Diff, 2))

        if (condition[0] == 0.0):
            reward = -2.0
            done = True
            is_success = False
        elif (condition[1] == 1.0):
            # elif (Diff2 < SuccessThreshold):
            reward = np.sign(int(Diff1 - Diff2))
            done = True
            is_success = True
        else:
            reward = np.sign(int(Diff1 - Diff2))
            done = False
            is_success = False
        
        a = WebotsPeroperties.Camera.getImageArray()
        b = np.array(a, dtype=np.uint8)
        obs = cv2.rotate(cv2.flip(np.array(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), dtype=np.uint8), 1), cv2.ROTATE_90_COUNTERCLOCKWISE)
        # RGBObs = cv2.rotate(cv2.flip(np.array(cv2.cvtColor(b, cv2.COLOR_RGB2BGR), dtype=np.uint8), 1), cv2.ROTATE_90_COUNTERCLOCKWISE)
        info = {
            'is_success': is_success,
            'Horizon': self.Horizon
        }

        # cv2.imshow("obs", obs)
        # cv2.waitKey(20)

        # return PreprocessFrame(obs), reward, done, info, RGBObs
        return PreprocessFrame(obs), reward, done, info

    def reset(self):
        global BestAction

        while True:
            # Random x, y, a for robot
            self.Reposition()

            # Random yaw, pitch for robot
            RandomYaw = random.uniform(-math.pi/2, math.pi/2)
            RandomPitch = random.uniform(5*math.pi/180, 25*math.pi/180)
            
            # Perform random yaw, pitch for robot
            WebotsPeroperties.NeckJoint.setPosition(RandomYaw)
            WebotsPeroperties.HeadJoint.setPosition(RandomPitch)
            WebotsPeroperties.robot.step(10*WebotsPeroperties.timestep)
    
            # Read head positions
            NeckPose = WebotsPeroperties.NeckSensor.getValue()
            HeadPose = WebotsPeroperties.HeadSensor.getValue()

            # Update head positions for camera matrix
            Code.update_cam(NeckPose, HeadPose)
            
            # Read Field of view points on the field local to the robot((x,y) of 4 points, 8 in total)
            a=Code.foo()
            # print(a)
            p = Polygon([(a[0], a[1]), (a[2], a[3]), (a[4], a[5]), (a[6], a[7])])

            # Read robot global pose
            pose = Code.pose()

            # Generate a random point in the field of view described in global coordinates
            point_in_poly = get_random_point_in_polygon(p, pose)
            # print(point_in_poly.x, point_in_poly.y)

            # Convert and perform the generated ball pose
            ballpose = pose_global([point_in_poly.x, point_in_poly.y, 0], pose)
            ball = [ballpose[0], ballpose[1], 0.07]
            WebotsPeroperties.Ball.resetPhysics()
            WebotsPeroperties.robot.simulationResetPhysics()
            WebotsPeroperties.BallTranslation.setSFVec3f(ball)
            WebotsPeroperties.Ball.resetPhysics()
            WebotsPeroperties.robot.simulationResetPhysics()
            WebotsPeroperties.robot.step(10*WebotsPeroperties.timestep)
            
            # Update new ball pose in cpp code
            Code.updateballFilters(ballpose[0], ballpose[1])
            
            # Calculate best action(number) based on new ball and robot position
            BestAction = Code.HeadUpdate()
            # print("Best Action:", BestAction)

            # Calculate yaw, pitch of best action using its number
            DesiredYaw = (YawEndPoint - ((BestAction) % YawNumbers) * YawStep) #* math.pi / 180.0
            DesiredPitch = (PitchEndPoint - (int((BestAction) / YawNumbers)) * PitchStep) #* math.pi / 180.0
            
            # Read head positions and update camera matrix
            NeckPose2 = WebotsPeroperties.NeckSensor.getValue()
            HeadPose2 = WebotsPeroperties.HeadSensor.getValue()
            Code.update_cam(NeckPose2, HeadPose2)

            # Determine conditions whether 0: The ball is out of FOV, and 1: The success is reached.
            condition = Code.Status(BestAction, 0)

            # Read ball pose from webots and set in cpp code
            BallPose = WebotsPeroperties.BallTranslation.getSFVec3f()
            Code.updateballFilters(BallPose[0], BallPose[1])
            
            CurrentYaw = -(NeckPose2 * 180.0/math.pi)
            CurrentPitch = HeadPose2 * 180.0/math.pi
            Diff2 = math.sqrt( (abs(CurrentYaw-DesiredYaw))**2 + (abs(CurrentPitch-DesiredPitch))**2 )
            # print(BestAction, DesiredYaw, DesiredPitch)
            # print(condition)
            if (condition[1] == 0.0):
                # if (Diff2 > SuccessThreshold):
                break

        a = WebotsPeroperties.Camera.getImageArray()

        obs = cv2.cvtColor(cv2.rotate(cv2.flip(np.array(a, dtype=np.uint8), 1), cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.COLOR_RGB2GRAY)
        # RGBObs = cv2.cvtColor(cv2.rotate(cv2.flip(np.array(a, dtype=np.uint8), 1), cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.COLOR_RGB2BGR)
        # cv2.imshow("obs", obs)
        # cv2.waitKey(20)
        self.Horizon = 0

        # return PreprocessFrame(obs), RGBObs
        return PreprocessFrame(obs)
    
    def ResetFromText(self, Poses):
        
        RandomX = Poses["RandomX"]
        RandomY = Poses["RandomY"]
        RandomA = Poses["RandomA"]
        
        euler = [0.0 * math.pi/180.0, 20.0*math.pi/180.0, RandomA-math.pi/2]
        R = eulerAnglesToRotationMatrix(euler)
        theta = math.acos((R[0][0] + R[1][1] + R[2][2] - 1)/2)
        m = [(R[2][1]-R[1][2])*(1.0/2.0*math.sin(theta)), (R[0][2]-R[2][0])*(1.0/2.0*math.sin(theta)), (R[1][0]-R[0][1])*(1.0/2.0*math.sin(theta))]
        RandomRotation = []
        RandomRotation.append(m[1])
        RandomRotation.append(m[2])
        RandomRotation.append(m[0])
        RandomRotation.append(theta)
        Length = math.sqrt(math.pow(RandomRotation[0], 2) + math.pow(RandomRotation[1], 2) + math.pow(RandomRotation[2], 2))
        RandomRotation[0] = RandomRotation[0] / Length
        RandomRotation[1] = RandomRotation[1] / Length
        RandomRotation[2] = RandomRotation[2] / Length

        WebotsPeroperties.Ashkan.resetPhysics()
        WebotsPeroperties.robot.simulationResetPhysics()
        WebotsPeroperties.AshkanRotation.setSFRotation(RandomRotation)
        WebotsPeroperties.Ashkan.resetPhysics()
        WebotsPeroperties.robot.simulationResetPhysics()
        RandomPosition = []
        RandomPosition.append(-RandomX)
        RandomPosition.append(0.411324)
        RandomPosition.append(RandomY)
        WebotsPeroperties.Ashkan.resetPhysics()
        WebotsPeroperties.robot.simulationResetPhysics()
        WebotsPeroperties.AshkanTranslation.setSFVec3f(RandomPosition)
        WebotsPeroperties.Ashkan.resetPhysics()
        WebotsPeroperties.robot.simulationResetPhysics()

        Code.Update(RandomX, RandomY, RandomA, -0.035, 0.0, 0.0, 0.0)
        
        # Random yaw, pitch for robot
        RandomYaw = Poses["RandomYaw"]
        RandomPitch = Poses["RandomPitch"]
        
        # Perform random yaw, pitch for robot
        WebotsPeroperties.NeckJoint.setPosition(RandomYaw)
        WebotsPeroperties.HeadJoint.setPosition(RandomPitch)
        WebotsPeroperties.robot.step(10*WebotsPeroperties.timestep)

        # Read head positions
        NeckPose = WebotsPeroperties.NeckSensor.getValue()
        HeadPose = WebotsPeroperties.HeadSensor.getValue()

        # Update head positions for camera matrix
        Code.update_cam(NeckPose, HeadPose)

        # Generate a random point in the field of view described in global coordinates
        point_in_poly = [Poses["point_in_polyX"], Poses["point_in_polyY"]]
        # print(point_in_poly.x, point_in_poly.y)

        # Read robot global pose
        pose = Code.pose()

        # Convert and perform the generated ball pose
        ballpose = pose_global([point_in_poly[0], point_in_poly[1], 0], pose)
        ball = [-ballpose[0], 0.07, ballpose[1]]
        WebotsPeroperties.Ball.resetPhysics()
        WebotsPeroperties.robot.simulationResetPhysics()
        WebotsPeroperties.BallTranslation.setSFVec3f(ball)
        WebotsPeroperties.Ball.resetPhysics()
        WebotsPeroperties.robot.simulationResetPhysics()
        WebotsPeroperties.robot.step(10*WebotsPeroperties.timestep)
        
        # Update new ball pose in cpp code
        Code.updateballFilters(ballpose[0], ballpose[1])

        
        a = WebotsPeroperties.Camera.getImageArray()
        obs = cv2.cvtColor(cv2.rotate(cv2.flip(np.array(a, dtype=np.uint8), 1), cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.COLOR_RGB2GRAY)
        # RGBObs = cv2.cvtColor(cv2.rotate(cv2.flip(np.array(a, dtype=np.uint8), 1), cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.COLOR_RGB2BGR)
        # cv2.imshow("obs", obs)
        # cv2.waitKey(20)
        self.Horizon = 0
        # return PreprocessFrame(obs), RGBObs
        return PreprocessFrame(obs)
    
    def GenerateRandomPositions(self):
        global BestAction

        for _ in range(100):
            while True:
                # Random x, y, a for robot
                # global RandomX, RandomY, RandomA
                # Randomly position the robot
                RandomX = random.uniform(-3, 3) # RandomX + 0.01
                RandomY = random.uniform(-1.5, 1.5) # RandomY + 0.01
                RandomA = random.uniform(-math.pi, math.pi) # RandomA + 0.01
                euler = [0.0 * math.pi/180.0, 20.0*math.pi/180.0, RandomA-math.pi/2]
                R = eulerAnglesToRotationMatrix(euler)
                theta = math.acos((R[0][0] + R[1][1] + R[2][2] - 1)/2)
                m = [(R[2][1]-R[1][2])*(1.0/2.0*math.sin(theta)), (R[0][2]-R[2][0])*(1.0/2.0*math.sin(theta)), (R[1][0]-R[0][1])*(1.0/2.0*math.sin(theta))]
                RandomRotation = []
                RandomRotation.append(m[1])
                RandomRotation.append(m[2])
                RandomRotation.append(m[0])
                RandomRotation.append(theta)
                Length = math.sqrt(math.pow(RandomRotation[0], 2) + math.pow(RandomRotation[1], 2) + math.pow(RandomRotation[2], 2))
                RandomRotation[0] = RandomRotation[0] / Length
                RandomRotation[1] = RandomRotation[1] / Length
                RandomRotation[2] = RandomRotation[2] / Length


                WebotsPeroperties.Ashkan.resetPhysics()
                WebotsPeroperties.robot.simulationResetPhysics()
                WebotsPeroperties.AshkanRotation.setSFRotation(RandomRotation)
                WebotsPeroperties.Ashkan.resetPhysics()
                WebotsPeroperties.robot.simulationResetPhysics()
                RandomPosition = []
                RandomPosition.append(-RandomX)
                RandomPosition.append(0.411324)
                RandomPosition.append(RandomY)
                WebotsPeroperties.Ashkan.resetPhysics()
                WebotsPeroperties.robot.simulationResetPhysics()
                WebotsPeroperties.AshkanTranslation.setSFVec3f(RandomPosition)
                WebotsPeroperties.Ashkan.resetPhysics()
                WebotsPeroperties.robot.simulationResetPhysics()

                Code.Update(RandomX, RandomY, RandomA, -0.035, 0.0, 0.0, 0.0)
                # Code.HeadUpdate(RandomX, RandomY, RandomA)

                # Random yaw, pitch for robot
                RandomYaw = random.uniform(-math.pi/2, math.pi/2)
                RandomPitch = random.uniform(5*math.pi/180, 25*math.pi/180)
                
                # Perform random yaw, pitch for robot
                WebotsPeroperties.NeckJoint.setPosition(RandomYaw)
                WebotsPeroperties.HeadJoint.setPosition(RandomPitch)
                WebotsPeroperties.robot.step(10*WebotsPeroperties.timestep)
        
                # Read head positions
                NeckPose = WebotsPeroperties.NeckSensor.getValue()
                HeadPose = WebotsPeroperties.HeadSensor.getValue()

                # Update head positions for camera matrix
                Code.update_cam(NeckPose, HeadPose)
                
                # Read Field of view points on the field local to the robot((x,y) of 4 points, 8 in total)
                a=Code.foo()
                # print(a)
                p = Polygon([(a[0], a[1]), (a[2], a[3]), (a[4], a[5]), (a[6], a[7])])

                # Read robot global pose
                pose = Code.pose()

                # Generate a random point in the field of view described in global coordinates
                point_in_poly = get_random_point_in_polygon(p, pose)
                # print(point_in_poly.x, point_in_poly.y)

                # Convert and perform the generated ball pose
                ballpose = pose_global([point_in_poly.x, point_in_poly.y, 0], pose)
                ball = [-ballpose[0], 0.07, ballpose[1]]
                WebotsPeroperties.Ball.resetPhysics()
                WebotsPeroperties.robot.simulationResetPhysics()
                WebotsPeroperties.BallTranslation.setSFVec3f(ball)
                WebotsPeroperties.Ball.resetPhysics()
                WebotsPeroperties.robot.simulationResetPhysics()
                WebotsPeroperties.robot.step(10*WebotsPeroperties.timestep)
                
                # Update new ball pose in cpp code
                Code.updateballFilters(ballpose[0], ballpose[1])
                
                # Calculate best action(number) based on new ball and robot position
                BestAction = Code.HeadUpdate()
                # print("Best Action:", BestAction)

                # Calculate yaw, pitch of best action using its number
                DesiredYaw = (YawEndPoint - ((BestAction) % YawNumbers) * YawStep) #* math.pi / 180.0
                DesiredPitch = (PitchEndPoint - (int((BestAction) / YawNumbers)) * PitchStep) #* math.pi / 180.0
                
                # Read head positions and update camera matrix
                NeckPose2 = WebotsPeroperties.NeckSensor.getValue()
                HeadPose2 = WebotsPeroperties.HeadSensor.getValue()
                Code.update_cam(NeckPose2, HeadPose2)

                # Determine conditions whether 0: The ball is out of FOV, and 1: The success is reached.
                condition = Code.Status(BestAction, 0)

                # Read ball pose from webots and set in cpp code
                BallPose = WebotsPeroperties.BallTranslation.getSFVec3f()
                Code.updateballFilters(BallPose[0], BallPose[1])
                
                CurrentYaw = -(NeckPose2 * 180.0/math.pi)
                CurrentPitch = HeadPose2 * 180.0/math.pi
                Diff2 = math.sqrt( (abs(CurrentYaw-DesiredYaw))**2 + (abs(CurrentPitch-DesiredPitch))**2 )
                # print(BestAction, DesiredYaw, DesiredPitch)
                # print(condition)
                if (condition[1] == 0.0):
                    # if (Diff2 > SuccessThreshold):
                    break
            
            f= open("./Poses.txt","a+")
            f.write(str(np.round(RandomX, 2)))
            f.write(" , ")
            f.write(str(np.round(RandomY, 2)))
            f.write(" , ")
            f.write(str(np.round(RandomA, 2)))
            f.write(" , ")
            f.write(str(np.round(RandomYaw, 2)))
            f.write(" , ")
            f.write(str(np.round(RandomPitch, 2)))
            f.write(" , ")
            f.write(str(np.round(point_in_poly.x, 2)))
            f.write(" , ")
            f.write(str(np.round(point_in_poly.y, 2)))
            f.write("\n")
            f.close()
        
