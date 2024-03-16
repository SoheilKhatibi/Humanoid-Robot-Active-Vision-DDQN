import WebotsPeroperties
import HParams
from Agent import Agent
from Environment import Environment
import time
import math
import cv2
import numpy as np
import Code
from controller import GPS
# t0 = time.time()
# env = Environment()
# while WebotsPeroperties.robot.step(WebotsPeroperties.timestep) != -1:
#     t = time.time()
#     if (t - t0 > 1):
#         env.Reposition()
#         t0 = t
#     pass


def main():
    Code.Entry()
    Code.set_camera_info(HParams.focalLength, HParams.focalBase, HParams.scaleA, HParams.scaleB, HParams.width, HParams.height, HParams.radialDK1, HParams.radialDK2, HParams.radialDK3, HParams.tangentialDP1, HParams.tangentialDP2, HParams.x_center, HParams.y_center)
    Camera = WebotsPeroperties.robot.getCamera("Camera")
    Camera.enable(WebotsPeroperties.timestep)
    Keyboard = WebotsPeroperties.Keyboard()
    Keyboard.enable(WebotsPeroperties.timestep)
    # HeadGPS = WebotsPeroperties.robot.getFromDef("HeadGPS")
    HeadGPS = GPS("gps")
    HeadGPS.enable(WebotsPeroperties.timestep)

    WebotsPeroperties.robot.step(WebotsPeroperties.timestep)
    # WebotsPeroperties.NeckJoint.setPosition(0)
    # WebotsPeroperties.HeadJoint.setPosition(45 * math.pi/180)
    while WebotsPeroperties.robot.step(WebotsPeroperties.timestep) != -1:
        key = Keyboard.getKey()
        NeckPose = WebotsPeroperties.NeckSensor.getValue()
        HeadPose = WebotsPeroperties.HeadSensor.getValue()
        if key > 0:
            # print(key)
            if key == 87:
                HeadPose = HeadPose - 1*math.pi/180
            elif key == 68:
                NeckPose = NeckPose + 1*math.pi/180
            elif key == 88:
                HeadPose = HeadPose + 1*math.pi/180
            elif key == 65:
                NeckPose = NeckPose - 1*math.pi/180
            elif key == 83:
                NeckPose = 0
                HeadPose = 0
            print(np.round(NeckPose*180/math.pi, 2), np.round(HeadPose*180/math.pi, 2))
            WebotsPeroperties.NeckJoint.setPosition(NeckPose)
            WebotsPeroperties.HeadJoint.setPosition(HeadPose)
        # a = HeadGPS.getValues()
        # print(a)
        tra = WebotsPeroperties.AshkanTranslation.getSFVec3f()
        rot = WebotsPeroperties.AshkanRotation.getSFRotation()
        Code.Update(-tra[0], tra[2], (rot[3] * rot[1]) + math.pi/2, -0.035, 0.0, 0.0, 0.0)
        # print("qqqqqqqqqqqq")
        Code.update_cam(NeckPose, HeadPose)
        # print("eeeeeeeeeeee")
        # print(tra, rot)
        Code.HeadUpdate()
        # a = Camera.getImageArray()
        # obs = cv2.cvtColor(cv2.rotate(cv2.flip(np.array(Camera.getImageArray(), dtype=np.uint8), 1), cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.COLOR_RGB2BGR)
        # Code.get_image()

if (__name__ == '__main__'):
    main()
