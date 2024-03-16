import numpy as NP
import cv2
import HParams

def PreprocessFrame(obs):
    obs = cv2.resize(obs, (HParams.FrameWidth, HParams.FrameHeight))
    obs = NP.reshape(obs, (HParams.FrameHeight, HParams.FrameWidth, HParams.StackSize))
    return obs

