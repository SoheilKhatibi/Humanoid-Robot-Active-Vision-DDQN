import HParams
from Agent import Agent
from Environment import Environment
import time
import math
import cv2
import numpy as np
import random
from stable_baselines import DQN

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

def main():
    env = Environment()
    if (HParams.Gym):
        model = DQN.load("deepq_soccer")
        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            # env.render()
    else:
        Player = Agent()
        Player.LoadNetwork('saved_networks/' + HParams.ModelNum + '/')
        for _ in range(HParams.TotalEpisodes):
            Terminal = False
            obs = env.reset()
            i = 0
            while not Terminal:
                ts = time.time()
                Action = Player.GetActionAtTest(obs)
                # Action = random.randrange(HParams.ActionSize)
                next_obs, Reward, Terminal, _ = env.step(Action)
                # print(i)
                print(Action, Actions[Action], Reward, Terminal)
                # cv2.imshow("title", cv2.resize(np.reshape(obs, (HParams.FrameHeight, HParams.FrameWidth)), (HParams.FrameWidth*4, HParams.FrameHeight*4)))
                # cv2.imshow("Ntitle", np.reshape(next_obs, (HParams.FrameHeight, HParams.FrameWidth)))
                # cv2.waitKey(0)
                i += 1
                # print(obs.shape)

                obs = next_obs
                te = time.time()
                # print("timeimage222222222222:", te - ts)
        


if (__name__ == '__main__'):
    main()
