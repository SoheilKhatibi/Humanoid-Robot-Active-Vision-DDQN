import WebotsPeroperties
import tensorflow as tf
import HParams
from Agent import Agent
from Environment import Environment
import time
import math
import cv2
import numpy as np
import Code
import random
from stable_baselines.common.env_checker import check_env
from stable_baselines.deepq.policies import CnnPolicy, FeedForwardPolicy, register_policy
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, lstm
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

# t0 = time.time()
# env = Environment()
# while WebotsPeroperties.robot.step(WebotsPeroperties.timestep) != -1:
#     t = time.time()
#     if (t - t0 > 1):
#         env.Reposition()
#         t0 = t
#     pass

# def main():
#     env = Environment()
#     # It will check your custom environment and output additional warnings if needed
#     check_env(env)

def MyCNN(scaled_images, **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=5, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_2 = tf.nn.max_pool2d(layer_1, [2, 2], [2, 2], padding = "VALID")
    layer_3 = activ(conv(layer_2, 'c2', n_filters=64, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_4 = activ(conv(layer_3, 'c3', n_filters=64, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_5 = conv_to_fc(layer_4)
    return activ(linear(layer_5, 'fc', n_hidden=512, init_scale=np.sqrt(2)))

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs, cnn_extractor=MyCNN, feature_extraction="cnn")


def main():
    env = Environment()
    if (HParams.Gym):
        model = DQN(CustomPolicy,
                env,
                gamma=0.99,
                learning_rate=0.0005,
                buffer_size=1000000,
                exploration_fraction=0.85,
                exploration_final_eps=0.02,
                exploration_initial_eps=1.0,
                train_freq=1,
                batch_size=32,
                double_q=True,
                learning_starts=500,
                target_network_update_freq=10000,
                prioritized_replay=True,
                prioritized_replay_alpha=0.6,
                prioritized_replay_beta0=0.4,
                prioritized_replay_beta_iters=None,
                prioritized_replay_eps=1e-06,
                param_noise=False,
                verbose=1,
                tensorboard_log="./dqnlogtb/",
                _init_setup_model=True,
                policy_kwargs=None,
                full_tensorboard_log=True,
                seed=None)
        model.learn(total_timesteps=30000)
        model.save("deepq_soccer")
    else:
        Player = Agent()
        for _ in range(HParams.TotalEpisodes):
            Terminal = False
            obs = env.reset()
            i = 0
            while not Terminal:
                ts = time.time()
                Action = Player.PredictAction(obs)
                # Action = random.randrange(HParams.ActionSize)
                next_obs, Reward, Terminal, info = env.step(Action)
                WebotsPeroperties.robot.step(WebotsPeroperties.timestep)
                # print(i)
                # cv2.imshow("title", np.reshape(obs, (HParams.FrameHeight, HParams.FrameWidth)))
                # cv2.imshow("Ntitle", np.reshape(next_obs, (HParams.FrameHeight, HParams.FrameWidth)))
                # cv2.waitKey(0)
                # print(obs.shape)
                # print(Action, Actions[Action], Reward, Terminal)

                obs = Player.Update(obs, Action, Reward, Terminal, next_obs)
                te = time.time()
                i += 1
                # print("timeimage222222222222:", te - ts)


if (__name__ == '__main__'):
    main()
