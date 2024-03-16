import numpy as np
import tensorflow as tf
import HParams
from collections import deque
import random
from DQN import DQNNet
import os
import cv2
import os
import time

class Agent:
    def __init__(self):
        self.ActionSize = HParams.ActionSize
        self.Epsilon = HParams.InitialEpsilon
        self.EpsilonAnnealingStep = (HParams.InitialEpsilon - HParams.FinalEpsilon) / (HParams.TotalEpisodes - 1000)
        self.TimeStep = 0

        self.TotalRewardInEveryEpisode = 0
        self.TotalMaxQInEveryEpisode = 0
        self.TotalLossInEveryEpisode = 0
        self.TotalDurationInEveryEpisode = 0
        
        self.Episode = 0

        self.ExperienceReplayMemory = deque()

        self.SourceNetwork = DQNNet("SourceNetwork")
        SourceNetworkWeights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="SourceNetwork")

        self.TargetNetwork = DQNNet("TargetNetwork")
        TargetNetworkWeights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="TargetNetwork")
        
        assert len(SourceNetworkWeights) == len(TargetNetworkWeights)

        with tf.name_scope("TargetNetworkUpdateOperation"):
            self.TargetNetworkUpdateOperation = [TargetVariables.assign(SourceVariables) for TargetVariables, SourceVariables in zip(TargetNetworkWeights, SourceNetworkWeights)]

        with tf.name_scope("Loss"):
            self.A = tf.placeholder(tf.int64, [None], name="ChosenAction")
            self.Y = tf.placeholder(tf.float32, [None], name="TargetNetworkOutputForChosenAction")
            AOneHot = tf.one_hot(self.A, self.ActionSize, 1.0, 0.0)
            QValue = tf.reduce_sum(tf.multiply(self.SourceNetwork.Output, AOneHot), reduction_indices=1)
            Error = tf.abs(self.Y - QValue)
            QuadraticPart = tf.clip_by_value(Error, 0.0, 1.0)
            LinearPart = Error - QuadraticPart
            self.Loss = tf.reduce_mean(0.5 * tf.square(QuadraticPart) + LinearPart)

        self.Optimizer = tf.train.RMSPropOptimizer(HParams.LearningRate, momentum = HParams.Momentum, epsilon = HParams.MinGrad)
        # self.GradientsUpdate = self.Optimizer.minimize(self.Loss, var_list=SourceNetworkWeights)

        self.Gradients = self.Optimizer.compute_gradients(self.Loss, var_list=SourceNetworkWeights)
        self.GradientsUpdate = self.Optimizer.apply_gradients(self.Gradients, name='GradientsUpdate')

        if HParams.PlotSourceNetworkGradientsAndVariables:
            with tf.variable_scope("Histograms"):
                for Gradient, Variable in self.Gradients:
                    if Gradient is not None:
                        tf.summary.histogram("Gradients/" + Variable.name, Gradient)
                        tf.summary.histogram("Variables/" + Variable.name, Variable)

        if HParams.PlotBothNetworkWeights:
            with tf.variable_scope("BothNetworkWeights"):
                for i in range(len(TargetNetworkWeights)):
                    with tf.variable_scope(str(i)):
                    # with tf.variable_scope("Source"):
                    # with tf.variable_scope("Target"):
                        tf.summary.histogram("SourceVariables/" + SourceNetworkWeights[i].name, SourceNetworkWeights[i])
                        tf.summary.histogram("TargetVariables/" + TargetNetworkWeights[i].name, TargetNetworkWeights[i])

        self.Sess = tf.compat.v1.InteractiveSession()
        self.Saver = tf.compat.v1.train.Saver(SourceNetworkWeights, max_to_keep = HParams.MaximumNumberOfModelsToKeep)
        self.SummaryPlaceholders, self.UpdateOps, self.SummaryOp = self.SetupSummary()
        self.HistogramSummary = tf.compat.v1.summary.merge_all(scope="Histograms")
        self.BothNetworkWeights = tf.compat.v1.summary.merge_all(scope="BothNetworkWeights")
        # self.SourceNetworkSum = tf.compat.v1.summary.merge_all(scope="Source")
        # self.TargetNetworkSum = tf.compat.v1.summary.merge_all(scope="Target")
        self.SummaryWriter = tf.compat.v1.summary.FileWriter(HParams.TensorboardDataSavePath, self.Sess.graph)

        if not os.path.exists(HParams.ModelSavePath):
            os.makedirs(HParams.ModelSavePath)

        self.Sess.run(tf.compat.v1.global_variables_initializer())
        
        self.Sess.run(self.TargetNetworkUpdateOperation)

    def SetupSummary(self):
        TotalRewardPerEpisode = tf.Variable(0.)
        AverageMaxQPerEpisode = tf.Variable(0.)
        DurationPerEpisode = tf.Variable(0.)
        AverageLossPerEpisode = tf.Variable(0.)
        with tf.variable_scope("Scalars"):
            tf.compat.v1.summary.scalar('Total Reward Per Episode', TotalRewardPerEpisode)
            tf.compat.v1.summary.scalar('Average Max Q Per Episode', AverageMaxQPerEpisode)
            tf.compat.v1.summary.scalar('Duration Per Episode', DurationPerEpisode)
            tf.compat.v1.summary.scalar('Average Loss Per Episode', AverageLossPerEpisode)
        SummaryVariables = [TotalRewardPerEpisode, AverageMaxQPerEpisode, DurationPerEpisode, AverageLossPerEpisode]
        SummaryPlaceholders = [tf.compat.v1.placeholder(tf.float32) for _ in range(len(SummaryVariables))]
        UpdateOps = [SummaryVariables[i].assign(SummaryPlaceholders[i]) for i in range(len(SummaryVariables))]
        SummaryOp = tf.compat.v1.summary.merge_all(scope="Scalars")
        return SummaryPlaceholders, UpdateOps, SummaryOp

    def PredictAction(self, State):
        if self.Epsilon >= random.random(): # or self.TimeStep < InitialReplaySize:
            Action = random.randrange(self.ActionSize)
        else:
            Action = np.argmax(self.SourceNetwork.Output.eval(feed_dict={self.SourceNetwork.InputState: [State]}))

        return Action
    
    def Update(self, obs, Action, Reward, Terminal, next_obs):
        # print(self.TimeStep)
        # next_obs = np.append(obs[:, :, 1:], next_obs, axis=2)

        # Record the contents of replay memory components
        if (HParams.SaveImages):
            os.makedirs("./Directory/" + str(self.Episode) + "-" + str(self.TimeStep), exist_ok=True)
            f= open("./Directory/" + str(self.Episode) + "-" + str(self.TimeStep) + "/info.txt","w+")
            f.write(str(Action))
            f.write("\n")
            f.write(str(Reward))
            f.write("\n")
            f.write(str(Terminal))
            f.close()
            # print(obs.shape, next_obs.shape)
            cv2.imwrite("./Directory/" + str(self.Episode) + "-" + str(self.TimeStep) + "/S.png", obs.reshape(HParams.FrameHeight, HParams.FrameWidth))
            cv2.imwrite("./Directory/" + str(self.Episode) + "-" + str(self.TimeStep) + "/S'.png", next_obs.reshape(HParams.FrameHeight, HParams.FrameWidth))
        # Record the contents of replay memory components

        self.ExperienceReplayMemory.append((obs, Action, Reward, next_obs, Terminal))
        if len(self.ExperienceReplayMemory) > HParams.MemorySize:
            self.ExperienceReplayMemory.popleft()

        if self.TimeStep >= HParams.PretrainLength:
            
            if self.TimeStep % HParams.MaxTau == 0:
                self.Sess.run(self.TargetNetworkUpdateOperation)

            if self.TimeStep % HParams.TrainInterval == 0:
                # ts = time.time()
                self.TrainNetwork()
                # te = time.time()
                # print("Network training time:", te - ts)

            

        self.TotalRewardInEveryEpisode += Reward
        self.TotalMaxQInEveryEpisode += np.max(self.SourceNetwork.Output.eval(feed_dict={self.SourceNetwork.InputState: [obs]}))
        self.TotalDurationInEveryEpisode += 1

        if Terminal:
            
            if self.TimeStep >= HParams.PretrainLength:

                if self.Epsilon > HParams.FinalEpsilon: # and self.TimeStep >= InitialReplaySize:
                    self.Epsilon -= self.EpsilonAnnealingStep

                if self.Episode % HParams.ModelSaveFrequency == 0:
                    self.Saver.save(self.Sess, HParams.ModelSavePath + str(self.Episode) + '/', global_step=self.TimeStep)
                    print('Successfully saved: ' + HParams.ModelSavePath + str(self.Episode) + '/')

                Info = [self.TotalRewardInEveryEpisode, self.TotalMaxQInEveryEpisode / float(self.TotalDurationInEveryEpisode),
                        self.TotalDurationInEveryEpisode, self.TotalLossInEveryEpisode / (float(self.TotalDurationInEveryEpisode) / float(HParams.TrainInterval))]

                for i in range(len(Info)):
                    self.Sess.run(self.UpdateOps[i], feed_dict={
                        self.SummaryPlaceholders[i]: float(Info[i])
                    })

                SummaryStr = self.Sess.run(self.SummaryOp)
                self.SummaryWriter.add_summary(SummaryStr, self.Episode + 1)
            
            Mode = 'Explore&Exploit'
            
            print('EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.0f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7}'.format(
                self.Episode + 1, self.TimeStep, self.TotalDurationInEveryEpisode, self.Epsilon,
                self.TotalRewardInEveryEpisode, self.TotalMaxQInEveryEpisode / float(self.TotalDurationInEveryEpisode),
                self.TotalLossInEveryEpisode / (float(self.TotalDurationInEveryEpisode) / float(HParams.TrainInterval)), Mode))

            self.TotalRewardInEveryEpisode = 0
            self.TotalMaxQInEveryEpisode = 0
            self.TotalLossInEveryEpisode = 0
            self.TotalDurationInEveryEpisode = 0
            self.Episode += 1

        self.TimeStep += 1

        return next_obs
    
    def TrainNetwork(self):
        StateBatch = []
        ActionBatch = []
        RewardBatch = []
        NextStateBatch = []
        TerminalBatch = []
        YBatch = []

        Minibatch = random.sample(self.ExperienceReplayMemory, HParams.BatchSize)
        for Data in Minibatch:
            StateBatch.append(Data[0])
            ActionBatch.append(Data[1])
            RewardBatch.append(Data[2])
            NextStateBatch.append(Data[3])
            TerminalBatch.append(Data[4])

        
        TerminalBatch = np.array(TerminalBatch) + 0

        TargetQValuesBatch = self.TargetNetwork.Output.eval(feed_dict={self.TargetNetwork.InputState: NextStateBatch})
        YBatch = RewardBatch + (1 - TerminalBatch) * HParams.Gamma * np.max(TargetQValuesBatch, axis=1)
        # print(np.array(YBatch).shape, np.array(RewardBatch).shape, np.array(TerminalBatch).shape, np.max(TargetQValuesBatch, axis=1).shape)

        if HParams.PlotBothNetworkWeights:
            BothNetworkWeights = self.Sess.run(self.BothNetworkWeights)
            self.SummaryWriter.add_summary(BothNetworkWeights, self.TimeStep)
            # if self.TimeStep % MaxTau == 0:
                # TargetNetworkSum = self.Sess.run(self.TargetNetworkSum)
                # self.SummaryWriter.add_summary(TargetNetworkSum, self.TimeStep)
                # print("It is now done!!! :",self.TimeStep)

        Loss, _ = self.Sess.run([self.Loss, self.GradientsUpdate], feed_dict={
            self.SourceNetwork.InputState: StateBatch,
            self.A: ActionBatch,
            self.Y: YBatch
        })
        
        if HParams.PlotSourceNetworkGradientsAndVariables:
            HistogramSummary = self.Sess.run(self.HistogramSummary, feed_dict={
                self.SourceNetwork.InputState: StateBatch,
                self.A: ActionBatch,
                self.Y: YBatch
            })
            self.SummaryWriter.add_summary(HistogramSummary, self.TimeStep + 1)


        self.TotalLossInEveryEpisode += Loss

    def LoadNetwork(self, NetName):
        Checkpoint = tf.train.get_checkpoint_state(NetName)
        print(NetName)
        if Checkpoint and Checkpoint.model_checkpoint_path:
            self.Saver.restore(self.Sess, Checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + Checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')
    
    def GetActionAtTest(self, State):
        Action = np.argmax(self.SourceNetwork.Output.eval(feed_dict={self.SourceNetwork.InputState: [State]}))
        
        self.TimeStep += 1

        return Action