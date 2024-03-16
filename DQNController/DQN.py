import tensorflow as TF
import HParams


class DQNNet:
    def __init__(self, Name):
        self.ActionSize = HParams.ActionSize
        self.LearningRate = HParams.LearningRate
        
        with TF.variable_scope(Name):
            
            # self.Actions = TF.placeholder(TF.float32, [None, self.ActionSize], name="Actions")
            # self.TargetQ = TF.placeholder(TF.float32, [None], name="TargetQ")
            
            
            self.InputState = TF.placeholder(TF.float32, [None, HParams.FrameHeight, HParams.FrameWidth, HParams.StackSize], name="InputState")

            self.Conv1 = TF.layers.conv2d(inputs = self.InputState,
                                          filters = 32,
                                          kernel_size = [5, 5],
                                          padding = "VALID",
                                          kernel_initializer=TF.contrib.layers.xavier_initializer_conv2d(),
                                          name = "Conv1")
            self.Conv1Out = TF.nn.elu(self.Conv1, name="Conv1Out")

            self.Pooling1 = TF.nn.max_pool2d(self.Conv1Out, [2, 2], [2, 2], padding = "VALID", name="Pooling1")

            self.Conv2 = TF.layers.conv2d(inputs = self.Pooling1,
                                          filters = 64,
                                          kernel_size = [3,3],
                                          strides = [2,2],
                                          padding = "VALID",
                                          kernel_initializer=TF.contrib.layers.xavier_initializer_conv2d(),
                                          name = "Conv2")
            self.Conv2Out = TF.nn.elu(self.Conv2, name="Conv2Out")
            
            self.Conv3 = TF.layers.conv2d(inputs = self.Conv2Out,
                                          filters = 64,
                                          kernel_size = [3,3],
                                          strides = [2,2],
                                          padding = "VALID",
                                          kernel_initializer=TF.contrib.layers.xavier_initializer_conv2d(),
                                          name = "Conv3")
            self.Conv3Out = TF.nn.elu(self.Conv3, name="Conv3Out")
            
            self.Flatten = TF.contrib.layers.flatten(self.Conv3Out)
            
            self.Fc = TF.layers.dense(inputs = self.Flatten,
                                      units = 512,
                                      activation = TF.nn.elu,
                                      kernel_initializer=TF.contrib.layers.xavier_initializer(),
                                      name="Fc")
            
            self.Output = TF.layers.dense(inputs = self.Fc, 
                                          kernel_initializer=TF.contrib.layers.xavier_initializer(),
                                          units = self.ActionSize, 
                                          activation=None)
            

            # self.Q = TF.reduce_sum(TF.multiply(self.Output, self.Actions))
            # self.Loss = TF.reduce_mean(TF.square(self.TargetQ - self.Q))
            # self.Optimizer = TF.train.AdamOptimizer(self.LearningRate).minimize(self.Loss)
