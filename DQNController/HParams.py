ActionSize = 9

InitialEpsilon = 1
FinalEpsilon = 0.05
TotalEpisodes = 40000

LearningRate =  0.00025
Momentum = 0.95
MinGrad = 0.01

FrameHeight = 120
FrameWidth = 160
StackSize = 1

PlotSourceNetworkGradientsAndVariables = False
PlotBothNetworkWeights = False
MaximumNumberOfModelsToKeep = 10000
TensorboardDataSavePath = 'summary/'
ModelSavePath = 'saved_networks/'
MemorySize = 1000000
BatchSize = 32
PretrainLength = BatchSize
MaxTau = 1500
TrainInterval = 1
Gamma = 0.9
ModelSaveFrequency = 100
InitialReplaySize = 500
ModelNum = '9900'
SaveImages = False
Gym = True

TotalExperimentEpisodesNumber = 100
Active = True
LEStart = 0.0
LEEnd = 2.01
LEAnnealingStep = 0.01

FirstSaveModelTimeFrame = 0
SaveModelTimestep = 1000
LastSaveModelTimeFrame = 1000000

# Camera parameters
focalLength = 577 + 0
focalBase = 640
scaleA = 2
scaleB = 2
width = 640
height = 480
radialDK1 = 0
radialDK2 = 0
radialDK3 = 0
tangentialDP1 = 0
tangentialDP2 = 0
x_center = 320
y_center = 240
# Camera parameters
