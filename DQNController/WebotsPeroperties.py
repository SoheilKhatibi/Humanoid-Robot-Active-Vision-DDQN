import controller
from controller import Supervisor
from controller import Keyboard
from controller import GPS

robot = Supervisor()

timestep = int(robot.getBasicTimeStep())

Ashkan = robot.getFromDef("Ashkan")
Ball = robot.getFromDef("BALL")

Camera = robot.getCamera("Camera")
# HeadGPS = robot.getFromDef("HeadGPS")

AshkanTranslation = Ashkan.getField("translation")
AshkanRotation = Ashkan.getField("rotation")

BallTranslation = Ball.getField("translation")

NeckJoint = robot.getMotor('Neck')
HeadJoint = robot.getMotor('Head')

NeckSensor = controller.PositionSensor('NeckS')
HeadSensor = controller.PositionSensor('HeadS')

NeckSensor.enable(timestep)
HeadSensor.enable(timestep)