--Localization parameters

world = {}
world.n = 100
world.xLineBoundary = 4.5
world.yLineBoundary = 3.0
world.xPadding = 0.50
world.yPadding = 0.70
world.xMax = world.xLineBoundary + world.xPadding
world.yMax = world.yLineBoundary + world.yPadding

world.ballYellow = {{4.5, 0.0}}
world.ballCyan = {{-4.5, 0.0}}

world.goalWidth = 2.6
world.goalHeight = 1.15
world.goalDepth = 0.6

world.enableBackPosts = 1

world.postYellow = {}
world.postYellow[1] = {4.5, 1.3}
world.postYellow[2] = {4.5, -1.3}

world.postCyan = {}
world.postCyan[1] = {-4.5, -1.3}
world.postCyan[2] = {-4.5, 1.3}

if world.enableBackPosts == 1 then
    world.postYellow[3] = {4.1, 1.3}
    world.postYellow[4] = {4.1, -1.3}
    world.postCyan[3] = {-4.1, -1.3}
    world.postCyan[4] = {-4.1, 1.3}
end
world.spot = {}
world.spot[1] = {-2.4, 0}
world.spot[2] = {2.4, 0}

world.cResample = 1 --Resampling interval
world.useOnlyOdometry = 0
world.odomScale = {1.15, 1, 0}
world.imuYaw = 1

local nao_field = 0

if (nao_field > 0) then
    world.goalWidth = 1.6
    world.goalHeight = 0.85
    world.postYellow = {}
    world.postYellow[1] = {4.5, 0.8}
    world.postYellow[2] = {4.5, -0.8}
    world.postCyan = {}
    world.postCyan[1] = {-4.5, -0.8}
    world.postCyan[2] = {-4.5, 8.0}
end

-- default positions for our kickoff
world.initPosition1 = {
    {4.5, 0}, --Goalie
    {0.9, 0}, --Attacker
    {2.5, -1}, --Defender
    {2, 1} --Supporter
}

world.initPositionReady1 = {
    {4.5, 0}, --Goalie
    {0.9, 0}, --Attacker
    {2.5, -2}, --Defender
    {2.5, 2} --Supporter
}

-- default positions for opponents' kickoff
-- Center circle radius: 0.6
world.initPosition2 = {
    {4.5, 0}, --Goalie
    {0.95, 0}, --Attacker
    {2.5, -1}, --Defender
    {2, 1} --Supporter
}

world.initPositionReady2 = {
    {4.5, 0}, --Goalie
    {0.95, 0}, --Attacker
    {2.5, -2}, --Defender
    {2.5, 2} --Supporter
}

-- default positions for dropball
-- Center circle radius: 0.6
world.initPosition3 = {
    {4.5, 0}, --Goalie
    {0.95, 0}, --Attacker
    {2.5, -1.3}, --Defender
    {2.5, 1.3} --Supporter
}

world.startPosition = {
    {3.5, -3}, --Goalie
    {0.95, -3}, --Attacker
    {2.5, -3}, --Defender
    {1.5, -3} --Supporter
}

world.initPositionPenalty = {-2.1, 0}

-- filter weights
world.rGoalFilter = 0.02
world.aGoalFilter = 0.05
world.rPostFilter = 0.02
world.aPostFilter = 0.10

world.rLandmarkFilter = 0.05
world.aLandmarkFilter = 0.10

world.rCornerFilter = 0.01
world.aCornerFilter = 0.03

world.aLineFilter = 0.02

--New two-goalpost localization
world.use_new_goalposts = 1

--Bahareh Foroughi: New Occupancy Map Parameters:
occ = {}
occ.tmax = 1500
occ.free = 1
occ.occupied = -1
occ.unknown = 0
occ.upperLimit = 10000
occ.mapsize = {}
occ.mapsize.x = 90
occ.mapsize.y = 60

world.use_occ_map = 0
--world.use_occ_map = 1;

world.use_penalty_goal = 0
world.use_penalty_head = 1

world.use_goali_to_mirror_model = 0
world.use_same_colored_goal = 1
world.use_autoSwitch_mirror_model = 0
world.use_goali_ball_to_mirror_model = 1

world.use_new_monitor = 0
world.send_fps = 5

--Use line information to fix angle
world.use_line_angles = 1

world.use_corner_type = 0 --LT

world.cornerT = {}
world.cornerT[1] = {4.5, 2.5, 0}
world.cornerT[2] = {4.5, -2.5, 0}
world.cornerT[3] = {-4.5, 2.5, math.pi}
world.cornerT[4] = {-4.5, -2.5, math.pi}
--cirlce T corners
world.cornerT[5] = {0, 3, 0.5 * math.pi}
world.cornerT[6] = {0, -3, -0.5 * math.pi}

world.cornerL = {}
world.cornerL[1] = {4.5, 3, -0.75 * math.pi}
world.cornerL[2] = {4.5, -3, 0.75 * math.pi}
world.cornerL[3] = {-4.5, 3, -0.25 * math.pi}
world.cornerL[4] = {-4.5, -3, 0.25 * math.pi}

world.penaltyArea = {}
world.penaltyArea[1] = {3.5, 0, 0}
world.penaltyArea[2] = {-3.5, 0, math.pi}

--Penalty box edge
world.cornerL[5] = {-3.5, 2.5, -0.75 * math.pi}
world.cornerL[6] = {-3.5, -2.5, 0.75 * math.pi}
world.cornerL[7] = {3.5, 2.5, -0.25 * math.pi}
world.cornerL[8] = {3.5, -2.5, 0.25 * math.pi}

--corner X edge
world.cornerX = {}
world.cornerX[1] = {0, 0.75}
world.cornerX[2] = {0, -0.75}

world.ukf = {}
world.ukf.dimention = 3
world.ukf.alpha = 1
world.ukf.beta = 0
world.ukf.kappa = 0
world.ukf.max_ukfs_num = 8
world.ukf.distanc_thrd = 0.2
world.ukf.focuse_thrd = 0.09
world.ukf.thred_cluster = 6
world.ukf.monitor_samples = 1

world.monitor_particles = 1
world.monitor_ukfs = 1

world.enable_goal_update = 0
world.enable_corner_update = 0
world.enable_boundary_update = 0
world.enable_line_update = 0
world.enable_parallelLine_update = 0
world.enable_penaltyArea_update = 0
world.enable_spot = 0
world.enable_centerT = 0
world.enable_cornerT = 0
world.enable_circle = 0

world.max_ukfs_num = 8

world.enable_mirror = 0
world.enable_manual_positioning = 0
world.enable_auto_positioning = 1

world.ManualPenaltyKick = 0
world.Enable_ActiveVision = 1

world.XKickUpdate = 0.03

world.MyPlotConfig = false

world.obstaclIinitSTD = {0.1, 0.1, 0.1};

world.ActiveVisionBoundaryConfig = true
world.ActiveVisionLineConfig = true
world.ActiveVisionLCornerConfig = true
world.ActiveVisionTCornerConfig = true

return world
