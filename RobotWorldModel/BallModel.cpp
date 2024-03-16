#include "BallModel.h"

#include "shmProvider.h"
#include "ukfmodel.h"
#include "luatables.h"

double ball_gamma = 0.3;
double XKickUpdate;
bool GPSUpdate = true;
double BallGlobal[3], BallLocal[3];
double* RobotPose;

void BallModelEntry(){
    LuaTable input (LuaTable::fromFile("WorldConfigForCppModel.lua"));
    XKickUpdate = input["XKickUpdate"].getDefault<double>(false);
}

BallModel::BallModel(int InputOwnerNumber, double x, double y){
    p = 0;
    vx = 0;
    vy = 0;
    time = getTime();
    mean = arma::mat(2, 1, arma::fill::zeros);
    mean(0,0) = x;
    mean(1,0) = y;
    cov = arma::mat(2, 2, arma::fill::zeros);
    cov(0,0) = 0.01;
    cov(1,1) = 0.01;
    OwnerNumber = InputOwnerNumber;
}

void BallModel::get_xy(double& x, double& y){
    x = mean(0,0);
    y = mean(1,0);
}

void BallModel::get_ra(double& rG,double& aG){
    rG = r;
    aG = a;
}

void BallModel::get_deviation(double& dr,double& da){
    dr = sqrt(rVar);
    da = sqrt(aVar);
}

void BallModel::update(bool Detected, arma::mat& V, double varX, double varY, std::string& headState, double GPSPoseX, double GPSPoseY, arma::mat& pose){
    double KickDone = 0;//mcmKick.get<double>("IsDone")[0];
    if (KickDone) {
        UpdateKick();
    }

    if (GPSUpdate){
        BallGlobal[0] = GPSPoseX;//wcmBall.get<double>("GTX")[0];
        BallGlobal[1] = GPSPoseY;//wcmBall.get<double>("GTY")[0];
        BallGlobal[2] = 0;//wcmBall.get<double>("GTY")[0];
        double RobotPose[3] = {pose(0, 0), pose(1, 0), pose(2, 0)};
        pose_relative(BallGlobal, RobotPose, BallLocal);
        mean(0,0) = BallLocal[0];
        mean(1,0) = BallLocal[1];
    }else if (Detected){
        observation_xy(V[0], V[1], varX, varY);
    }

    // if (Detected){
    //     p = (1-ball_gamma)*p+ball_gamma;
    //     time = getTime();
    // }else if (headState == "headTrack" /*&& wcmRobot.get<double>("headIsOnBall")[0] == 1*/ ){
    //     p = (0.95)*p;
    // }else if (headState !="headSweep" && headState !="headTrack" ){
    //     p = (0.95)*p;
    // }
    // p = std::min(std::max(p,(double)0),(double)1);
}

void BallModel::observation_ra(double rL,double aL,double rErr=1,double aErr=1){
    double rVarL = pow(rErr, 2);
    double aVarL = pow(aErr, 2);

    double dr = rL - r;
    r = r + (rVar * dr)/(rVar + rVarL);
    double da = mod_angle(aL - a);
    a = a + (aVar * da)/(aVar + aVarL);

    rVar = (rVar * rVarL)/(rVar + rVarL);
    if (rVar + rVarL < pow(dr, 2)){
        rVar = pow(dr, 2);
    }

    aVar = (aVar * aVarL)/(aVar + aVarL);
    if (aVar + aVarL < pow(da, 2)){
        aVar = pow(da, 2);
    }
}

void BallModel::observation_xy(double x,double y,double VarX=1,double VarY=1){
    arma::mat Q(2,2,arma::fill::zeros);
    Q(0,0) = VarX;
    Q(1,1) = VarY;
    arma::mat gain = cov * (cov + Q).i();
    arma::mat Z(2,1);
    Z(0,0) = x;
    Z(1,0) = y;
    mean = mean + gain * (Z-mean);
    cov = (arma::eye(2,2) - gain) * cov;
}

void BallModel::odometry(double dx,double dy,double da){
  
    
    double ca = cos(-da);
    double sa = sin(-da);
    
    arma::mat A(2,2,arma::fill::zeros);
    A << ca << -sa << arma::endr
      << sa << ca << arma::endr;
    
    arma::mat u(2,1,arma::fill::zeros);
    u << dx << arma::endr
      << dy << arma::endr;
    
    
    mean = A * mean - u; 
    
    
    arma::mat Q(2,2,arma::fill::zeros);
    Q(0,0) = 0.1 * dx;
    Q(1,1) = 0.1 * dy;
    cov = A * cov * A.t() + Q;
    arma::mat ProcessNoise(2,2,arma::fill::zeros);
    ProcessNoise << 0.005 << 0 << arma::endr
                 << 0 << 0.005 << arma::endr;
    cov = cov + ProcessNoise;
}

void BallModel::SetMean(double x, double y){
    mean(0,0) = x;
    mean(1,0) = y;
    time = getTime();
}

void BallModel::SetProbability(double probability){
    p = probability;
}

void BallModel::UpdateKick(){
    // double* Pose = wcmRobot.get<double>("pose");
    // double PoseA = Pose[2];
    mean(0,0) = mean(0,0) + XKickUpdate;
    arma::mat ProcessNoise(2,2,arma::fill::zeros);
    ProcessNoise << 0.01 << 0 << arma::endr
                 << 0 << 0.003 << arma::endr;
    cov = cov + ProcessNoise;
}

void pose_relative(double* pGlobal,double* Pose,double* RelativePose){
    float ca = cos(Pose[2]);
    float sa = sin(Pose[2]);
    
    float px = pGlobal[0]-Pose[0];
    float py = pGlobal[1]-Pose[1];
    float pa = pGlobal[2]-Pose[2];
    
    RelativePose[0] = ca*px + sa*py;
    RelativePose[1] = -sa*px + ca*py;
    RelativePose[2] = mod_angle(pa);
}