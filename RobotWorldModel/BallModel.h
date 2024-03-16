#ifndef BALLMODEL_H
#define BALLMODEL_H

#include <math.h>
#include <armadillo>

// using namespace std;
// using namespace arma;

class BallModel{
public:
    BallModel(int InputOwnerNumber, double x, double y);
    void get_xy(double& x, double& y);
    void get_ra(double& rG,double& aG);
    void get_deviation(double& dr,double& da);
    void update(bool Detected, arma::mat& V, double varX, double varY, std::string& headState, double GPSPoseX, double GPSPoseY, arma::mat& pose);
    void observation_ra(double rL,double aL,double rErr,double aErr);
    void observation_xy(double x,double y,double rErr,double aErr);
    void odometry(double dx,double dy,double da);
    void SetMean(double x, double y);
    void SetProbability(double probability);
    void UpdateKick();
    
    double r;
    double a;
    double p;
    double vx;
    double vy;
    double time;
    double rVar;
    double aVar;

    arma::mat mean;
    arma::mat cov;
    int OwnerNumber;


private:

    
};

void BallModelEntry();
void pose_relative(double* pGlobal,double* Pose,double* RelativePose);

#endif // BALLMODEL_H
