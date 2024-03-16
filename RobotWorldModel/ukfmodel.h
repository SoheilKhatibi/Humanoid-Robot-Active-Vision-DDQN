#ifndef UKFMODEL_H
#define UKFMODEL_H

#include <armadillo>
#include <math.h>
#include <vector>

// using namespace std;
// using namespace arma;


// lines
struct Line{
    arma::mat sp,ep;
    double length,angle;
    int type;
};

// possible positions
struct PossiblePosition{
    double x,y,a;
    bool Front;
};

class UKFModel{
public:
    UKFModel();
    void initialize(arma::mat s, arma::mat p, double w);
    void generate_sigma_points();
    arma::mat mean_of_sigmaPoints();
    double get_model_weight();
    void converge(double x,double y,double a);
    void set_weight(double w);
    arma::mat cov_of_sigmaPoints();
    void predict_pos(double dx,double dy,double da);
    void checkBoundary();
    arma::mat get_pose();
    void observe_line(Line& A);
    void line_update_2d(double distance,double angle,std::vector<Line> lmPos);
    bool is_horizental(Line& A);
    arma::mat cov_of_sigmaPoint_and_predictedZ_2d(arma::mat* Z,arma::mat& mZ);
    void update_model_weight(double masurmentW);
    void observe_boundary_single_line(Line& A);
    void observe_boundary_two_line(Line& line1,Line& line2);
    void observe_corner(std::vector<arma::mat>& pos,double angle,arma::mat& vCenter);
    void pose_update(std::vector<PossiblePosition>& poses,arma::mat& Q,double r);
    arma::mat cov_of_sigmaPoint_and_predictedZ(arma::mat* Z,arma::mat& mZ);
    void observe_cornerT(Line& line1,Line& line2,double angle,arma::mat& intersection);
    void observe_cornerL(Line& line1,Line& line2,double angle,arma::mat& intersection);
    void observe_centerT(arma::mat& intersection,double angle);
    void observe_centerCircle(arma::mat& intersection,double angle);
    void observe_field_cornerT(arma::mat& intersection,double angle);
    void observe_penalty_area(Line& line1,Line& line2,double angle,arma::mat& intersection);
    void observe_parallel_lines(Line& line);
    void observe_spot(double* v);
    void landmark_update_2d(std::vector<arma::mat>& lmPos,double* v);
    arma::mat cov_of_sigmaPoint_and_predictedZ_pose2d(arma::mat* Z , arma::mat& mZ);
    arma::mat get_mean();
    arma::mat get_cov();
    double modelWeight;
private:
    arma::cube sigmaPoints;
    arma::mat spWeight;
    arma::mat spImportanceFac;
    arma::mat cov;
    arma::mat mean;
    double wSlow;
    double wFast;
    bool only_update_weigth;
};


arma::mat compute_cholesky_decomposition(arma::mat A);
double mod_angle(double a);
arma::mat state_vector_diff(arma::mat& v1 ,arma::mat& v2);
void robot_to_line_distance(arma::mat& p1, arma::mat& p2, double& distanceToLine, double& x, double& y);
arma::mat pose_global(arma::mat pRelative,arma::mat Pose);
double get_gussian_prob(double x,double s);
arma::mat pose_relative(arma::mat pGlobal, arma::mat Pose);
arma::mat mean_of_predictedZ_dis_angle(arma::mat* t);
arma::mat cov_of_predictedZ_2d(arma::mat* Z, arma::mat& mZ);
arma::mat state_vector_diff_2d(arma::mat& v1 , arma::mat& v2);
double FindMin(arma::mat t, int& imin);
void JustForTest();
void get_odometry(std::vector<double>& uOdometry0 ,std::vector<double>& uOdometry,double* walkULeft,double* walkURight);
void init_kalmanFilter(UKFModel& UKF,double x,double y,double a,arma::mat uncCovMat,double w);
void se2_interpolate(double t,double* u1,double* u2,double* answer);
void init();
int get_associative_landmark(arma::mat& robotPose,double* detectedLandmark, std::vector<arma::mat>& landmarksInField);
arma::mat mean_of_predictedZ(arma::mat* t);
arma::mat cov_of_predictedZ_pose2d(arma::mat* Z,arma::mat& mZ);
arma::mat pose2d_vector_diff(arma::mat& v1,arma::mat& v2);

#endif // UKFMODEL_H
