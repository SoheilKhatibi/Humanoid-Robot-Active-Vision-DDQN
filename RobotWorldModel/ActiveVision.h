#ifndef ACTIVEVISION_H
#define ACTIVEVISION_H

#include <boost/foreach.hpp>
#include <math.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
// #include "opencv/cv.h"
#include "shmProvider.h"
#include "ukfmodel.h"
#include "luatables.h"
#include "BallModel.h"


//ActiveVision Parameters
#define ObservationsNumber 25
#define ActionsNumber 40
#define YawNumbers 10
#define PitchNumbers 4
#define NumOfFieldBoundaryLines 4
#define NumOfLines 7
#define NumOfLCorners 8
#define NumOfTCorners 6
#define TotalLinesOfField 11
#define NumOfUselessLines 4

namespace ActiveVision{


struct Point{
    float x,y;
};

struct Observation{
    int number;
    std::string name;
    std::string Type;
    float tLastObserved;
    Point startPoint,EndPoint;
};

struct Position{
    int number;
    float Yaw;
    float Pitch;
    Point BounPointz[4];
};

struct Line{
    float A,B;
    bool IsVertical = false;
};
}   


void Update_Head(std::vector<UKFModel>& UKFModels, double& ball_confidence, std::vector<BallModel>& OwnBallModels, double poseX, double poseY, double poseA, int& BestAction);
void Initiate();
void determinePositions(cv::Point3f Pose);
int TheBestAction(int currentPositionNumber, cv::Point3f Pose, double& ballX, double& ballY, double& ball_confidence, std::vector<BallModel>& OwnBallModels);
int Visible(ActiveVision::Position action,ActiveVision::Observation observation,cv::Point3f Pose,ActiveVision::Point **BoundPoints, bool AtDedication);
ActiveVision::Line Equa(ActiveVision::Point k,ActiveVision::Point l);
int Intersection(ActiveVision::Line M,ActiveVision::Line N,ActiveVision::Point* Q);
int IsPointInPolygon(int nvert,ActiveVision::Point **vert,ActiveVision::Point test);
void pose_global(float* pRelative,cv::Point3f Pose,float* GlobalPose);
void pose_relative(float* pGlobal,cv::Point3f Pose,float* RelativePose);
float absolute(float x);
int validateLine(ActiveVision::Point A,ActiveVision::Point B);
int sign(float x);
void Detecteds(int ActionNumber,cv::Point3f Pose,std::string Name, double& ballX, double& ballY, double& ball_confidence);
void FieldOfView(int ActionNumber,cv::Point3f Pose,ActiveVision::Point *BoundPoints);
int hamming_distance(unsigned x, unsigned y);
void CheckConditions(int ActionNumber, double BallX, double BallY, double PoseX, double PoseY, double PoseA, double* Conditions, int ForCheck);
unsigned int countSetBits(unsigned int n);
void SaveShowMat(int ExpEpisodeNum, int step);
//ActiveVision Parameters

#endif // ACTIVEVISION_H
