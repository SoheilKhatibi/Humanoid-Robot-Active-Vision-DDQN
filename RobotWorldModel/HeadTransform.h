#ifndef lua_headtransform_h_DEFINED
#define lua_headtransform_h_DEFINED
#include "MatrixTransform.h"
#include <iostream>
#include <vector>

using namespace std;




struct TorsoParams{
  double bodyHeight;
  double footX;
  double supportX;
  double imuTilt;
  double imuRoll;
  double hipYaw;
  TorsoParams(double bodyHeight_=0,double footX_=0,double imuTilt_=0,double imuRoll_=0,double hipYaw_=0,double supportX_=0);
  TorsoParams(const TorsoParams & torso);
  void operator=(const TorsoParams & rhs);
  friend ostream & operator<<(ostream & out,TorsoParams torso){
  out<<torso.bodyHeight<<" "<<torso.footX<<" "<<torso.supportX<<" "<<torso.imuTilt<<" "<<torso.imuRoll<<" "<<torso.hipYaw<<" ";
  return out;
}
};
struct CameraParams{
  double neckX;
  double neckZ;
  double headPitch;
  double headTilt;
  double cameraX;
  double cameraY;
  double cameraZ;
  double cameraPitch;
  double cameraRoll;
  double cameraTilt;
  CameraParams(double neckX_=0,double neckZ_=0,double headPitch_=0,double headTilt_=0,double cameraX_=0,double cameraY_=0,double cameraZ_=0,
	       double cameraPitch_=0,double cameraRoll_=0,double cameraTilt_=0);
  CameraParams(const CameraParams & cam);
  void operator=(const CameraParams & rhs);
  friend ostream & operator<<(ostream & out,CameraParams cam){
    out<<cam.neckX<<" "<<cam.neckZ<<" "<<cam.headPitch<<" "<<cam.headTilt<<" "<<cam.cameraX<<" "<<cam.cameraY<<" "<<cam.cameraZ<<" "<<cam.cameraPitch<<" "<<cam.cameraRoll<<" "<<cam.cameraRoll<<" "<<cam.cameraTilt;
    return out;
  }
};
void updateParam(TorsoParams torsoParams,CameraParams cameraParams,double focalLength_);
vector<double> robotToImg(double x,double y,double z);
vector<double> robotToLabelA(double x,double y,double z);

vector<double> radialDistortedPoint(double x,double y);
vector<double> tangentialDistortedPoint(double x,double y);
vector<double> labelARadialDistortedPoint(double x,double y);
vector<double> labelATangentialDistortedPoint(double x,double y);
void rayIntersectA(int x,int y,double* v);
int pixelToImageBallRadius(int x, int y,double realRadius);
double getPixelWidthA(int x,int y);
void labelAToRobotByPoint(int px, int py, double *v, double& IsOnField);


#endif
