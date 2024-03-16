#ifdef __cplusplus
extern "C"
{
#endif

#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"

#ifdef __cplusplus
}
#endif

#include "Transform.h"
#include "HeadTransform.h"
#include <iostream>
#include "shmProvider.h"
#include <math.h>

using namespace std;

Transform transformMatrix;
double focalLength;

TorsoParams::TorsoParams(double bodyHeight_, double footX_, double imuTilt_, double imuRoll_, double hipYaw_, double supportX_)
    : bodyHeight(bodyHeight_), footX(footX_), imuTilt(imuTilt_), imuRoll(imuRoll_), hipYaw(hipYaw_), supportX(supportX_) {}
TorsoParams::TorsoParams(const TorsoParams &torso)
    : bodyHeight(torso.bodyHeight), footX(torso.footX), imuTilt(torso.imuTilt), imuRoll(torso.imuRoll), hipYaw(torso.hipYaw), supportX(torso.supportX) {}
void TorsoParams::operator=(const TorsoParams &rhs){
    bodyHeight = rhs.bodyHeight;
    footX = rhs.footX;
    imuTilt = rhs.imuTilt;
    imuRoll = rhs.imuRoll;
    hipYaw = rhs.hipYaw;
    supportX = rhs.supportX;
}

CameraParams::CameraParams(double neckX_, double neckZ_, double headPitch_, double headTilt_, double cameraX_, double cameraY_, double cameraZ_,
                           double cameraPitch_, double cameraRoll_, double cameraTilt_) : neckX(neckX_), neckZ(neckZ_), headPitch(headPitch_), headTilt(headTilt_), cameraX(cameraX_), cameraY(cameraY_), cameraZ(cameraZ_), cameraPitch(cameraPitch_), cameraRoll(cameraRoll_), cameraTilt(cameraTilt_) {}
CameraParams::CameraParams(const CameraParams &cam) : neckX(cam.neckX), neckZ(cam.neckZ), headPitch(cam.headPitch), headTilt(cam.headTilt), cameraX(cam.cameraX), cameraY(cam.cameraY), cameraZ(cam.cameraZ), cameraPitch(cam.cameraPitch), cameraRoll(cam.cameraRoll), cameraTilt(cam.cameraTilt) {}

void CameraParams::operator=(const CameraParams &rhs){
    neckX = rhs.neckX;
    neckZ = rhs.neckZ;
    headPitch = rhs.headPitch;
    headTilt = rhs.headTilt;
    cameraX = rhs.cameraX;
    cameraY = rhs.cameraY;
    cameraZ = rhs.cameraZ;
    cameraPitch = rhs.cameraPitch;
    cameraRoll = rhs.cameraRoll;
    cameraTilt = rhs.cameraTilt;
}

void updateParam(TorsoParams torsoParams, CameraParams cameraParams, double focalLength_){
    transformMatrix = Transform(-torsoParams.footX + torsoParams.supportX, 0, torsoParams.bodyHeight);
    transformMatrix.rotateZ(torsoParams.hipYaw).rotateY(torsoParams.imuTilt).rotateX(torsoParams.imuRoll).translate(cameraParams.neckX, 0, cameraParams.neckZ).rotateZ(cameraParams.headTilt).rotateY(cameraParams.headPitch).translate(cameraParams.cameraX, cameraParams.cameraY, cameraParams.cameraZ).rotateZ(cameraParams.cameraTilt).rotateY(cameraParams.cameraPitch).rotateX(cameraParams.cameraRoll);

    focalLength = focalLength_;
}

vector<double> robotToImg(double x, double y, double z){
    vector<double> pointInRobot = {x, y, z, 1};
    Transform invHead = transformMatrix.inverse();
    vector<double> pointInCamera = invHead * pointInRobot;
    pointInCamera = {pointInCamera[0] * (focalLength / pointInCamera[0]), pointInCamera[1] * (focalLength / pointInCamera[0]), pointInCamera[2] * (focalLength / pointInCamera[0])};
    vector<double> pointInImage = {cameraInfo.imgXctr - pointInCamera[1], cameraInfo.imgYctr - pointInCamera[2]};
    vector<double> corrected = radialDistortedPoint(pointInImage[0], pointInImage[1]);
    corrected = tangentialDistortedPoint(corrected[0], corrected[1]);
    return corrected;
}

vector<double> robotToLabelA(double x, double y, double z){
    vector<double> poingInimage = robotToImg(x, y, z);
    poingInimage[0] /= cameraInfo.scaleA;
    poingInimage[1] /= cameraInfo.scaleA;
    return poingInimage;
}

vector<double> radialDistortedPoint(double x, double y){
    double xCtr = cameraInfo.imgXctr;
    double yCtr = cameraInfo.imgYctr;
    double norX = (x - xCtr) / cameraInfo.focalLength;
    double norY = (yCtr - y) / cameraInfo.focalLength;
    double r2 = pow(norX, 2) + pow(norY, 2);
    double norCorX = norX * (1 + cameraInfo.radialDistortion[0] * r2 + cameraInfo.radialDistortion[1] * pow(r2, 2) + cameraInfo.radialDistortion[2] * pow(r2, 3));
    double norCorY = norY * (1 + cameraInfo.radialDistortion[0] * r2 + cameraInfo.radialDistortion[1] * pow(r2, 2) + cameraInfo.radialDistortion[2] * pow(r2, 3));
    double corX = norCorX * cameraInfo.focalLength + xCtr;
    double corY = -(norCorY * cameraInfo.focalLength) + yCtr;

    return {corX, corY};
}

vector<double> tangentialDistortedPoint(double x, double y){
    double xCtr = cameraInfo.imgXctr;
    double yCtr = cameraInfo.imgYctr;
    double norX = (x - xCtr) / cameraInfo.focalLength;
    double norY = (yCtr - y) / cameraInfo.focalLength;
    double r2 = pow(norX, 2) + pow(norY, 2);
    double norCorX = norX + (2 * cameraInfo.tangentialDistortion[0] * norX * norY + cameraInfo.tangentialDistortion[1] * (r2 + 2 * pow(norX, 2)));
    double norCorY = norY + (cameraInfo.tangentialDistortion[0] * (r2 + 2 * pow(norY, 2)) + 2 * cameraInfo.tangentialDistortion[1] * norX * norY);
    double corX = norCorX * cameraInfo.focalLength + xCtr;
    double corY = -(norCorY * cameraInfo.focalLength) + yCtr;
    return {corX, corY};
}

vector<double> labelARadialDistortedPoint(double x, double y){
    double xCtr = cameraInfo.x0A;
    double yCtr = cameraInfo.y0A;
    double norX = (x - xCtr) / cameraInfo.focalA;
    double norY = (yCtr - y) / cameraInfo.focalA;
    double r2 = pow(norX, 2) + pow(norY, 2);
    double norCorX = norX * (1 + cameraInfo.radialDistortion[0] * r2 + cameraInfo.radialDistortion[1] * pow(r2, 2) + cameraInfo.radialDistortion[2] * pow(r2, 3));
    double norCorY = norY * (1 + cameraInfo.radialDistortion[0] * r2 + cameraInfo.radialDistortion[1] * pow(r2, 2) + cameraInfo.radialDistortion[2] * pow(r2, 3));
    double corX = norCorX * cameraInfo.focalA + xCtr;
    double corY = -(norCorY * cameraInfo.focalA) + yCtr;

    return {corX, corY};
}

vector<double> labelATangentialDistortedPoint(double x, double y){
    double xCtr = cameraInfo.x0A;
    double yCtr = cameraInfo.y0A;
    double norX = (x - xCtr) / cameraInfo.focalA;
    double norY = (yCtr - y) / cameraInfo.focalA;
    double r2 = pow(norX, 2) + pow(norY, 2);
    double norCorX = norX + (2 * cameraInfo.tangentialDistortion[0] * norX * norY + cameraInfo.tangentialDistortion[1] * (r2 + 2 * pow(norX, 2)));
    double norCorY = norY + (cameraInfo.tangentialDistortion[0] * (r2 + 2 * pow(norY, 2)) + 2 * cameraInfo.tangentialDistortion[1] * norX * norY);
    double corX = norCorX * cameraInfo.focalA + xCtr;
    double corY = -(norCorY * cameraInfo.focalA) + yCtr;
    return {corX, corY};
}

void rayIntersectA(int x, int y, double *v){
    double *p0 = neckPos;
    vector<double> corrected = labelARadialDistortedPoint(x, y);
    corrected = labelATangentialDistortedPoint(corrected[0], corrected[1]);
    x = corrected[0];
    y = corrected[1];

    double p1[4] = {cameraInfo.focalA, -(x - cameraInfo.x0A), -(y - cameraInfo.y0A), 1.0};
    mat4x4ByVec4(cameraMatrix, p1);

    double v0 = p1[0] - p0[0];
    double v1 = p1[1] - p0[1];
    double v2 = p1[2] - p0[2];
    double t = -p0[2] / v2;

    if (t < 0)
        t = -t;

    v[0] = p0[0] + t * v0;
    v[1] = p0[1] + t * v1;
}

void labelAToRobotByPoint(int px, int py, double *v, double& IsOnField){
    double invFocalA = 1 / cameraInfo.focalA;
    vector<double> corrected = labelARadialDistortedPoint(px, py);
    corrected = labelATangentialDistortedPoint(corrected[0], corrected[1]);
    px = corrected[0];
    py = corrected[1];
    double dirVec[4] = {1, (cameraInfo.x0A - px)*invFocalA , (cameraInfo.y0A - py)*invFocalA , 0};
    mat4x4ByVec4(cameraMatrix, dirVec);
    double cameraVec[4] = {cameraMatrix[0][3], cameraMatrix[1][3], cameraMatrix[2][3], 0};
    double t = cameraVec[2] / dirVec[2];
    v[0] = cameraVec[0] - t * dirVec[0];
    v[1] = cameraVec[1] - t * dirVec[1];
    IsOnField = t;
}

int pixelToImageBallRadius(int x, int y, double realRadius){
    double v[2], IsOnfield;
    labelAToRobotByPoint(x, y, v, IsOnfield);
    double dis = sqrt(v[0] * v[0] + v[1] * v[1]);
    double camera_height = cameraMatrix[2][3];
    double camera_dis = sqrt(camera_height * camera_height + dis * dis);
    return (double)((cameraInfo.focalA * realRadius) / camera_dis);
}

double getPixelWidthA(int x, int y){
    int xLeft = max(0, x - 2);
    int xRight = min(x + 2, 320);
    int disX = fabs(xRight - xLeft);
    double vLeft[2];
    rayIntersectA(xLeft, y, vLeft);
    double vRight[2];
    rayIntersectA(xRight, y, vRight);
    return (fabs((vRight[1] - vLeft[1]) / disX));
}
