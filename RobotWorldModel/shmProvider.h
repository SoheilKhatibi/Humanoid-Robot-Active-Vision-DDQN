#ifndef SHMPROVIDER_H

#define SHMPROVIDER_H

#include <boost/foreach.hpp>
#include <vector>

extern double xLineBoundary;
extern double yLineBoundary;

extern double xMax;
extern double yMax;

extern int role;
extern int InitRole;
extern bool webots;

extern double getTime();

extern double mod_angle(double a);

struct CameraInfo{
    double focalLength;
    double focalBase;
    double focalA;
    double focalB;
    int width;
    int height;
    int scaleA;
    int scaleB;
    double x0A;
    double y0A;
    double imgXctr;
    double imgYctr;
    std::vector<double> radialDistortion;
    std::vector<double> tangentialDistortion;
};

// Camera Parameters
extern CameraInfo cameraInfo;
extern double cameraMatrix[4][4];
extern double neckPos[4];
extern int horizonB;
extern int horizonBall;
// Camera Parameters

#endif
