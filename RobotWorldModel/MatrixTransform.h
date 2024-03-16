#ifndef MATRIX_TRANSFORM_H
#define MATRIX_TRANSFORM_H

#include <vector>
#include <string>
#include <iostream>
using namespace std;
class Transform{
public:
  Transform();
  Transform(double x,double y,double z);
  Transform & rotateZ(double angle);
  Transform & rotateY(double angle);
  Transform & rotateX(double angle);
  Transform & translate(double dx,double dy,double dz);
  Transform operator*(const Transform & rhs);
  vector<double> operator*(const vector<double> & rhs);
  vector<double> getRPY();
  Transform & operator=(const Transform & rhs);
  Transform inverse();
  double & operator()(int i,int j);
  double operator()(int i,int j) const;
  void print2();
  // const char *tostring();
  
  
private:
  vector<vector<double> > elements;
  
};

#endif
