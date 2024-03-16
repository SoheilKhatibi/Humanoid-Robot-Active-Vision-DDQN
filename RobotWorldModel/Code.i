%module Code

%include "std_vector.i"

%{
// #include <armadillo>
#include "luaModel.h"
#include "ukfmodel.h"
#include "ActiveVision.h"
#include "BallModel.h"
#include "Transform.h"
#include "MatrixTransform.h"
#include "HeadTransform.h"
#include "shmProvider.h"
%}

// %include <armadillo>
%include "luaModel.h"
%include "ukfmodel.h"
%include "BallModel.h"
%include "Transform.h"
%include "MatrixTransform.h"
%include "HeadTransform.h"
%include "shmProvider.h"

%typemap(out) double *foo %{
  $result = PyList_New(8); // use however you know the size here
  for (int i = 0; i < 8; ++i) {
    PyList_SetItem($result, i, PyFloat_FromDouble($1[i]));
  }
  delete $1; // Important to avoid a leak since you called new
%}

%inline %{
double *foo() {
  double *toReturn = new double[8];
  ProjectedPoints(toReturn);
  return toReturn;
}
%}

%typemap(out) double *pose %{
  $result = PyList_New(3); // use however you know the size here
  for (int i = 0; i < 3; ++i) {
    PyList_SetItem($result, i, PyFloat_FromDouble($1[i]));
  }
  delete $1; // Important to avoid a leak since you called new
%}

%inline %{
double *pose() {
  double *toReturn = new double[3];
  getpose(toReturn);
  return toReturn;
}
%}

%typemap(out) double *Status %{
  $result = PyList_New(4); // use however you know the size here
  for (int i = 0; i < 4; ++i) {
    PyList_SetItem($result, i, PyFloat_FromDouble($1[i]));
  }
  delete $1; // Important to avoid a leak since you called new
%}

%inline %{
double *Status(int ActNum, int ForCheck) {
  double *toReturn = new double[4];
  illustrate(toReturn, ActNum, ForCheck);
  return toReturn;
}
%}