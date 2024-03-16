#include "MatrixTransform.h"
#include <math.h>

using namespace std;

Transform::Transform(){
  elements = {
    {1,0,0,0},
    {0,1,0,0},
    {0,0,1,0},
    {0,0,0,1}
  };
}

Transform::Transform(double x,double y,double z){
  elements = {
    {1,0,0,x},
    {0,1,0,y},
    {0,0,1,z},
    {0,0,0,1}
  };
}


Transform& Transform::rotateZ(double angle){
  Transform rhs;
  rhs(0,0) = cos(angle);
  rhs(0,1) = -sin(angle);
  rhs(1,0) = sin(angle);
  rhs(1,1) = cos(angle);
  (*this) = (*this)*rhs;
  return (*this);
}


Transform& Transform::rotateY(double angle){
  Transform rhs;
  rhs(0,0) = cos(angle);
  rhs(2,0) = -sin(angle);
  rhs(0,2) = sin(angle);
  rhs(2,2) = cos(angle);
  (*this) = (*this)*rhs;
  return (*this);
}


Transform& Transform::rotateX(double angle){
  Transform rhs;
  rhs(1,1) = cos(angle);
  rhs(1,2) = -sin(angle);
  rhs(2,1) = sin(angle);
  rhs(2,2) = cos(angle);
  (*this) = (*this)*rhs;
  return (*this);
}


Transform& Transform::translate(double dx, double dy, double dz){
  Transform rhs(dx,dy,dz);
  (*this) = (*this) * rhs;
  return (*this);
}


Transform Transform::operator*(const Transform& rhs){
  Transform ans;
  for(int i = 0;i<4;i++)
    for(int j = 0;j<4;j++){
      double sum = 0;
      for(int k = 0;k<4;k++)
	sum += elements[i][k]*rhs(k,j);
      ans(i,j) = sum;
    }
  return ans;
}

vector< double > Transform::operator*(const vector< double >& rhs){
  vector<double> ans(4);
  for(int i = 0;i<4;i++){
    ans[i] = elements[i][0]*rhs[0]+
	      elements[i][1]*rhs[1]+
	      elements[i][2]*rhs[2]+
	      elements[i][3]*rhs[3];
	      
  }
  return ans;
}



vector< double > Transform::getRPY(){
  vector<double> ans;
  ans = {
    atan2(elements[3][2],elements[3][3]),
    atan2(-elements[3][1],sqrt( pow(elements[3][2],2) + pow(elements[3][3],2) ) ),
    atan2(elements[2][1],elements[1][1])
  };
  return ans;
}


Transform & Transform::operator=(const Transform& rhs){
  for(int i = 0;i<4;i++)
    for(int j = 0;j<4;j++)
      elements[i][j] = rhs(i,j);
  return (*this);
}


Transform Transform::inverse(){

  Transform ans;
  for(int i = 0;i<3;i++)
    for(int j = 0;j<3;j++)
      ans(i,j) = elements[j][i];
  for(int i = 0;i<3;i++){
    double sum = 0;
    for(int j = 0;j<3;j++)
      sum+= ans(i,j)*elements[j][3];
    ans(i,3) = -sum;
  }
  return ans;
  
}


double & Transform::operator()(int i, int j){
  return elements[i][j];
}


double Transform::operator()(int i, int j) const{
  return elements[i][j];
}

void Transform::print2(){
  std::cout<<":"<<std::endl;
  for (int i=0; i<4; i++){
    for (int j=0; j<4; j++){
      std::cout<<elements[i][j]<<"  ";
    }
    std::cout<<std::endl;
  }
}

// const char *Transform::print2(){
//   std::string a = "4567891";
//   for (int i=0; i<4; i++){
//     for (int j=0; j<4; j++){
//       // a = a+elements[i][j];
//       a.append(std::to_string(elements[i][j]));
//     }
//     std::cout<<std::endl;
//   }
//   return a.c_str();
// }


