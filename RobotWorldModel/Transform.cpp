#include <math.h>

void rotZ(double a, double (&matrix)[4][4]){
  double ca = cos(a);
  double sa = sin(a);
  matrix[0][0] = ca; 
  matrix[0][1] =-sa;
  matrix[0][2] = 0;
  matrix[0][3] = 0;
  matrix[1][0] = sa;
  matrix[1][1] = ca;
  matrix[1][2] = 0;
  matrix[1][3] = 0;
  matrix[2][0] = 0;
  matrix[2][1] = 0;
  matrix[2][2] = 1;
  matrix[2][3] = 0;
  matrix[3][0] = 0;
  matrix[3][1] = 0;
  matrix[3][2] = 0;
  matrix[3][3] = 1;
}
 
void rotY(double a, double (&matrix)[4][4]){
  double ca = cos(a);
  double sa = sin(a);
  matrix[0][0] = ca; 
  matrix[0][1] = 0;
  matrix[0][2] = sa;
  matrix[0][3] = 0;
  matrix[1][0] = 0;
  matrix[1][1] = 1;
  matrix[1][2] = 0;
  matrix[1][3] = 0;
  matrix[2][0] = -sa;
  matrix[2][1] = 0;
  matrix[2][2] = ca;
  matrix[2][3] = 0;
  matrix[3][0] = 0;
  matrix[3][1] = 0;
  matrix[3][2] = 0;
  matrix[3][3] = 1;
}
 
void rotX(double a, double (&matrix)[4][4]){
  double ca = cos(a);
  double sa = sin(a);
  matrix[0][0] = 1; 
  matrix[0][1] = 0;
  matrix[0][2] = 0;
  matrix[0][3] = 0;
  matrix[1][0] = 0;
  matrix[1][1] = ca;
  matrix[1][2] =-sa;
  matrix[1][3] = 0;
  matrix[2][0] = 0;
  matrix[2][1] = sa;
  matrix[2][2] = ca;
  matrix[2][3] = 0;
  matrix[3][0] = 0;
  matrix[3][1] = 0;
  matrix[3][2] = 0;
  matrix[3][3] = 1;
}

void trans(double dx, double dy, double dz, double (&matrix)[4][4]){
  matrix[0][0] = 1; 
  matrix[0][1] = 0;
  matrix[0][2] = 0;
  matrix[0][3] = dx;
  matrix[1][0] = 0;
  matrix[1][1] = 1;
  matrix[1][2] = 0;
  matrix[1][3] = dy;
  matrix[2][0] = 0;
  matrix[2][1] = 0;
  matrix[2][2] = 1;
  matrix[2][3] = dz;
  matrix[3][0] = 0;
  matrix[3][1] = 0;
  matrix[3][2] = 0;
  matrix[3][3] = 1;
}

void mul_mat44(double A[4][4] , double B[4][4], double (&Result)[4][4] ){
  double C[4][4] = {0};
  int i,j;

	for (i = 0;i<4;i++)
      for (j = 0;j<4;j++) 
{             C[i][j] = (A[i][0] * B[0][j])+ (A[i][1] * B[1][j]) + (A[i][2] * B[2][j]) + (A[i][3] * B[3][j]);}
                  
     for (i = 0;i<4;i++)
      for (j = 0;j<4;j++) 
{             Result[i][j] = C[i][j];}
}

void mat4x4ByVec4(double mat[4][4],double vec[4]){
  double outVec[4] = {0};
  for(int i=0;i<4;i++){
    outVec[i] = mat[i][0] * vec[0] + mat[i][1] * vec[1] + mat[i][2] * vec[2] + mat[i][3] * vec[3];
  }
  vec[0] = outVec[0];
  vec[1] = outVec[1];
  vec[2] = outVec[2];
  vec[3] = outVec[3];
}

void mul_mat41(double A[4][4] , double (&B)[4]){
  double C[4] = {0};
  int i;
     for (i = 0;i<4;i++)
      {C[i] = A[i][0] * B[0]
              + A[i][1] * B[1]
              + A[i][2] * B[2]
              + A[i][3] * B[3];}
      for (i = 0;i<4;i++)
		B[i]=C[i];
}
