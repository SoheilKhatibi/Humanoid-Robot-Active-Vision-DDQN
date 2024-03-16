#ifndef Transform_h_DEFINED
#define Transform_h_DEFINED

void rotX(double a, double (&matrix)[4][4]);
void rotY(double a, double (&matrix)[4][4]);
void rotZ(double a, double (&matrix)[4][4]);
void trans(double dx, double dy, double dz, double (&matrix)[4][4]);
void mul_mat44(double A[4][4] , double B[4][4], double (&Result)[4][4] );
void mul_mat41(double A[4][4] , double (&B)[4]);
void mat4x4ByVec4(double mat[4][4],double vec[4]);

#endif
