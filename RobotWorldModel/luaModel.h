#ifndef LUAMODEL_H
#define LUAMODEL_H

#include "ukfmodel.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

void Entry();
void Update(double x, double y, double a, double FootX, double SupX, double BallGPSPoseX, double BallGPSPoseY);
int HeadUpdate();
void set_camera_info(double focalLength, double focalBase, double scaleA, double scaleB, double width, double height, double radialDK1, double radialDK2, double radialDK3, double tangentialDP1, double tangentialDP2, double x_center, double y_center);
void get_image();
void update_cam(double headYaw, double headPitch);
void ProjectedPoints(double* points);
void getpose(double* pose);
UKFModel get_best_ukf();
void update_shm();
bool check_position_confidence();
void remove_waste_ukfs();
void check_ukfs_in_own_field();
void converg_ukf_models(double x, double y,double a);
void init_ukfs_own_sides(int n);
void init_ukfs_top_own_side(int n);
void init_ukfs_bottom_own_side(int n);
double get_sum_ukfs_weight();
bool is_diff_models_high(arma::mat p1,arma::mat p2,double scale=1);
UKFModel merge_two_ukfs(int i,int j);
void new_ukf(double x,double y,double a);
double circular_mean(std::vector<double> angles,std::vector<double> w);
void update_ukf_filter(UKFModel& Model);
double getTime();
bool IsThereHisModel(int number);
void updateballFilters(double GPSPoseX, double GPSPoseY);
void illustrate(double* fstatus, int ActionNumber, int ForCheck);
void SaveLocalization(int ExpEpisodeNum, int step);

#endif
