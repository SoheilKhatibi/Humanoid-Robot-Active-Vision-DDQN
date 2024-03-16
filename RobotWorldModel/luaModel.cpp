
#include <math.h>
#include <vector>

#include <iostream>
#include "luaModel.h"

#include "ukfmodel.h"
#include "BallModel.h"
#include "ActiveVision.h"
#include "MatrixTransform.h"

#include "shmProvider.h"
#include "HeadTransform.h"

#include "luatables.h"
#include "webots/robot.h"
#include "webots/camera.h"
#include "webots/position_sensor.h"

vector<UKFModel> UKFModels;



vector<BallModel> OwnBallModels;
vector<BallModel> TeammateBallModels;

using namespace std;
using namespace arma;

vector<double> uOdometry0(3,0);
vector<double> uOdometry(3,0);

double distanc_thrd;

double xLineBoundary;
double yLineBoundary;

bool webots = true;

mat Position;
vector<double> Pose(3);

double pose_tGoal = 0;

vector<double> Goal_Attack(3);
vector<double> Goal_Defend(3);

double PostAttack1X,PostAttack1Y,PostAttack2X,PostAttack2Y;
vector<double> PostAttack1(2);
vector<double> PostAttack2(2);

int role;
int InitRole = 1;
int enable_manual_positioning;
int reposition_particles;

int ManualPenaltyKick;

mat homePosition(4,2);
mat readyStartPosition(4,2);
mat PenaltyhomePosition(1,2);

double ballX,ballY;
double TeammateBallX,TeammateBallY,TeammateBallP;
double ball_confidence;


double Odom[3];
double* odomScale = Odom;
double gameState;


int enable_goal_update;
int enable_line_update;
int enable_penaltyArea_update;
int enable_parallelLine_update;
int enable_corner_update;
int enable_boundary_update;
int enable_spot;
int enable_centerT;
int enable_cornerT;
int enable_circle;

// Camera Parameters
CameraInfo cameraInfo;
double cameraMatrix[4][4];
double neckPos[4];
int horizonB;
int horizonBall;
// Camera Parameters
#define TIME_STEP 32

cv::Point3f GPSPose; 
float LocalXOffsetPose[3], RealGPSPose[3];

void Entry(){
    LuaTable input (LuaTable::fromFile("WorldConfigForCppModel.lua"));
    distanc_thrd = input["ukf"]["distanc_thrd"].getDefault<double>(false);
    enable_manual_positioning = input["enable_manual_positioning"].getDefault<double>(false);
    xLineBoundary = input["xLineBoundary"].getDefault<double>(false);
    yLineBoundary = input["yLineBoundary"].getDefault<double>(false);

    odomScale[0] = input["odomScale"][1].getDefault<double>(false);
    odomScale[1] = input["odomScale"][2].getDefault<double>(false);
    odomScale[2] = input["odomScale"][3].getDefault<double>(false);

    enable_goal_update = input["enable_goal_update"].getDefault<double>(false);
    enable_corner_update = input["enable_corner_update"].getDefault<double>(false);
    enable_boundary_update = input["enable_boundary_update"].getDefault<double>(false);
    enable_line_update = input["enable_line_update"].getDefault<double>(false);
    enable_parallelLine_update = input["enable_parallelLine_update"].getDefault<double>(false);
    enable_penaltyArea_update = input["enable_penaltyArea_update"].getDefault<double>(false);
    enable_spot = input["enable_spot"].getDefault<double>(false);
    enable_centerT = input["enable_centerT"].getDefault<double>(false);
    enable_cornerT = input["enable_cornerT"].getDefault<double>(false);
    enable_circle = input["enable_circle"].getDefault<double>(false);

    homePosition(0,0) = input["initPosition1"][1][1].getDefault<double>(false);
    homePosition(0,1) = input["initPosition1"][1][2].getDefault<double>(false);
    homePosition(1,0) = input["initPosition1"][2][1].getDefault<double>(false);
    homePosition(1,1) = input["initPosition1"][2][2].getDefault<double>(false);
    homePosition(2,0) = input["initPosition1"][3][1].getDefault<double>(false);
    homePosition(2,1) = input["initPosition1"][3][2].getDefault<double>(false);
    homePosition(3,0) = input["initPosition1"][4][1].getDefault<double>(false);
    homePosition(3,1) = input["initPosition1"][4][2].getDefault<double>(false);

    readyStartPosition(0,0) = input["readyStartPosition"][1][1].getDefault<double>(false);
    readyStartPosition(0,1) = input["readyStartPosition"][1][2].getDefault<double>(false);
    readyStartPosition(1,0) = input["readyStartPosition"][2][1].getDefault<double>(false);
    readyStartPosition(1,1) = input["readyStartPosition"][2][2].getDefault<double>(false);
    readyStartPosition(2,0) = input["readyStartPosition"][3][1].getDefault<double>(false);
    readyStartPosition(2,1) = input["readyStartPosition"][3][2].getDefault<double>(false);
    readyStartPosition(3,0) = input["readyStartPosition"][4][1].getDefault<double>(false);
    readyStartPosition(3,1) = input["readyStartPosition"][4][2].getDefault<double>(false);

    PenaltyhomePosition(0,0) = input["initPositionPenalty"][1].getDefault<double>(false);
    PenaltyhomePosition(0,1) = input["initPositionPenalty"][2].getDefault<double>(false);
    
    ManualPenaltyKick = input["ManualPenaltyKick"].getDefault<double>(false);

    Goal_Attack[0] = input["postCyan"][1][1].getDefault<double>(false);
    Goal_Attack[1] = 0;
    Goal_Attack[2] = 0;
    
    Goal_Defend[0] = input["postYellow"][1][1].getDefault<double>(false);
    Goal_Defend[1] = 0;
    Goal_Defend[2] = 0;

    PostAttack1X = input["postCyan"][1][1].getDefault<double>(false);
    PostAttack1Y = input["postCyan"][1][2].getDefault<double>(false);
    PostAttack2X = input["postCyan"][2][1].getDefault<double>(false);
    PostAttack2Y = input["postCyan"][2][2].getDefault<double>(false);
    PostAttack1[0] = PostAttack1X;
    PostAttack1[1] = PostAttack1Y;
    PostAttack2[0] = PostAttack2X;
    PostAttack2[1] = PostAttack2Y;

    vector<double> OdomScaleVec(3);
    OdomScaleVec[0] = odomScale[0];
    OdomScaleVec[1] = odomScale[1];
    OdomScaleVec[2] = odomScale[2];
    
    Pose[0] = 0;
    Pose[1] = 0;
    Pose[2] = M_PI;
    
    UKFModel firstModel;
    mat initCov(3,3);
    initCov = "0.01 0 0; 0 0.01 0; 0 0 0;";
    initCov(2,2) = pow((10*M_PI/180),2);
    init_kalmanFilter(firstModel, Pose[0], Pose[1], Pose[2], initCov, 0.0005);

    firstModel.set_weight(0.2);
    UKFModels.push_back(firstModel);

    BallModel FirstBallModel(0, 1, 0);
    OwnBallModels.push_back(FirstBallModel);

    init();
    Initiate();
    BallModelEntry();
}

void Update(double x, double y, double a, double FootX, double SupX, double BallGPSPoseX, double BallGPSPoseY){
    GPSPose.x = x;
    GPSPose.y = y;
    GPSPose.z = a;
    LocalXOffsetPose[0] = FootX - SupX;
    LocalXOffsetPose[1] = 0;
    LocalXOffsetPose[2] = 0;
    pose_global(LocalXOffsetPose, GPSPose, RealGPSPose);
    converg_ukf_models(RealGPSPose[0], y, mod_angle(a));
    
    
    
    updateballFilters(BallGPSPoseX, BallGPSPoseY);
}

UKFModel get_best_ukf(){

    UKFModel best = UKFModels[0];
    for (int i = 1 ; i<UKFModels.size(); i++){
        // cout<<"HAHAHA333333333333"<<endl;
        if (best.get_model_weight()<UKFModels[i].get_model_weight())
            best = UKFModels[i];
    }
    return best;
}

void update_shm(){

}

bool check_position_confidence(){
    UKFModel bestModel = get_best_ukf();
    mat BestModelPose = bestModel.get_pose();

    mat TempPose;  
    for(int i = 0; i<UKFModels.size(); i++){
        TempPose = UKFModels[i].get_pose();
        if  (fabs(mod_angle(BestModelPose(2, 0) - TempPose(2, 0) )) > 100*M_PI/180){
            return false;
        }
    }

    return true;
}

void check_ukfs_in_own_field(){
    mat p;
    for (int i=0;i<UKFModels.size();i++) {
        p = UKFModels[i].get_pose();
        if (p(0,0) < 0.1){
            UKFModels.erase(UKFModels.begin() + i);
        }
    }
}

void remove_waste_ukfs(){
    // remove ukfs with low weight
    
    // cout<<"Cpp counting2 "<<endl;
    // for(int i = 0 ;i<UKFModels.size();i++){
    //     cout<<UKFModels[i].get_model_weight()<<"  ";
    // }

    vector<UKFModel> newModels;
    // remove near ukfs
    mat p1;
    mat p2;                 
    for(int i=0;i<UKFModels.size();i++){   
        p1 = UKFModels[i].get_pose();
        for (int j=0;j<UKFModels.size();j++){ 
            if((i < j) && (UKFModels[i].get_model_weight() > 0 && UKFModels[j].get_model_weight() >0  )) { 
                p2 = UKFModels[j].get_pose();
                // cout<<i<<" "<<j<<" "<<is_diff_models_high(p1,p2)<<endl;;
                if (!is_diff_models_high(p1,p2)) {
                    newModels.push_back(merge_two_ukfs(i,j));
                }
            }
        }
    }
    
    // cout<<"Cpp counting2 "<<endl;
    // for(int i = 0 ;i<UKFModels.size();i++){
    //     cout<<UKFModels[i].get_model_weight()<<"  ";
    // }
    
    
    for (int k=0;k<UKFModels.size();k++){

        // cout<<"Size2: "<<k<<" "<<UKFModels.size()<<endl;
        double w = UKFModels[k].get_model_weight();
        // cout<<k<<" "<<w<<endl;
        if ( w < 0.0004 && UKFModels.size() > 1) {
            UKFModels.erase(UKFModels.begin() + k);
        }
    }

    // for(int i = 0 ;i<UKFModels.size();i++){
    //     cout<<"Cpp counting2 "<<i<<"  "<<UKFModels[i].get_model_weight()<<endl;
    // }

    for (int k = 0;k<newModels.size();k++){
        UKFModels.push_back(newModels[k]);
    }

    // for(int i = 0 ;i<UKFModels.size();i++){
    //     cout<<"Cpp counting2 "<<i<<"  "<<UKFModels[i].get_model_weight()<<endl;
    // }



}

void converg_ukf_models(double x, double y,double a){
    for (int i = 0; i<UKFModels.size();i++){
        UKFModels[i].converge(x,y,a);
    }
}

void init_ukfs_own_sides(int n){
    UKFModels.clear();
    double startX = 1;
    double endX = xLineBoundary;
    for ( int i = 0; i<(n/2);i++ ){
        double x = startX + (endX - startX)/(n/2) * i;
        new_ukf(x,yLineBoundary,270*M_PI/180);
        new_ukf(x,-yLineBoundary,90*M_PI/180);
    }
}

void init_ukfs_top_own_side(int n){
    UKFModels.clear();
    double startX = 1;
    double endX = xLineBoundary;
    for (int i = 0;i< n-1;i++){
        double x = startX + (endX - startX)/(n) * i;
        new_ukf(x,yLineBoundary,270*M_PI/180);
    }
}

void init_ukfs_bottom_own_side(int n){
    UKFModels.clear();
    double startX = 1;
    double endX = xLineBoundary;
    for (int i = 0;i<n-1;i++){
        double x = startX + (endX - startX)/(n) * i;
        new_ukf(x,-yLineBoundary,90*M_PI/180);
    }  
}

double get_sum_ukfs_weight(){
    double sumW = 0;
    for (int i = 0; i<UKFModels.size();i++){
        sumW = sumW + UKFModels[i].get_model_weight();
    }
  return sumW;
}

bool is_diff_models_high(mat p1,mat p2,double scale){
    double dx = p1(0,0) - p2(0,0);
    double dy = p1(1,0) - p2(1,0);
    double r = sqrt(pow(dx,2) + pow(dy,2));  
    if ( r > scale * distanc_thrd){
        return true;
    }
    double da = mod_angle(p1(2,0) - p2(2,0));
    
    if (da > 45*M_PI/180)
        return true;
    
    return false;
}

UKFModel merge_two_ukfs(int i,int j){
    double sumW = get_sum_ukfs_weight();
    double wi, wj;
    wi = UKFModels[i].get_model_weight();
    wj = UKFModels[j].get_model_weight();

    double nwi,nwj;
    nwi = wi / sumW;
    nwj = wj / sumW;
    double sumij = nwi + nwj;
    mat mi = UKFModels[i].get_mean();
    mat mj = UKFModels[j].get_mean();
    mat newMean(3,1);
    newMean = ( nwi * mi + nwj * mj) / sumij;
    vector<double> angles(2);
    vector<double> weights(2);
    angles[0] = mi(2,0);
    angles[1] = mj(2,0);
    weights[0] = wi;
    weights[1] = wj;
    newMean(2,0) = circular_mean(angles,weights);  
    mat erri = state_vector_diff(mi,newMean);
    mat errj = state_vector_diff(mj,newMean);
    mat covi = UKFModels[i].get_cov();
    mat covj = UKFModels[j].get_cov();
    mat newCov(3,3,fill::zeros);
    double newWeight = 0;
    if (wi>wj){
        newCov= covi;
        newWeight = wi;
    }else{
        newCov= covj;
        newWeight = wj;
    }
	UKFModels[i].set_weight(-1);
	UKFModels[j].set_weight(-1);
	UKFModel newUkf;
	init_kalmanFilter(newUkf,newMean(0,0),newMean(1,0),newMean(2,0),newCov,newWeight);
	return newUkf;
}

void new_ukf(double x,double y,double a){
    UKFModel ukf;
    
    mat initCov(3,3);
    initCov = "0.01 0 0; 0 0.01 0; 0 0 0;";
    initCov(2,2) = pow((10*M_PI/180),2);


    init_kalmanFilter(ukf,x,y,a,initCov,1);
    UKFModels.push_back(ukf);
}

double circular_mean(vector<double> angles,vector<double> w){
    double sum_sin = 0;
    double sum_cos = 0;
  
    for(int i = 0;i< angles.size();i++){
        sum_sin = sum_sin + sin(angles[i]) * w[i];
        sum_cos = sum_cos + cos(angles[i]) * w[i];
    }
  
    return atan2(sum_sin,sum_cos);
}

void update_ukf_filter(UKFModel& Model){

    // // line update
    // if (enable_line_update ==1){
    //     if(vcmLine.get<double>("detected")[0]==1){
    //         double nLines=vcmLine.get<double>("nLines")[0];
    //         double* v1x=vcmLine.get<double>("v1x");
    //         double* v1y=vcmLine.get<double>("v1y");
    //         double* v2x=vcmLine.get<double>("v2x");
    //         double* v2y=vcmLine.get<double>("v2y");
    //         double* vLength=vcmLine.get<double>("vLength");
    //         double* vAngle=vcmLine.get<double>("vAngel");
    //         for(int i=0;i<nLines;i++){
    //             Line LocalLine;

    //             LocalLine.sp = mat(3,1,fill::zeros);
    //             LocalLine.sp(0,0) = v1x[i];
    //             LocalLine.sp(1,0) = v1y[i];

    //             LocalLine.ep = mat(3,1,fill::zeros);
    //             LocalLine.ep(0,0) = v2x[i];
    //             LocalLine.ep(1,0) = v2y[i];
                
    //             LocalLine.length=vLength[i];
    //             LocalLine.angle=vAngle[i];
    //             if(vLength[i]>1.3){
    //                 Model.observe_line(LocalLine);
    //             }
    //         }
    //     }
    // }

    // // penalty area update
	// if (enable_penaltyArea_update == 1){
	// 	if (vcmPenaltyArea.get<double>("detected")[0]==1){
	// 		double* l1 = vcmPenaltyArea.get<double>("l1");
	// 		double* l2 = vcmPenaltyArea.get<double>("l2");
	// 		double angle = vcmPenaltyArea.get<double>("angle")[0];
	// 		mat intersection(2,1);
    //         intersection(0,0) = l1[2];
    //         intersection(1,0) = l1[3];
	// 		Line line1;
	// 		line1.sp = mat(3,1,fill::zeros);
    //         line1.sp(0,0) = l1[0];
    //         line1.sp(1,0) = l1[1];

	// 		line1.ep = mat(3,1,fill::zeros);
    //         line1.ep(0,0) = l1[2];
    //         line1.ep(1,0) = l1[3];

	// 		Line line2;
	// 		line2.sp = mat(3,1,fill::zeros);
    //         line2.sp(0,0) = l2[0];
    //         line2.sp(1,0) = l2[1];

	// 		line2.ep = mat(3,1,fill::zeros);
    //         line2.ep(0,0) = l2[2];
    //         line2.ep(1,0) = l2[3];

	// 		Model.observe_penalty_area(line1,line2,angle,intersection);
    //     }
    // }

    // // parallel line update
	// if (enable_parallelLine_update ==1){
	// 	if (vcmParallelLine.get<double>("detected")[0]>0){
	// 		double n=vcmParallelLine.get<double>("nParallelLine")[0];
	// 		double* v1x=vcmParallelLine.get<double>("v1x");
	// 		double* v1y=vcmParallelLine.get<double>("v1y");
	// 		double* v2x=vcmParallelLine.get<double>("v2x");
	// 		double* v2y=vcmParallelLine.get<double>("v2y");
	// 		double* vLength=vcmParallelLine.get<double>("vLength");
	// 		double* vAngle=vcmParallelLine.get<double>("vAngel");
	// 		double* type=vcmParallelLine.get<double>("type");
	// 		for (int i=0;i<n;i++){
	// 			Line line;
    //             line.sp = mat(3,1);
    //             line.sp(0,0) = v1x[i];
    //             line.sp(1,0) = v1y[i];
	// 		    line.ep = mat(3,1);
    //             line.ep(0,0) = v2x[i];
    //             line.ep(1,0) = v2y[i];
	// 			line.length=vLength[i];
	// 			line.angle=vAngle[i];
	// 			line.type=type[i];
	// 			// print(line.type,v1x[i],v1y[i],v2x[i],v2y[i])
	// 			Model.observe_parallel_lines(line);
    //         }
    //     }
    // }

    // // corener update
    // if (enable_corner_update == 1){
	// 	if (vcmLandMarks.get<double>("detected")[0] == 1){
	// 		double* l1v1x=vcmLandMarks.get<double>("l1v1x");
	// 		double* l1v1y=vcmLandMarks.get<double>("l1v1y");
	// 		double* l1v2x=vcmLandMarks.get<double>("l1v2x");
	// 		double* l1v2y=vcmLandMarks.get<double>("l1v2y");
	// 		double* l1vAngle=vcmLandMarks.get<double>("l1vAngle");
	// 		double* l1vLength=vcmLandMarks.get<double>("l1vLength");
			
	// 		double* l2v1x=vcmLandMarks.get<double>("l2v1x");
	// 		double* l2v1y=vcmLandMarks.get<double>("l2v1y");
	// 		double* l2v2x=vcmLandMarks.get<double>("l2v2x");
	// 		double* l2v2y=vcmLandMarks.get<double>("l2v2y");
	// 		double* l2vAngle=vcmLandMarks.get<double>("l2vAngle");
	// 		double* l2vLength=vcmLandMarks.get<double>("l2vLength");

	// 		double* vAngle=vcmLandMarks.get<double>("vAngle");
			
	// 		double* vIntersectionX=vcmLandMarks.get<double>("vIntersectionX");
	// 		double* vIntersectionY=vcmLandMarks.get<double>("vIntersectionY");

    //         double* cornerType=vcmLandMarks.get<double>("type");
	// 		// print("-----------------------------------------------")
	// 		// print("nCorners",vcm.get_landMarks_nLandMarks())
	// 		double nCorner=vcmLandMarks.get<double>("nLandMarks")[0];
	// 		for(int i=0;i<nCorner;i++){

	// 			Line line1;
	// 			Line line2;
	// 			double angle=vAngle[i]*M_PI/180;
	// 			mat intersection(2,1); 
    //             intersection(0,0)= vIntersectionX[i];
    //             intersection(1,0)= vIntersectionY[i];

	// 			line1.sp = mat(2,1);
    //             line1.sp(0,0) = l1v1x[i];
    //             line1.sp(1,0) = l1v1y[i];
                
    //             line1.ep = mat(2,1);
    //             line1.ep(0,0) = l1v2x[i];
    //             line1.ep(1,0) = l1v2y[i];
				
	// 			line1.angle =l1vAngle[i]*M_PI/180;
	// 			line1.length = l1vLength[i];
				
    //             line2.sp = mat(2,1);
    //             line2.sp(0,0) = l2v1x[i];
    //             line2.sp(1,0) = l2v1y[i];
                
    //             line2.ep = mat(2,1);
    //             line2.ep(0,0) = l2v2x[i];
    //             line2.ep(1,0) = l2v2y[i];
				
	// 			line2.angle =l2vAngle[i]*M_PI/180;
	// 			line2.length = l2vLength[i];
				
    //             // print("cornerType",cornerType[i])
	// 			if(cornerType[i] == 1){ //T corner
	// 			    // print("intersection = " ,intersection[1],intersection[2])
	// 				// print("intersection angle = " ,vAngle[i])
    //                 // cout<<"TTTTTTTTTTTTTTTTTTTTTT"<<endl;
	// 				Model.observe_cornerT(line1,line2,angle,intersection);
    //             }else if(cornerType[i] == 2){
	// 				// print("intersection = " ,intersection[1],intersection[2])
	// 				// print("intersection angle = " ,vAngle[i])
    //                 // cout<<"LLLLLLLLLLLLLLLLLLLLLL"<<endl;
	// 				Model.observe_cornerL(line1,line2,angle,intersection);
    //             }
    //         }
    //     }
    // }

    // // field boundary update
    // if (enable_boundary_update ==1){
    //     //print("boundary update")
    //     if(vcmBoundary.get<double>("detect")[0] == 1){
    //         Line line1;
            
    //         line1.sp = mat(3,1,fill::zeros);
    //         line1.sp(0,0) = vcmBoundary.get<double>("line1v1")[0];
    //         line1.sp(1,0) = vcmBoundary.get<double>("line1v1")[1];

    //         line1.ep = mat(3,1,fill::zeros);
    //         line1.ep(0,0) = vcmBoundary.get<double>("line1v2")[0];
    //         line1.ep(1,0) = vcmBoundary.get<double>("line1v2")[1];

    //         line1.angle = vcmBoundary.get<double>("line1Angle")[0];
    //         line1.length = vcmBoundary.get<double>("line1Length")[0];
    //         if(vcmBoundary.get<double>("nLines")[0]== 1){
    //             Model.observe_boundary_single_line(line1);
    //         }else if(vcmBoundary.get<double>("nLines")[0] == 2){
    //             Line line2;

    //             line2.sp = mat(3,1,fill::zeros);
    //             line2.sp(0,0) = vcmBoundary.get<double>("line2v1")[0];
    //             line2.sp(1,0) = vcmBoundary.get<double>("line2v1")[1];

    //             line2.ep = mat(3,1,fill::zeros);
    //             line2.ep(0,0) = vcmBoundary.get<double>("line2v2")[0];
    //             line2.ep(1,0) = vcmBoundary.get<double>("line2v2")[1];

    //             line2.angle = vcmBoundary.get<double>("line2Angle")[0];
    //             line2.length = vcmBoundary.get<double>("line2Length")[0];
    //             // print("get_boundary_detect() == 1")
    //             if(role > 0){
    //                 Model.observe_boundary_two_line(line1,line2);
    //             }
    //         }
    //     }
    // }

    // // penalty spot update
	// if (enable_spot == 1){
    //     if (vcmPenaltySpot.get<double>("detect")[0]==1){
    //         Model.observe_spot(vcmPenaltySpot.get<double>("vPoint"));                 
    //     }
    // }

    // if (enable_centerT == 1){
    //     if (vcmCenterT.get<double>("detect")[0] == 1){
    //         double* intersection = vcmCenterT.get<double>("vIntersection");
    //         mat intersectionMat(2,1);
    //         intersectionMat(0,0) = intersection[0];
    //         intersectionMat(1,0) = intersection[1];
    //         double angle = vcmCenterT.get<double>("vAngel")[0];
    //         Model.observe_centerT(intersectionMat,angle*M_PI/180);
    //     }
    // }
    
    // if (enable_circle){
        
    //     if (vcmCircle.get<double>("detect")[0]){
            
    //         mat intersectionMat(2,1);
    //         intersectionMat(0,0) = vcmCircle.get<double>("px")[0];
    //         intersectionMat(1,0) = vcmCircle.get<double>("py")[0];
    //         double angle = vcmCircle.get<double>("angle")[0];
            
    //         Model.observe_centerCircle(intersectionMat,angle*M_PI/180);
    //     }
    // }

	// if (enable_cornerT == 1){
    //     if (vcmCornerT.get<double>("detect")[0] == 1){
    //         double* intersection = vcmCornerT.get<double>("vIntersection");
    //         mat intersectionMat(2,1);
    //         intersectionMat(0,0) = intersection[0];
    //         intersectionMat(1,0) = intersection[1];
    //         double angle = vcmCornerT.get<double>("vAngel")[0];
	// 	    // print("cornerT",intersection[1],intersection[2],angle);
	// 		Model.observe_field_cornerT(intersectionMat,angle);
    //     }
    // }

}

int HeadUpdate(){
    int BestAction;
    UKFModel best = get_best_ukf();
    mat Position = best.get_pose();
    Update_Head(UKFModels, ball_confidence, OwnBallModels, Position(0,0), Position(1,0), Position(2,0), BestAction);
    return BestAction;
}

float TimeFactor(ActiveVision::Observation observation){
    float t;
    return (1-exp(t-observation.tLastObserved));
}

double getTime(){
    // if (webots) {
        // return wb_robot_get_time();
    // }
    // else {
        struct timeval t;
        gettimeofday(&t, NULL);
        return t.tv_sec + 1E-6*t.tv_usec;
    // }
}

bool IsThereHisModel(int number){
    
    return false;
}

void get_image(){
    WbDeviceTag distance_sensor, camera = wb_robot_get_device("Camera");
    wb_camera_enable(camera, TIME_STEP);
    const unsigned char* a;
    a = wb_camera_get_image(camera);

}

void updateballFilters(double GPSPoseX, double GPSPoseY){
    mat V(3,1,fill::zeros);
    double varX,varY;
    std::string headState;
    bool Detected = false;
    
    for(int i=0;i<OwnBallModels.size();i++){
        arma::mat pose = UKFModels[0].get_mean();
	    OwnBallModels[i].update(Detected, V, varX, varY, headState, GPSPoseX, GPSPoseY, pose);
    }
    
}

void set_camera_info(double focalLength, double focalBase, double scaleA, double scaleB, double width, double height, double radialDK1, double radialDK2, double radialDK3, double tangentialDP1, double tangentialDP2, double x_center, double y_center) {
    cameraInfo.focalLength = focalLength;
    cameraInfo.focalBase= focalBase;
    cameraInfo.scaleA= scaleA;
    cameraInfo.scaleB= scaleB;
    cameraInfo.width= width;
    cameraInfo.height= height;
    cameraInfo.imgXctr = x_center;
    cameraInfo.imgYctr = y_center;
    cameraInfo.radialDistortion = {radialDK1,radialDK2,radialDK3};
    cameraInfo.tangentialDistortion = {tangentialDP1,tangentialDP2};  
    cameraInfo.focalA = (cameraInfo.focalLength * cameraInfo.width) / (cameraInfo.scaleA * cameraInfo.focalBase);
    cameraInfo.focalB = cameraInfo.focalA / cameraInfo.scaleB ;
    cameraInfo.x0A = 0.5 *  cameraInfo.imgXctr;
    cameraInfo.y0A = 0.5 *  cameraInfo.imgYctr;
    //cout<<cameraInfo.width<<","<<cameraInfo.height<<","<<cameraInfo.scaleA<<","<<cameraInfo.scaleB<<","<<cameraInfo.x0A<<endl;
}

void update_cam(double headYaw, double headPitch){
    // Now bodyHeight, Tilt, camera pitch angle bias are read from vcm
    // local pTorso = mcm.get_motion_pTorso()
    
    // bodyHeight=vcm.get_camera_bodyHeight()
    // bodyTilt=vcm.get_camera_bodyTilt();

    // local imuRoll = Body.get_sensor_imuAngle(1)
    // local imuTilt = Body.get_sensor_imuAngle(2)

    Transform tNeck, tHead;
    tNeck = tNeck.translate(-(-0.035) + 0  , 0, 0.411324);
    tNeck = tNeck.rotateY(20.0*M_PI/180.0);
    tNeck = tNeck.translate(0, 0, 0.05007); // pitch0 is Robot specific head angle bias (for OP)
    tNeck = tNeck.rotateZ(-headYaw).rotateY(headPitch);
    tHead = tNeck.translate(0.02004, 0, 0.04079);

    vector<double> vHead{0, 0, 0, 1};
    vHead = tHead * vHead;
    // vHead = std::divides<double>(vHead / vHead[3]);
    
    // for (double x : vHead) 
    //     cout << x << " ";
    // cout<<endl;
    
    for(int i= 0; i<4; i++){
        for(int j=0; j<4; j++){
            cameraMatrix[i][j] = tHead(i, j);
        }
    }

}

void ProjectedPoints(double* points){
    
    ActiveVision::Point RealBoundPoints[4];
    float GlobalPose[3],Prelative[3];
    
    double v[2], IsOnField[4];
    labelAToRobotByPoint(0, 0, v, IsOnField[0]);
    RealBoundPoints[0].x = v[0];
    RealBoundPoints[0].y = v[1];
    labelAToRobotByPoint(319, 0, v, IsOnField[1]);
    RealBoundPoints[1].x = v[0];
    RealBoundPoints[1].y = v[1];
    labelAToRobotByPoint(319, 239, v, IsOnField[2]);
    RealBoundPoints[2].x = v[0];
    RealBoundPoints[2].y = v[1];
    labelAToRobotByPoint(0, 239, v, IsOnField[3]);
    RealBoundPoints[3].x = v[0];
    RealBoundPoints[3].y = v[1];
    
    // cout<<RealBoundPoints[2].x<<"    "<<RealBoundPoints[2].y<<"   "<<RealBoundPoints[3].x<<"   "<<RealBoundPoints[3].y<<endl;
    double Theta;
    if (IsOnField[0]>0){
        labelAToRobotByPoint(0, 229, v, IsOnField[0]);
        RealBoundPoints[0].x = v[0];
        RealBoundPoints[0].y = v[1];
        Theta = atan2(RealBoundPoints[0].y - RealBoundPoints[3].y, RealBoundPoints[0].x - RealBoundPoints[3].x);
        RealBoundPoints[0].x = RealBoundPoints[3].x + 20*cos(Theta);
        RealBoundPoints[0].y = RealBoundPoints[3].y + 20*sin(Theta);
    }

    if (IsOnField[1]>0){
        labelAToRobotByPoint(319, 229, v, IsOnField[1]);
        RealBoundPoints[1].x = v[0];
        RealBoundPoints[1].y = v[1];
        Theta = atan2(RealBoundPoints[1].y - RealBoundPoints[2].y, RealBoundPoints[1].x - RealBoundPoints[2].x);
        RealBoundPoints[1].x = RealBoundPoints[2].x + 20*cos(Theta);
        RealBoundPoints[1].y = RealBoundPoints[2].y + 20*sin(Theta);
    }

    for (int i=0; i<4; i++){
        points[2*i] = RealBoundPoints[i].x;
        points[2*i+1] = RealBoundPoints[i].y;
    }   

}

void getpose(double* pose){
    UKFModel best = get_best_ukf();
    mat Position = best.get_pose();
    pose[0] = Position(0, 0);
    pose[1] = Position(1, 0);
    pose[2] = Position(2, 0);
}

void illustrate(double* fstatus, int ActionNumber, int ForCheck){
    int TheBallIsInOnlineFOV;
    UKFModel best = get_best_ukf();
    mat Position = best.get_pose();
    CheckConditions(ActionNumber, OwnBallModels[0].mean(0, 0), OwnBallModels[0].mean(1, 0), Position(0,0), Position(1,0), Position(2,0), fstatus, ForCheck);
}

void SaveLocalization(int ExpEpisodeNum, int step){
    SaveShowMat(ExpEpisodeNum, step);
}