#include "ActiveVision.h"
#include "shmProvider.h"

#include "HeadTransform.h"
#include <bitset>

//ActiveVision Parameters
double P[ActionsNumber];
double YawStartPoint= 90,YawEndPoint=-90,PitchStartPoint=65,PitchEndPoint=5;
double YawStep = (YawEndPoint - YawStartPoint)/(YawNumbers-1);
double PitchStep = (PitchEndPoint - PitchStartPoint)/(PitchNumbers-1);
vector<ActiveVision::Position> Actions;
vector<ActiveVision::Observation> Observations;

ActiveVision::Observation UselessLine;

bool ActiveVisionBoundaryConfig;
bool ActiveVisionLineConfig;
bool ActiveVisionLCornerConfig;
bool ActiveVisionTCornerConfig;
bool MyPlotConfig;

double FieldX = 4.5, FieldY = 3, FieldPaddingX = 0.3, FieldPaddingY = 0.3, CornerLPoseX = FieldX, CornerLPoseY = FieldY, PenaltyAreaLPoseX = 3.5, PenaltyAreaLPoseY = 2.5;

int VisibilityStatus[(ActionsNumber) * (ObservationsNumber)];
double LineFeatures[(ActionsNumber) * (TotalLinesOfField) * 4];

int LMap1[NumOfLCorners] = {4, 7, 6, 5, 0, 8, 9, 0};
int LMap2[NumOfLCorners] = {6, 4, 5, 7, 8, 0, 0, 9};

int TMap1[NumOfTCorners] = {4, 4, 5, 5, 6, 7};
int TMap2[NumOfTCorners] = {25, 26, 27, 28, 0, 0};

cv::Mat Field;
int FieldWidth=90,FieldHeight=60,FieldPading=15,ScaleToPixel=5;
cv::Mat temp,temp1,temp2,Input;

cv::Mat ShowMat;

vector<double> EnDiffBall(ActionsNumber);
//ActiveVision Parameters

int BestActionVisibilityCode, CurrentHeadPositionVisibilityCode;

bool DrawRobot=true, DrawBall=true, DrawSupposedFOV=true, DrawPointsOnlineSHM=false, DrawPointsOnlineCpp=true, WriteActionInfo=false;

double BallIsInOnlineFOV;

void Update_Head(vector<UKFModel>& UKFModels, double& ball_confidence, vector<BallModel>& OwnBallModels, double poseX, double poseY, double poseA, int& BestAction){
    // Calculate current entropy
    double CurrentEntropy = 0;
    double weightSum = 0;
    for(int ModelCounter = 0;ModelCounter<UKFModels.size();ModelCounter++){
        weightSum = weightSum + UKFModels[ModelCounter].get_model_weight();
    }
    for(int ModelCounter = 0;ModelCounter<UKFModels.size();ModelCounter++){
        CurrentEntropy = CurrentEntropy + det(UKFModels[ModelCounter].get_cov()) * UKFModels[ModelCounter].get_model_weight()/weightSum;
    }
    // Calculate current entropy

    // Determine VisibilityStatus
    // double* pose = 0;//wcmRobot.get<double>("pose");
    double currentPositionNumber = 0;//wcmRobot.get<double>("headActionNumber")[0];
    cv::Point3f Pose;
    Pose.x = poseX;
    Pose.y = poseY;
    Pose.z = poseA;
    arma::mat uPose(3,1);
    uPose(0,0) = poseX;
    uPose(1,0) = poseY;
    uPose(2,0) = poseA;
    int BestActionNumber = TheBestAction(currentPositionNumber, Pose, OwnBallModels[0].mean(0, 0), OwnBallModels[0].mean(1, 0), ball_confidence, OwnBallModels);
    // Determine VisibilityStatus
    
    vector<double> ModelCovDets(UKFModels.size());
    vector<double> Entropy(ActionsNumber);
    for(int AcNum=0; AcNum<ActionsNumber; AcNum++){
        vector<UKFModel> TestingModel(UKFModels);
        // TestingModel[0].get_cov().print("Before:");
        // cout<<"For Action "<<AcNum<<": "<<endl;
        for(int ModelCounter = 0; ModelCounter<TestingModel.size();ModelCounter++){
            int FeatureCounter;
            arma::mat LocalLine;
            arma::mat tempPoint(3,1);
            arma::mat tempPoint1(3,1);
            arma::mat tempPoint2(3,1);

            if (ActiveVisionBoundaryConfig){
                int FbStatus[4] = {VisibilityStatus[(AcNum) * ObservationsNumber], VisibilityStatus[(AcNum) * ObservationsNumber + 1], VisibilityStatus[(AcNum) * ObservationsNumber + 2], VisibilityStatus[(AcNum) * ObservationsNumber + 3]};
                int Fbitr = 0, FbFeatureCounter[2];
                for (int i = 0; i<NumOfFieldBoundaryLines; i++){
                    if (FbStatus[i] == 1){
                        FbFeatureCounter[Fbitr] = ( (AcNum) * 4 * TotalLinesOfField ) + i * 4;
                        Fbitr++;
                    }
                }
                int FbCounter = FbStatus[0] + FbStatus[1] + FbStatus[2] + FbStatus[3];
                if (FbCounter > 0){

                    // if (FbHorizental == 0){
                        // FeatureCounter = FeatureCounter + 4;
                    // }

                    Line line1;
                    
                    line1.sp = arma::mat(3, 1, arma::fill::zeros);
                    line1.ep = arma::mat(3, 1, arma::fill::zeros);
                    
                    tempPoint(0,0) = LineFeatures[FbFeatureCounter[0]];
                    tempPoint(1,0) = LineFeatures[FbFeatureCounter[0]+1];
                    tempPoint(2,0) = 0;
                    
                    LocalLine = pose_relative(tempPoint, uPose);
                    line1.sp(0,0) = LocalLine(0,0);
                    line1.sp(1,0) = LocalLine(1,0);

                    tempPoint(0,0) = LineFeatures[FbFeatureCounter[0]+2];
                    tempPoint(1,0) = LineFeatures[FbFeatureCounter[0]+3];
                    tempPoint(2,0) = 0;
                    
                    LocalLine = pose_relative(tempPoint, uPose);
                    line1.ep(0,0) = LocalLine(0,0);
                    line1.ep(1,0) = LocalLine(1,0);
                    
                    if (FbCounter == 1 ){
                        // cout<<"Fb1"<<endl;
                        TestingModel[ModelCounter].observe_boundary_single_line(line1);
                    }else if (FbCounter == 2 ){
                        // cout<<"Fb2"<<endl;
                        // FeatureCounter = FeatureCounter + 4;
                        Line line2;
                    
                        line2.sp = arma::mat(3,1,arma::fill::zeros);
                        line2.ep = arma::mat(3,1,arma::fill::zeros);
                    
                        tempPoint(0,0) = LineFeatures[FbFeatureCounter[1]];
                        tempPoint(1,0) = LineFeatures[FbFeatureCounter[1]+1];
                        tempPoint(2,0) = 0;

                        LocalLine = pose_relative(tempPoint, uPose);
                        line2.sp(0,0) = LocalLine(0,0);
                        line2.sp(1,0) = LocalLine(1,0);

                        tempPoint(0,0) = LineFeatures[FbFeatureCounter[1]+2];
                        tempPoint(1,0) = LineFeatures[FbFeatureCounter[1]+3];
                        tempPoint(2,0) = 0;
                    
                        LocalLine = pose_relative(tempPoint, uPose);
                        line2.ep(0,0) = LocalLine(0,0);
                        line2.ep(1,0) = LocalLine(1,0);

                        if ((line1.sp(0,0) == line2.sp(0,0)) && (line1.sp(1,0) == line2.sp(1,0))){
                            tempPoint = line1.sp;
                            line1.sp = line1.ep;
                            line1.ep = tempPoint;
                        }else if ((line1.sp(0,0) == line2.ep(0,0)) && (line1.sp(1,0) == line2.ep(1,0))){
                            tempPoint = line2.sp;
                            line2.sp = line1.sp;
                            line1.sp = tempPoint;
                            
                            tempPoint = line2.ep;
                            line2.ep = line1.ep;
                            line1.ep = tempPoint;
                        }else if ((line1.ep(0,0) == line2.ep(0,0)) && (line1.ep(1,0) == line2.ep(1,0))){

                            tempPoint = line2.sp;
                            line2.sp = line2.ep;
                            line2.ep = tempPoint;
                        }

                        if(role > 0){
                            TestingModel[ModelCounter].observe_boundary_two_line(line1,line2);
                        }
                    }
                }
            }

            if (ActiveVisionLineConfig){
                // line update
                for (int Nlines=NumOfFieldBoundaryLines; Nlines<TotalLinesOfField; Nlines++){
                    int LineVisibility = VisibilityStatus[(AcNum) * ObservationsNumber + Nlines];
                    if (LineVisibility==1 ){
                        // cout<<"Line "<<Nlines<<endl;
                        FeatureCounter = ( (AcNum) * 4 * TotalLinesOfField ) + Nlines*4;
                        Line line;
                        line.sp = arma::mat(3,1,arma::fill::zeros);
                        line.ep = arma::mat(3,1,arma::fill::zeros);
                        tempPoint(0,0) = LineFeatures[FeatureCounter];
                        tempPoint(1,0) = LineFeatures[FeatureCounter+1];
                        tempPoint(2,0) = 0;
                        LocalLine = pose_relative(tempPoint, uPose);
                        line.sp(0,0) = LocalLine(0,0);
                        line.sp(1,0) = LocalLine(1,0);
                        tempPoint(0,0) = LineFeatures[FeatureCounter+2];
                        tempPoint(1,0) = LineFeatures[FeatureCounter+3];
                        tempPoint(2,0) = 0;
                        LocalLine = pose_relative(tempPoint, uPose);
                        line.ep(0,0) = LocalLine(0,0);
                        line.ep(1,0) = LocalLine(1,0);
                        TestingModel[ModelCounter].observe_line(line);
                    }
                }
            }
            
            if (ActiveVisionLCornerConfig){
                // L corener update
                for (int LLandMarkCounter=TotalLinesOfField; LLandMarkCounter<TotalLinesOfField+NumOfLCorners ; LLandMarkCounter++){
                    if (VisibilityStatus[(AcNum * ObservationsNumber) + LLandMarkCounter] == 1){
                        // cout<<"L"<<endl;
                        FeatureCounter = ( (AcNum) * 4 * TotalLinesOfField ) + LMap1[LLandMarkCounter-TotalLinesOfField]*4;
                        tempPoint1(0,0) = LineFeatures[FeatureCounter];
                        tempPoint1(1,0) = LineFeatures[FeatureCounter+1];
                        tempPoint1(2,0) = 0;
                        tempPoint2(0,0) = LineFeatures[FeatureCounter+2];
                        tempPoint2(1,0) = LineFeatures[FeatureCounter+3];
                        tempPoint2(2,0) = 0;

                        if (LLandMarkCounter == 15){
                            tempPoint1(0,0) = Observations[25].startPoint.x;
                            tempPoint1(1,0) = Observations[25].startPoint.y;
                            tempPoint1(2,0) = 0;
                            tempPoint2(0,0) = Observations[25].EndPoint.x;
                            tempPoint2(1,0) = Observations[25].EndPoint.y;
                            tempPoint2(2,0) = 0;
                        }else if (LLandMarkCounter == 18){
                            tempPoint1(0,0) = Observations[28].startPoint.x;
                            tempPoint1(1,0) = Observations[28].startPoint.y;
                            tempPoint1(2,0) = 0;
                            tempPoint2(0,0) = Observations[28].EndPoint.x;
                            tempPoint2(1,0) = Observations[28].EndPoint.y;
                            tempPoint2(2,0) = 0;
                        }

                        Line line1;
                        line1.sp = arma::mat(3,1,arma::fill::zeros);
                        line1.ep = arma::mat(3,1,arma::fill::zeros);
                        LocalLine = pose_relative(tempPoint1, uPose);
                        line1.sp(0,0) = LocalLine(0,0);
                        line1.sp(1,0) = LocalLine(1,0);
                        LocalLine = pose_relative(tempPoint2, uPose);
                        line1.ep(0,0) = LocalLine(0,0);
                        line1.ep(1,0) = LocalLine(1,0);

                        FeatureCounter = ( (AcNum) * 4 * TotalLinesOfField ) + LMap2[LLandMarkCounter-TotalLinesOfField]*4;
                        tempPoint1(0,0) = LineFeatures[FeatureCounter];
                        tempPoint1(1,0) = LineFeatures[FeatureCounter+1];
                        tempPoint1(2,0) = 0;
                        tempPoint2(0,0) = LineFeatures[FeatureCounter+2];
                        tempPoint2(1,0) = LineFeatures[FeatureCounter+3];
                        tempPoint2(2,0) = 0;

                        if (LLandMarkCounter == 16){
                            tempPoint1(0,0) = Observations[26].startPoint.x;
                            tempPoint1(1,0) = Observations[26].startPoint.y;
                            tempPoint1(2,0) = 0;
                            tempPoint2(0,0) = Observations[26].EndPoint.x;
                            tempPoint2(1,0) = Observations[26].EndPoint.y;
                            tempPoint2(2,0) = 0;
                        }else if (LLandMarkCounter == 17){
                            tempPoint1(0,0) = Observations[27].startPoint.x;
                            tempPoint1(1,0) = Observations[27].startPoint.y;
                            tempPoint1(2,0) = 0;
                            tempPoint2(0,0) = Observations[27].EndPoint.x;
                            tempPoint2(1,0) = Observations[27].EndPoint.y;
                            tempPoint2(2,0) = 0;
                        }
                        Line line2;
                        line2.sp = arma::mat(3,1,arma::fill::zeros);
                        line2.ep = arma::mat(3,1,arma::fill::zeros);
                        LocalLine = pose_relative(tempPoint1, uPose);
                        line2.sp(0,0) = LocalLine(0,0);
                        line2.sp(1,0) = LocalLine(1,0);
                        LocalLine = pose_relative(tempPoint2, uPose);
                        line2.ep(0,0) = LocalLine(0,0);
                        line2.ep(1,0) = LocalLine(1,0);
                        if ((line1.sp(0,0) == line2.sp(0,0)) && (line1.sp(1,0) == line2.sp(1,0))){
                            tempPoint = line1.sp;
                            line1.sp = line1.ep;
                            line1.ep = tempPoint;
                        }else if ((line1.sp(0,0) == line2.ep(0,0)) && (line1.sp(1,0) == line2.ep(1,0))){
                            tempPoint = line2.sp;
                            line2.sp = line1.sp;
                            line1.sp = tempPoint;
                            
                            tempPoint = line2.ep;
                            line2.ep = line1.ep;
                            line1.ep = tempPoint;
                        }else if ((line1.ep(0,0) == line2.ep(0,0)) && (line1.ep(1,0) == line2.ep(1,0))){

                            tempPoint = line2.sp;
                            line2.sp = line2.ep;
                            line2.ep = tempPoint;
                        }
                        double angle1 = atan2(line1.sp(1,0) - line1.ep(1,0),line1.sp(0,0) - line1.ep(0,0));
                        double angle2 = atan2(line2.ep(1,0) - line2.sp(1,0),line2.ep(0,0) - line2.sp(0,0));
                        double angle;
                        arma::mat intersection(2,1); 
                        intersection(0,0)= line1.ep(0,0);
                        intersection(1,0)= line1.ep(1,0);
                        if ((angle1*angle2>=0) || (abs(angle2)+ abs(angle1)<=M_PI) ){
                            angle=(angle1+angle2)/2;
                        }else{
                            angle=(angle1+angle2)/2;
                            if (angle>0){
                                angle=angle-M_PI;
                            }else{
                                angle=angle+M_PI;
                            }
                        }
                        TestingModel[ModelCounter].observe_cornerL(line1,line2,angle,intersection);
                    }
                }

            }

            if (ActiveVisionTCornerConfig){
                // T corener update
                for (int TLandMarkCounter=TotalLinesOfField+NumOfLCorners; TLandMarkCounter<ObservationsNumber ; TLandMarkCounter++){
                    if (VisibilityStatus[(AcNum * ObservationsNumber) + TLandMarkCounter] == 1){

                        // cout<<"T"<<endl;
                        FeatureCounter = ( (AcNum) * 4 * TotalLinesOfField ) + TMap1[TLandMarkCounter-(TotalLinesOfField+NumOfLCorners)]*4 ;
                        tempPoint1(0,0) = LineFeatures[FeatureCounter];
                        tempPoint1(1,0) = LineFeatures[FeatureCounter+1];
                        tempPoint1(2,0) = 0;
                        tempPoint2(0,0) = LineFeatures[FeatureCounter+2];
                        tempPoint2(1,0) = LineFeatures[FeatureCounter+3];
                        tempPoint2(2,0) = 0;                        

                        Line line1;

                        line1.sp = arma::mat(3,1,arma::fill::zeros);
                        line1.ep = arma::mat(3,1,arma::fill::zeros);

                        // tempPoint.t().print("1");

                        LocalLine = pose_relative(tempPoint1, uPose);
                        line1.sp(0,0) = LocalLine(0,0);
                        line1.sp(1,0) = LocalLine(1,0);
                        

                        // tempPoint.t().print("2");

                        LocalLine = pose_relative(tempPoint2, uPose);
                        line1.ep(0,0) = LocalLine(0,0);
                        line1.ep(1,0) = LocalLine(1,0);
                        
                        FeatureCounter = TMap2[TLandMarkCounter-(TotalLinesOfField+NumOfLCorners)];
                        tempPoint1(0,0) = Observations[FeatureCounter].startPoint.x;
                        tempPoint1(1,0) = Observations[FeatureCounter].startPoint.y;
                        tempPoint1(2,0) = 0;
                        tempPoint2(0,0) = Observations[FeatureCounter].EndPoint.x;
                        tempPoint2(1,0) = Observations[FeatureCounter].EndPoint.y;
                        tempPoint2(2,0) = 0;
                        if (TLandMarkCounter == 23 || TLandMarkCounter == 24){
                            FeatureCounter = ( (AcNum) * 4 * TotalLinesOfField ) + 10*4 ;
                            tempPoint1(0,0) = LineFeatures[FeatureCounter];
                            tempPoint1(1,0) = LineFeatures[FeatureCounter+1];
                            tempPoint1(2,0) = 0;
                            tempPoint2(0,0) = LineFeatures[FeatureCounter+2];
                            tempPoint2(1,0) = LineFeatures[FeatureCounter+3];
                            tempPoint2(2,0) = 0;
                        }
                        
                        Line line2;
                        line2.sp = arma::mat(3,1,arma::fill::zeros);
                        line2.ep = arma::mat(3,1,arma::fill::zeros);

                        LocalLine = pose_relative(tempPoint1, uPose);
                        line2.sp(0,0) = LocalLine(0,0);
                        line2.sp(1,0) = LocalLine(1,0);
                        
                        LocalLine = pose_relative(tempPoint2, uPose);
                        line2.ep(0,0) = LocalLine(0,0);
                        line2.ep(1,0) = LocalLine(1,0);

                        
                        if (( sqrt (pow(line2.sp(1,0)-line1.sp(1,0),2) + pow(line2.sp(0,0)-line1.sp(0,0),2)) > sqrt(pow((line2.ep(1,0)-line1.sp(1,0)),2) + pow((line2.ep(0,0)-line1.sp(0,0)),2)))) {
                            tempPoint = line2.sp;
                            line2.sp = line2.ep;
                            line2.ep = tempPoint;
                        }
                        
                        
                        arma::mat intersection(2,1); 
                        intersection(0,0)= line2.sp(0,0);
                        intersection(1,0)= line2.sp(1,0);
                        
                        ActiveVision::Point Q,S,E;
                        ActiveVision::Line M,N;

                        S.x = line1.sp(0,0);
                        S.y = line1.sp(1,0);

                        E.x = line1.ep(0,0);
                        E.y = line1.ep(1,0);
                        M = Equa(S,E);

                        S.x = line2.sp(0,0);
                        S.y = line2.sp(1,0);

                        E.x = line2.ep(0,0);
                        E.y = line2.ep(1,0);
                        N = Equa(S,E);

                        int ad = Intersection(M, N, &Q);
                        if ( sqrt( pow( Q.x-line2.sp(0,0) ,2) + pow(Q.y-line2.sp(1,0),2)) >  sqrt( pow (Q.x-line2.ep(0,0),2) + pow (Q.y-line2.ep(1,0),2)))  {
                            tempPoint = line2.ep;
                            line2.ep = line2.sp;
                            line2.sp = tempPoint;
                        }
                        double angle=atan2(line2.sp(1,0) - line2.ep(1,0),line2.sp(0,0) - line2.ep(0,0));

                        intersection(0,0)= line2.sp(0,0);
                        intersection(1,0)= line2.sp(1,0);

                        // line2.sp->print("2S");
                        // line2.ep->print("2E");


                        // cout<<"For Action "<<AcNum<<": "<<endl;
                        // intersection.print("intersection");
                        // cout<<angle*180/M_PI<<endl;

                        
                        TestingModel[ModelCounter].observe_cornerT(line1,line2,angle,intersection);
                    }
                }
            }

            arma::mat cCov = TestingModel[ModelCounter].get_cov();
            ModelCovDets[ModelCounter] = det(cCov);
        }

        double weightSum2 = 0;
        for (int ModelCounter=0; ModelCounter<TestingModel.size(); ModelCounter++){
            weightSum2 = weightSum2 + TestingModel[ModelCounter].get_model_weight();
        }
        
        Entropy[AcNum] = 0;
        
        for (int ModelCounter = 0;ModelCounter< TestingModel.size();ModelCounter++){
            Entropy[AcNum] = Entropy[AcNum] + ModelCovDets[ModelCounter] * TestingModel[ModelCounter].get_model_weight()/weightSum2;
        }
        // TestingModel[0].get_cov().print("After:");
        // cout<<Entropy[AcNum]<<endl;
        
    }
    double MaxEnDiff = -100000000;
    BestActionNumber = -1;
    vector<double> EnDiffLocalization(ActionsNumber);
    vector<double> EnDiff(ActionsNumber);
    for(int AcNum=0; AcNum<ActionsNumber; AcNum++){
        // cout<<"  "<<AcNum<<endl;
        EnDiffLocalization[AcNum] = CurrentEntropy - Entropy[AcNum];
        
        EnDiff[AcNum] = EnDiffLocalization[AcNum] + EnDiffBall[AcNum];
        
        // if (EnDiff[AcNum]!=0){
            // cout<<AcNum<<"   "<<EnDiff[AcNum]<<" salam "<<endl;
        // }
        
        if (EnDiff[AcNum] >= MaxEnDiff){
            BestActionNumber = AcNum;
            MaxEnDiff = EnDiff[AcNum];
        }
    }

    // if (MyPlotConfig){
        // for (int L=0;L<ActionsNumber;L++){
            // Detecteds(L, Pose, "Field", OwnBallModels[0].mean(0, 0), OwnBallModels[0].mean(1, 0), ball_confidence);
        // }
        // Detecteds(BestActionNumber, Pose, "Final", OwnBallModels[0].mean(0, 0), OwnBallModels[0].mean(1, 0), ball_confidence);
    // }
    // cout<<"BestActionNumber :   "<<BestActionNumber<<"   "<<MaxEnDiff<<endl;
    BestActionVisibilityCode = 0;
    for (int i=0; i<ObservationsNumber; i++){
        if (VisibilityStatus[(BestActionNumber) * ObservationsNumber + i] == 1){
            // cout<<"AA"<<i<<"\t";
            BestActionVisibilityCode = BestActionVisibilityCode + pow(2, i);
        }
    }
    // wcmRobot.set("headActionNumber",BestActionNumber);
    BestAction = BestActionNumber;
    // cout<<"BestAction:  "<<BestAction<<endl;
}

void Initiate(){

    LuaTable input (LuaTable::fromFile("WorldConfigForCppModel.lua"));
    ActiveVisionBoundaryConfig = input["ActiveVisionBoundaryConfig"].getDefault<bool>(false);
    ActiveVisionLineConfig = input["ActiveVisionLineConfig"].getDefault<bool>(false);
    ActiveVisionLCornerConfig = input["ActiveVisionLCornerConfig"].getDefault<bool>(false);
    ActiveVisionTCornerConfig = input["ActiveVisionTCornerConfig"].getDefault<bool>(false);
    MyPlotConfig = input["MyPlotConfig"].getDefault<bool>(false);

    Actions.resize(ActionsNumber);
    Observations.resize(ObservationsNumber + NumOfUselessLines);
    ifstream BOUNDS;
    BOUNDS.open("./soheil.txt",ios::in);
    string A;
      //Define Actions and Observations
    for(int x=0;x<ActionsNumber;x++){
        
        Actions[x].Yaw = YawStartPoint + (x%YawNumbers)*YawStep;
        // std::cout<<YawStartPoint<<"  "<<x%YawNumbers<<"  "<<YawStep<<std::endl;
        Actions[x].Pitch = PitchStartPoint + (x/YawNumbers)*PitchStep;
        Actions[x].number = x;
        // std::cout<<"Action "<<x<<" "<<Actions[x].Yaw<<"  "<<Actions[x].Pitch<<"  "<<Actions[x].number<<std::endl;
        BOUNDS>>A;
        BOUNDS>>A;
        BOUNDS>>Actions[x].BounPointz[0].x;
        BOUNDS>>Actions[x].BounPointz[0].y;
        // std::cout<<Actions[x].BounPointz[0].x<<" "<<Actions[x].BounPointz[0].y<<std::endl;
        BOUNDS>>Actions[x].BounPointz[1].x;
        BOUNDS>>Actions[x].BounPointz[1].y;
        // std::cout<<Actions[x].BounPointz[1].x<<" "<<Actions[x].BounPointz[1].y<<std::endl;
        BOUNDS>>Actions[x].BounPointz[2].x;
        BOUNDS>>Actions[x].BounPointz[2].y;
        // std::cout<<Actions[x].BounPointz[2].x<<" "<<Actions[x].BounPointz[2].y<<std::endl;
        BOUNDS>>Actions[x].BounPointz[3].x;
        BOUNDS>>Actions[x].BounPointz[3].y;
        // std::cout<<Actions[x].BounPointz[3].x<<" "<<Actions[x].BounPointz[3].y<<std::endl;
        BOUNDS>>A;
    }
    
    for(int y=0;y<ObservationsNumber;y++){
        Observations[y].number = y;
        Observations[y].tLastObserved = 0;
    }
    
    Observations[0].name = "fieldBoundaryHOwn";
    Observations[0].Type = "Line";

    Observations[0].startPoint.x = FieldX + FieldPaddingX;
    Observations[0].startPoint.y = FieldY + FieldPaddingY;
    
    Observations[0].EndPoint.x = FieldX + FieldPaddingX;
    Observations[0].EndPoint.y = -(FieldY + FieldPaddingY);
    


    Observations[1].name = "fieldBoundaryHOponent";
    Observations[1].Type = "Line";

    Observations[1].startPoint.x = -(FieldX + FieldPaddingX);
    Observations[1].startPoint.y = FieldY + FieldPaddingY;
    
    Observations[1].EndPoint.x = -(FieldX + FieldPaddingX);
    Observations[1].EndPoint.y = -(FieldY + FieldPaddingY);



    Observations[2].name = "fieldBoundaryVLeft";
    Observations[2].Type = "Line";

    Observations[2].startPoint.x = FieldX + FieldPaddingX;
    Observations[2].startPoint.y = FieldY + FieldPaddingY;
    
    Observations[2].EndPoint.x = -(FieldX + FieldPaddingX);
    Observations[2].EndPoint.y = FieldY + FieldPaddingY;
    


    Observations[3].name = "fieldBoundaryVRight";
    Observations[3].Type = "Line";

    Observations[3].startPoint.x = FieldX + FieldPaddingX;
    Observations[3].startPoint.y = -(FieldY + FieldPaddingY);
    
    Observations[3].EndPoint.x = -(FieldX + FieldPaddingX);
    Observations[3].EndPoint.y = -(FieldY + FieldPaddingY);
    


    Observations[4].name = "BoundLineHOwn";
    Observations[4].Type = "Line";

    Observations[4].startPoint.x = FieldX;
    Observations[4].startPoint.y = FieldY;
    
    Observations[4].EndPoint.x = FieldX;
    Observations[4].EndPoint.y = -FieldY;



    Observations[5].name = "BoundLineHOponent";
    Observations[5].Type = "Line";

    Observations[5].startPoint.x = -FieldX;
    Observations[5].startPoint.y = FieldY;
    
    Observations[5].EndPoint.x = -FieldX;
    Observations[5].EndPoint.y = -FieldY;
    


    Observations[6].name = "BoundLineVLeft";
    Observations[6].Type = "Line";

    Observations[6].startPoint.x = FieldX;
    Observations[6].startPoint.y = FieldY;
    
    Observations[6].EndPoint.x = -FieldX;
    Observations[6].EndPoint.y = FieldY;



    Observations[7].name = "BoundLineVRight";
    Observations[7].Type = "Line";

    Observations[7].startPoint.x = FieldX;
    Observations[7].startPoint.y = -FieldY;
    
    Observations[7].EndPoint.x = -FieldX;
    Observations[7].EndPoint.y = -FieldY;
    


    Observations[8].name = "PenaltyAreaLineHOwn";
    Observations[8].Type = "Line";

    Observations[8].startPoint.x = PenaltyAreaLPoseX;
    Observations[8].startPoint.y = PenaltyAreaLPoseY;
    
    Observations[8].EndPoint.x = PenaltyAreaLPoseX;
    Observations[8].EndPoint.y = -PenaltyAreaLPoseY;



    Observations[9].name = "PenaltyAreaLineHOponent";
    Observations[9].Type = "Line";

    Observations[9].startPoint.x = -PenaltyAreaLPoseX;
    Observations[9].startPoint.y = PenaltyAreaLPoseY;
    
    Observations[9].EndPoint.x = -PenaltyAreaLPoseX;
    Observations[9].EndPoint.y = -PenaltyAreaLPoseY;



    Observations[10].name = "MiddleLine";
    Observations[10].Type = "Line";

    Observations[10].startPoint.x = 0;
    Observations[10].startPoint.y = FieldY;
    
    Observations[10].EndPoint.x = 0;
    Observations[10].EndPoint.y = -FieldY;
    


    Observations[11].name = "CornerLOwnLeft";
    Observations[11].Type = "Point";

    Observations[11].startPoint.x = CornerLPoseX;
    Observations[11].startPoint.y = CornerLPoseY;



    Observations[12].name = "CornerLOwnRight";
    Observations[12].Type = "Point";

    Observations[12].startPoint.x = CornerLPoseX;
    Observations[12].startPoint.y = -CornerLPoseY;



    Observations[13].name = "CornerLOponentLeft";
    Observations[13].Type = "Point";

    Observations[13].startPoint.x = -CornerLPoseX;
    Observations[13].startPoint.y = CornerLPoseY;



    Observations[14].name = "CornerLOponentRight";
    Observations[14].Type = "Point";

    Observations[14].startPoint.x = -CornerLPoseX;
    Observations[14].startPoint.y = -CornerLPoseY;



    Observations[15].name = "PenaltyAreaLOwnLeft";
    Observations[15].Type = "Point";

    Observations[15].startPoint.x = PenaltyAreaLPoseX;
    Observations[15].startPoint.y = PenaltyAreaLPoseY;



    Observations[16].name = "PenaltyAreaLOwnRight";
    Observations[16].Type = "Point";

    Observations[16].startPoint.x = PenaltyAreaLPoseX;
    Observations[16].startPoint.y = -PenaltyAreaLPoseY;



    Observations[17].name = "PenaltyAreaLOponentLeft";
    Observations[17].Type = "Point";

    Observations[17].startPoint.x = -PenaltyAreaLPoseX;
    Observations[17].startPoint.y = PenaltyAreaLPoseY;



    Observations[18].name = "PenaltyAreaLOponentRight";
    Observations[18].Type = "Point";

    Observations[18].startPoint.x = -PenaltyAreaLPoseX;
    Observations[18].startPoint.y = -PenaltyAreaLPoseY;


    
    Observations[19].name = "PenaltyAreaTOwnLeft";
    Observations[19].Type = "Point";

    Observations[19].startPoint.x = FieldX;
    Observations[19].startPoint.y = PenaltyAreaLPoseY;



    Observations[20].name = "PenaltyAreaTOwnRight";
    Observations[20].Type = "Point";

    Observations[20].startPoint.x = FieldX;
    Observations[20].startPoint.y = -PenaltyAreaLPoseY;



    Observations[21].name = "PenaltyAreaTOponentLeft";
    Observations[21].Type = "Point";

    Observations[21].startPoint.x = -FieldX;
    Observations[21].startPoint.y = PenaltyAreaLPoseY;



    Observations[22].name = "PenaltyAreaTOponentRight";
    Observations[22].Type = "Point";

    Observations[22].startPoint.x = -FieldX;
    Observations[22].startPoint.y = -PenaltyAreaLPoseY;


    
    Observations[23].name = "MiddleTLeft";
    Observations[23].Type = "Point";

    Observations[23].startPoint.x = 0;
    Observations[23].startPoint.y = FieldY;



    Observations[24].name = "MiddleTRight";
    Observations[24].Type = "Point";

    Observations[24].startPoint.x = 0;
    Observations[24].startPoint.y = -FieldY;

    Observations[25].name = "OwnLeftUselessLine";
    Observations[25].Type = "Line";

    Observations[25].startPoint.x = FieldX;
    Observations[25].startPoint.y = PenaltyAreaLPoseY;
    
    Observations[25].EndPoint.x = PenaltyAreaLPoseX;
    Observations[25].EndPoint.y = PenaltyAreaLPoseY;

    Observations[26].name = "OwnRightUselessLine";
    Observations[26].Type = "Line";

    Observations[26].startPoint.x = FieldX;
    Observations[26].startPoint.y = -PenaltyAreaLPoseY;
    
    Observations[26].EndPoint.x = PenaltyAreaLPoseX;
    Observations[26].EndPoint.y = -PenaltyAreaLPoseY;

    Observations[27].name = "OponentLeftUselessLine";
    Observations[27].Type = "Line";

    Observations[27].startPoint.x = -FieldX;
    Observations[27].startPoint.y = PenaltyAreaLPoseY;
    
    Observations[27].EndPoint.x = -PenaltyAreaLPoseX;
    Observations[27].EndPoint.y = PenaltyAreaLPoseY;

    Observations[28].name = "OponentRightUselessLine";
    Observations[28].Type = "Line";

    Observations[28].startPoint.x = -FieldX;
    Observations[28].startPoint.y = -PenaltyAreaLPoseY;
    
    Observations[28].EndPoint.x = -PenaltyAreaLPoseX;
    Observations[28].EndPoint.y = -PenaltyAreaLPoseY;
    
    //
    Field = cv::Mat((FieldHeight+2*FieldPading)*ScaleToPixel,(FieldWidth+2*FieldPading)*ScaleToPixel,CV_8UC3,cv::Scalar(34,139,34));
    cv::Point CUL=cv::Point(FieldPading*ScaleToPixel,FieldPading*ScaleToPixel);
    cv::Point CUR((FieldWidth+FieldPading)*ScaleToPixel,FieldPading*ScaleToPixel);
    cv::Point CDL(FieldPading*ScaleToPixel,(FieldHeight+FieldPading)*ScaleToPixel);
    cv::Point CDR((FieldWidth+FieldPading)*ScaleToPixel,(FieldHeight+FieldPading)*ScaleToPixel);
    cv::Point CLineU ((CUR.x+CUL.x)/2,CUR.y);
    cv::Point CLineD ((CUR.x+CUL.x)/2,CDR.y);
    cv::line(Field,CUL,CUR,cv::Scalar(255,255,255),0.5*ScaleToPixel);
    cv::line(Field,CUL,CDL,cv::Scalar(255,255,255),0.5*ScaleToPixel);
    cv::line(Field,CDL,CDR,cv::Scalar(255,255,255),0.5*ScaleToPixel);
    cv::line(Field,CDR,CUR,cv::Scalar(255,255,255),0.5*ScaleToPixel);

    cv::line(Field,cv::Point(CUR.x,CUR.y+(5*ScaleToPixel)),cv::Point(CUR.x-(10*ScaleToPixel),CUR.y+(5*ScaleToPixel)),cv::Scalar(255,255,255),0.5*ScaleToPixel);
    cv::line(Field,cv::Point(CDR.x,CDR.y-(5*ScaleToPixel)),cv::Point(CDR.x-(10*ScaleToPixel),CDR.y-(5*ScaleToPixel)),cv::Scalar(255,255,255),0.5*ScaleToPixel);
    cv::line(Field,cv::Point(CDR.x-(10*ScaleToPixel),CDR.y-(5*ScaleToPixel)),cv::Point(CUR.x-(10*ScaleToPixel),CUR.y+(5*ScaleToPixel)),cv::Scalar(255,255,255),0.5*ScaleToPixel);

    cv::line(Field,cv::Point(CUL.x,CUL.y+(5*ScaleToPixel)),cv::Point(CUL.x+(10*ScaleToPixel),CUL.y+(5*ScaleToPixel)),cv::Scalar(255,255,255),0.5*ScaleToPixel);
    cv::line(Field,cv::Point(CDL.x,CDL.y-(5*ScaleToPixel)),cv::Point(CDL.x+(10*ScaleToPixel),CDL.y-(5*ScaleToPixel)),cv::Scalar(255,255,255),0.5*ScaleToPixel);
    cv::line(Field,cv::Point(CDL.x+(10*ScaleToPixel),CDL.y-(5*ScaleToPixel)),cv::Point(CUL.x+(10*ScaleToPixel),CUL.y+(5*ScaleToPixel)),cv::Scalar(255,255,255),0.5*ScaleToPixel);
    cv::line(Field,CLineU,CLineD,cv::Scalar(255,255,255),0.5*ScaleToPixel);

    cv::line(Field,cv::Point(CUR.x-22*ScaleToPixel,(CUR.y+CDR.y)/2),cv::Point(CUR.x-22*ScaleToPixel,(CUR.y+CDR.y)/2),cv::Scalar(255,255,255),1.5*ScaleToPixel);
    cv::line(Field,cv::Point(CUL.x+22*ScaleToPixel,(CUL.y+CDL.y)/2),cv::Point(CUL.x+22*ScaleToPixel,(CUL.y+CDL.y)/2),cv::Scalar(255,255,255),1.5*ScaleToPixel);
    cv::circle(Field,cv::Point(CLineU.x,(CLineD.y+CLineU.y)/2),7.5*ScaleToPixel,cv::Scalar(255,255,255),0.5*ScaleToPixel);
    cv::circle(Field,cv::Point(CLineU.x,(CLineD.y+CLineU.y)/2),0.5*ScaleToPixel,cv::Scalar(255,255,255),0.5*ScaleToPixel);
    
}

void determinePositions(cv::Point3f Pose){
    
    Observations[0].startPoint.x = sign(Pose.x) * (xMax);
    Observations[0].startPoint.y = sign(Pose.y) * (yMax);
    Observations[0].EndPoint.x =   sign(Pose.x) * (xMax);
    Observations[0].EndPoint.y =   0;
    
    Observations[1].startPoint.x = sign(Pose.x) * (xMax);
    Observations[1].startPoint.y = sign(Pose.y) * (yMax);
    Observations[1].EndPoint.x =   0;
    Observations[1].EndPoint.y =   sign(Pose.y) * (yMax);
    
    Observations[2].startPoint.x = sign(Pose.x) * (xLineBoundary);
    Observations[2].startPoint.y = sign(Pose.y) * (yLineBoundary);
    Observations[2].EndPoint.x =   sign(Pose.x) * (xLineBoundary);
    Observations[2].EndPoint.y =   0;
    
    Observations[3].startPoint.x = sign(Pose.x) * xLineBoundary;
    Observations[3].startPoint.y = sign(Pose.y) * yLineBoundary;
    Observations[3].EndPoint.x =   0;
    Observations[3].EndPoint.y =   sign(Pose.y) * yLineBoundary;
    
    Observations[4].startPoint.x = sign(Pose.x) * 3.5;
    Observations[4].startPoint.y = sign(Pose.y) * 2.5;
    Observations[4].EndPoint.x =   sign(Pose.x) * 3.5;
    Observations[4].EndPoint.y =   0;
    
    Observations[5].startPoint.x = 0;
    Observations[5].startPoint.y = sign(Pose.y) * yLineBoundary;
    Observations[5].EndPoint.x =   0;
    Observations[5].EndPoint.y =   0;
    
    Observations[6].startPoint.x = sign(Pose.x) * 3.5;
    Observations[6].startPoint.y = sign(Pose.y) * 2.5;
    
    Observations[7].startPoint.x = sign(Pose.x) * xLineBoundary;
    Observations[7].startPoint.y = sign(Pose.y) * yLineBoundary;
    
    Observations[8].startPoint.x = sign(Pose.x) * xLineBoundary;
    Observations[8].startPoint.y = sign(Pose.y) * 2.5;
    
    Observations[9].startPoint.x = 0;
    Observations[9].startPoint.y = sign(Pose.y) * yLineBoundary;
    
    Observations[10].startPoint.x = sign(Pose.x) * xLineBoundary;
    Observations[10].startPoint.y = sign(Pose.y) * 1.3;
    
    Observations[11].startPoint.x = sign(Pose.x) * xLineBoundary;
    Observations[11].startPoint.y = -sign(Pose.y) * 1.3;

    UselessLine.startPoint.x = sign(Pose.x) * 3.5;
    UselessLine.startPoint.y = sign(Pose.y) * 2.5;
    UselessLine.EndPoint.x =   sign(Pose.x) * 4.5;
    UselessLine.EndPoint.y =   sign(Pose.y) * 2.5;

}

int TheBestAction(int currentPositionNumber, cv::Point3f Pose, double& ballX, double& ballY, double& ball_confidence, vector<BallModel>& OwnBallModels){
    cv::Point3f RobotPixels;
    if (MyPlotConfig){
        Field.copyTo(temp);
        RobotPixels.x=(FieldWidth/2+FieldPading+(Pose.x*10))*ScaleToPixel;
        RobotPixels.y=(FieldHeight/2+FieldPading-(Pose.y*10))*ScaleToPixel;
        RobotPixels.z=Pose.z;
        
        cv::circle(temp,cv::Point(RobotPixels.x,RobotPixels.y),13,cv::Scalar(0,0,255),0.2*5);
        double r= 25;
        double x= r*cos(RobotPixels.z);
        double y= r*sin(RobotPixels.z);
        cv::line(temp,cv::Point(RobotPixels.x+x,RobotPixels.y-y),cv::Point(RobotPixels.x,RobotPixels.y),cv::Scalar(0,0,255),0.5*5);
    }
    
    for(int h=0;h<ActionsNumber;h++)
        P[h]=0;
    
    ActiveVision::Position currentPosition;
    currentPosition.Yaw = Actions[currentPositionNumber].Yaw;
    currentPosition.Pitch = Actions[currentPositionNumber].Pitch;
    
    float MaximumGain = 0;
    int TheBestActionNumber = 0;
    int IsVisible;
    int CurrentNumber;
    ActiveVision::Point *BoundPoints[4];
    float GlobalPose[3],Prelative[3];
    
    for(int i=0;i<ActionsNumber;i++){

        if (MyPlotConfig){
            temp.copyTo(temp1);
        }
        for(int w=0;w<4;w++){
            Prelative[0]=Actions[i].BounPointz[w].x;
            Prelative[1]=Actions[i].BounPointz[w].y;
            Prelative[2]=0;
            pose_global(Prelative,Pose,GlobalPose);
            BoundPoints[w] = new ActiveVision::Point;
            BoundPoints[w]->x = GlobalPose[0];
            BoundPoints[w]->y = GlobalPose[1];
        }
        
        //Draw The Bonding Polygon(lines and points)

        if (MyPlotConfig){
            int Bc;
            ActiveVision::Point F[4];
            for(Bc=0;Bc<4;Bc++){
                F[Bc].x = (FieldWidth/2+FieldPading+( BoundPoints[Bc]->x *10))*ScaleToPixel;
                F[Bc].y = (FieldHeight/2+FieldPading-( BoundPoints[Bc]->y *10))*ScaleToPixel;
            }
            for(Bc=0;Bc<4;Bc++){
                cv::circle(temp1, cv::Point(F[Bc].x,F[Bc].y), 5, cv::Scalar(255,255,255), -1, 4, 0);
                cv::line(temp1, cv::Point(F[Bc].x,F[Bc].y), cv::Point(F[(Bc+1)%4].x,F[(Bc+1)%4].y), cv::Scalar(255,255,255), 1, 4, 0);
            }

            // cv::imshow("ThePolygon",temp1);
            // cv::waitKey(500);
        }

        //Draw The Bonding Polygon(lines and points)

        arma::mat ballxy(3,1);
        ballxy(0,0) = ballX;
        ballxy(1,0) = ballY;
        ballxy(2,0) = 0;

        arma::mat uPose(3,1);
        uPose(0,0) = Pose.x;
        uPose(1,0) = Pose.y;
        uPose(2,0) = Pose.z;

        arma::mat ballGlobal=pose_global(ballxy,uPose);
        
        ActiveVision::Point Gball;
        Gball.x = ballGlobal[0];
        Gball.y = ballGlobal[1];
        
        // if (ball_confidence == 2){
        //     if (IsPointInPolygon(4, BoundPoints, Gball) == 2) {
        //         EnDiffBall[i] = 0.3 * (1 - OwnBallModels[0].p);
        //     }else{
        //         EnDiffBall[i] = -0.05 * (OwnBallModels[0].p);
        //     }
        // }else{
        //     EnDiffBall[i] = 0;
        // }

        
        if (IsPointInPolygon(4, BoundPoints, Gball) == 2) {
            EnDiffBall[i] = 0.3;
        }else{
            EnDiffBall[i] = -0.05;
        }

        for(int j=0;j<ObservationsNumber;j++){
            if (MyPlotConfig){
                temp1.copyTo(Input);
            }
            IsVisible = Visible(Actions[i],Observations[j],Pose,BoundPoints, true);
            // cout<<" action "<<i<<" observation "<<j<<" : "<<IsVisible<<endl;
            CurrentNumber = i * ObservationsNumber + j;
            VisibilityStatus[CurrentNumber] = IsVisible;
            P[i] = P[i] + IsVisible;
        }
        
        if (P[i]>=MaximumGain){
            MaximumGain = P[i];
            TheBestActionNumber = i;
        }
        // cout<<i<<" Value :"<<P[i]<<endl;
    }
    
    // cout<<"Number: "<<TheBestActionNumber<<endl;
    // cout<<Actions[TheBestActionNumber].Yaw<<" "<<Actions[TheBestActionNumber].Pitch<<endl;
    
    // string shmSegmentName = string("wcmRobot")+to_string(*teamNumber)+to_string(*robotId)+string("root");
    // static shm shmSegment(&shmSegmentName[0]);
    
    // shmSegment.setVector("VisibilityStatus", &VisibilityStatus[0],480);
    
    // shmSegment.setVector("LineFeatures", &LineFeatures[0],960);
    // for (int o =0;o<480;o++){
    //     if (VisibilityStatus[o] != 0)
    //         cout<<"C++: "<<o<<" "<<VisibilityStatus[o]<<"\t";
    // }
    // cout<<endl<<"****************************"<<endl;
    
    // for (int o =0;o<960;o++){
    //     cout<<"C++: "<<o<<" "<<LineFeatures[o]<<endl;
    // }
    // cout<<"************************************"<<endl;

    // int TestNum;
    // for (int i=0;i<ActionsNumber;i++){
    //     cout<<i<<" :"<<endl;
    //     for (int j=0;j<ObservationsNumber;j++){
    //         TestNum = i*ObservationsNumber + j;
    //         cout<<VisibilityStatus[TestNum]<<"  ";
    //     }
    //     cout<<endl;
    // }

    // for (int L=0;L<ActionsNumber;L++)
    //   Detecteds(L,Pose);
    
    return TheBestActionNumber;
}

int Visible(ActiveVision::Position action,ActiveVision::Observation observation,cv::Point3f Pose,ActiveVision::Point **BoundPoints, bool AtDedication){
    int FeatureCounter;
    float GlobalPose[3];
    float TEMP1,TEMP2;
    ActiveVision::Point P1,P2;
    float In1,In2,Po1,Po2;

    if (observation.Type == "Line"){
        ActiveVision::Line BoundLines[4];
        ActiveVision::Point s;
        int onBoundCounter=0,u=0;
        ActiveVision::Point intersections[2];
        ActiveVision::Line observedLine=Equa(observation.startPoint,observation.EndPoint);
        ActiveVision::Point F1,F2;

        if (MyPlotConfig){
            Input.copyTo(temp2);
        }
        
        //Draw the Point we expect to exist
        if (MyPlotConfig){
            F1.x = (FieldWidth/2+FieldPading+( observation.startPoint.x *10))*ScaleToPixel;
            F1.y = (FieldHeight/2+FieldPading-( observation.startPoint.y *10))*ScaleToPixel;
            F2.x = (FieldWidth/2+FieldPading+( observation.EndPoint.x *10))*ScaleToPixel;
            F2.y = (FieldHeight/2+FieldPading-( observation.EndPoint.y *10))*ScaleToPixel;
            cv::line(temp2, cv::Point(F1.x,F1.y), cv::Point(F2.x,F2.y), cv::Scalar(255,0,0), 5, 4, 0);
        }
        // Draw the Point we expect to exist
        

        // Draw the line coming from the Observation Points
        if (MyPlotConfig){
            if (observedLine.IsVertical>0) {
                F1.x = (FieldWidth/2+FieldPading+( observedLine.B *10))*ScaleToPixel;
                F1.y = (FieldHeight/2+FieldPading-( -20 *10))*ScaleToPixel;
                
                F2.x = (FieldWidth/2+FieldPading+( observedLine.B *10))*ScaleToPixel;
                F2.y = (FieldHeight/2+FieldPading-( 20 *10))*ScaleToPixel;
            }
            else{
                F1.x = (FieldWidth/2+FieldPading+( -20 *10))*ScaleToPixel;
                F1.y = (FieldHeight/2+FieldPading-( observedLine.A * (-20) + observedLine.B *10))*ScaleToPixel;
                
                F2.x = (FieldWidth/2+FieldPading+( 20 *10))*ScaleToPixel;
                F2.y = (FieldHeight/2+FieldPading-( observedLine.A * (20) + observedLine.B *10))*ScaleToPixel;
            }
            cv::line(temp2, cv::Point(F1.x,F1.y), cv::Point(F2.x,F2.y), cv::Scalar(0,255,255), 1, 4, 0);
        }
        // Draw the line coming from the Observation Points
        

        int Intersect;
        for(int i=0;i<4;i++){
            BoundLines[i]=Equa(*BoundPoints[i], *BoundPoints[(i+1)%4]);
            
            //Draw the Bounding lines
            if (MyPlotConfig){
                F1.x = (FieldWidth/2+FieldPading+( -20 *10))*ScaleToPixel;
                F1.y = (FieldHeight/2+FieldPading-( (BoundLines[i].A * (-20) + BoundLines[i].B) *10))*ScaleToPixel;
                F2.x = (FieldWidth/2+FieldPading+( 20 *10))*ScaleToPixel;
                F2.y = (FieldHeight/2+FieldPading-( (BoundLines[i].A * (20) + BoundLines[i].B) *10))*ScaleToPixel;
                cv::line(temp2, cv::Point(F1.x,F1.y), cv::Point(F2.x,F2.y), cv::Scalar(255,255,255), 1, 4, 0);
            }
            //Draw the Bounding lines
            
            // cout<<" observedLine: "<<observedLine.A<<"  "<<observedLine.B<<endl;
            // cout<<" BoundLineI: "<<BoundLines[i].A<<"  "<<BoundLines[i].B<<endl;
            Intersect = Intersection(observedLine, BoundLines[i], &s);
            // cout<<" Intersection: "<<s.x<<"  "<<s.y<<endl;
            // cout<<"######################################################################"<<endl;
            if (Intersect<2){
                return 0;
            }
            else if (Intersect>1){
                
                //Draw the intersections
                if (MyPlotConfig){
                    F1.x = (FieldWidth/2+FieldPading+( s.x *10))*ScaleToPixel;
                    F1.y = (FieldHeight/2+FieldPading-( s.y *10))*ScaleToPixel;
                    cv::circle(temp2, cv::Point(F1.x,F1.y), 5, cv::Scalar(0,0,0), -1, 4, 0);
                    
                    // cv::imshow("Lines and Intersections",temp2);
                    // cv::waitKey(20);
                }
                //Draw the intersections
                
                // cout<<BoundPoints[0]->x<<" "<<BoundPoints[0]->y<<endl;
                // cout<<BoundPoints[1]->x<<" "<<BoundPoints[1]->y<<endl;
                // cout<<BoundPoints[2]->x<<" "<<BoundPoints[2]->y<<endl;
                // cout<<BoundPoints[3]->x<<" "<<BoundPoints[3]->y<<endl;
                
                // cout<<"##############################################"<<endl;
                
                // cout<<" counter: "<<IsPointInPolygon(4,BoundPoints,s)<<endl;
                if (IsPointInPolygon(4, BoundPoints, s)==1){
                    intersections[u].x=s.x;
                    intersections[u].y=s.y;
                    onBoundCounter++;
                    u++;
                }
                // cout<<"####################"<<endl;
            }
        }
        if(onBoundCounter==0){
            return 0;
        }
        else{
            // cout<<" dddd: "<<onBoundCounter<<endl;
            ActiveVision::Point PointA,PointB;
            int xS=IsPointInPolygon(4,BoundPoints,observation.startPoint);
            int xE=IsPointInPolygon(4,BoundPoints,observation.EndPoint);
            if (xS ==0 && xE==0){
                if(observedLine.IsVertical >0){
                    TEMP1 = observation.startPoint.y - intersections[0].y;
                    TEMP2 = observation.EndPoint.y - intersections[0].y;
                }
                else{
                    TEMP1 = observation.startPoint.x - intersections[0].x ;
                    TEMP2 = observation.EndPoint.x - intersections[0].x ;
                }
                if (TEMP1 * TEMP2>0)
                    return 0;
                PointA.x = intersections[0].x;
                PointA.y = intersections[0].y;
                PointB.x = intersections[1].x;
                PointB.y = intersections[1].y;
            }
            else if (xS==0 || xE==0){
                //Determine Which is in
                if(xS>0){
                    P1 = observation.startPoint;
                    P2 = observation.EndPoint;
                }
                else{
                    P1 = observation.EndPoint;
                    P2 = observation.startPoint;
                }
                //Determine Which is in
                PointA = P1;
                if(observedLine.IsVertical >0){
                    Po1 = P1.y;
                    Po2 = P2.y;
                    In1 = intersections[1].y;
                    In2 = intersections[0].y;
                }
                else{
                    Po1 = P1.x;
                    Po2 = P2.x;
                    In1 = intersections[1].x;
                    In2 = intersections[0].x;
                }
                if (Po1 >=Po2){
                    if (In1 >= In2)
                        PointB = intersections[0];
                    else
                        PointB = intersections[1];
                }
                else{
                    if (In1 >= In2)
                        PointB = intersections[1];
                    else
                        PointB = intersections[0];
                }
            }
            else{
                PointA.x = observation.startPoint.x;
                PointA.y = observation.startPoint.y;
                PointB.x = observation.EndPoint.x;
                PointB.y = observation.EndPoint.y;
            }

            if (AtDedication){
                FeatureCounter = (action.number * 4 * TotalLinesOfField) + (observation.number * 4) ;

                // cout<<PointA.x<<" "<<PointA.x<<" "<<PointB.x<<" "<<PointB.x<<" "<<endl;
                // GlobalPose[0] = PointA.x;
                // GlobalPose[1] = PointA.y;
                // pose_relative(GlobalPose,Pose,Prelative);
                LineFeatures[FeatureCounter] = PointA.x;
                LineFeatures[FeatureCounter + 1] = PointA.y;

                // GlobalPose[0] = PointB.x;
                // GlobalPose[1] = PointB.y;
                // pose_relative(GlobalPose,Pose,Prelative);
                LineFeatures[FeatureCounter + 2] = PointB.x;
                LineFeatures[FeatureCounter + 3] = PointB.y;
            }
            
            // Draw then final candidate to see whether is longer than 1 or not

            if (MyPlotConfig){
                F1.x = (FieldWidth/2+FieldPading+( PointA.x *10))*ScaleToPixel;
                F1.y = (FieldHeight/2+FieldPading-( PointA.y *10))*ScaleToPixel;
                F2.x = (FieldWidth/2+FieldPading+( PointB.x *10))*ScaleToPixel;
                F2.y = (FieldHeight/2+FieldPading-( PointB.y *10))*ScaleToPixel;
                cv::circle(temp2, cv::Point(F1.x,F1.y), 5, cv::Scalar(0,0,255), -1, 4, 0);
                cv::circle(temp2, cv::Point(F2.x,F2.y), 5, cv::Scalar(0,0,255), -1, 4, 0);
                cv::line(temp2, cv::Point(F1.x,F1.y), cv::Point(F2.x,F2.y), cv::Scalar(0,0,255), 1, 4, 0);
                
                // cv::imshow("The Final Line Candidate",temp2);
                // cv::waitKey(500);
            }
            // Draw then final candidate to see whether is longer than 1 or not
            
            // cout<<"WWWWWWWW"<<F1.x<<" "<<F1.y<<" "<<F2.x<<" "<<F2.y<<endl;
            
            arma::mat A(3, 1, arma::fill::zeros);
            arma::mat B(3, 1, arma::fill::zeros);
            A(0, 0) = PointA.x;
            A(1, 0) = PointA.y;
            B(0, 0) = PointB.x;
            B(1, 0) = PointB.y;
            arma::mat uPose(3,1);
            uPose(0,0) = Pose.x;
            uPose(1,0) = Pose.y;
            uPose(2,0) = Pose.z;
            A = pose_relative(A, uPose);
            B = pose_relative(B, uPose);
            double dist, x, y;
            robot_to_line_distance(A, B, dist, x, y);
            if (dist > 4.5)
                return 0;
            else
                return validateLine(PointA, PointB);
        }
    }
    else{
        if (IsPointInPolygon(4,BoundPoints,observation.startPoint)>1 && sqrt (pow(Pose.x-observation.startPoint.x,2) + pow(Pose.y-observation.startPoint.y,2))< 4.5)
            return 1;
        else
            return 0;
    }
}

ActiveVision::Line Equa(ActiveVision::Point k,ActiveVision::Point l){
    ActiveVision::Line z;
    z.IsVertical = false;
    if (k.x == l.x){
        z.IsVertical = true;
        z.A = 0;
        z.B = k.x;
    }
    else{
        z.A = (k.y-l.y)/(k.x-l.x);
        z.B = k.y-z.A*k.x;
    }
    return z;
}

int Intersection(ActiveVision::Line M,ActiveVision::Line N,ActiveVision::Point* Q){
    
    if (M.IsVertical == true && N.IsVertical ==true){
        if (M.B == N.B)
            return 1;
        else
            return 0;
    }
    else if(M.IsVertical == true || N.IsVertical ==true){
        if (M.IsVertical == true){
            Q->x = M.B;
            Q->y = N.A * Q->x + N.B;
        }
        else{
            Q->x = N.B;
            Q->y = M.A * Q->x + M.B;
        }
        return 2;
    }
    else{
        if (M.A == N.A){
            if (M.B==N.B)
                return 1;
            else
                return 0;
        }
        else{
            Q->x = -(M.B-N.B)/(M.A-N.A);
            Q->y = M.A*Q->x+M.B;
            return 2;
        }
    }
}

int IsPointInPolygon(int nvert,ActiveVision::Point **vert,ActiveVision::Point test){
    
    // cout<<"***********************************************"<<endl;
    
    // cout<<vert[0]->x<<" "<<vert[0]->y<<endl;
    // cout<<vert[1]->x<<" "<<vert[1]->y<<endl;
    // cout<<vert[2]->x<<" "<<vert[2]->y<<endl;
    // cout<<vert[3]->x<<" "<<vert[3]->y<<endl;
    
    // cout<<test.x<<" "<<test.y<<endl;
    
    
    float a,b,d=0,e,x;
    int c=2;
    a = atan2(vert[0]->y-test.y,vert[0]->x-test.x);
    b = atan2(vert[1]->y-test.y,vert[1]->x-test.x);
    d = sign(mod_angle(b-a));
    e=a;
    for(int i=1;i<nvert+1;i++){
        b = atan2(vert[i%4]->y-test.y,vert[i%4]->x-test.x);
        x= mod_angle(b-a);
        // std::cout<<b*180/PI<<" "<<a*180/PI<<" "<<x*180/PI<<" "<<d<<std::endl;
        if (absolute(absolute(x)-M_PI)<0.0001){
            // cout<<"A"<<endl;
            c=1;
            break;
        }
        else if (x*d<=0){
            // cout<<"B"<<endl;
            c=0;
            break;
        }
        else{
            // cout<<"C"<<endl;
            a=b;
            d= sign(x);
        }
    }
    a=e;
    d=0;
    if (c>1){
        for(int i=nvert-1;i>0;i--){
            b = atan2(vert[i]->y-test.y,vert[i]->x-test.x);
            float x= mod_angle(b-a);
            // std::cout<<b*180/PI<<" "<<a*180/PI<<" "<<x*180/PI<<d<<std::endl;
            if (x*d<0){
                c=0;
                break;
            }
            else{
                a=b;
                d= sign(x);
            }
        }
    }
    
    // cout<<c<<endl;
    // cout<<"***********************************************"<<endl;
    return c;
}

void pose_global(float* pRelative,cv::Point3f Pose,float* GlobalPose){
    float ca = cos(Pose.z);
    float sa = sin(Pose.z);
    GlobalPose[0] = Pose.x + ca*pRelative[0] - sa*pRelative[1];
    GlobalPose[1] = Pose.y + sa*pRelative[0] + ca*pRelative[1];
    GlobalPose[2] = Pose.z + pRelative[2];
}

void pose_relative(float* pGlobal,cv::Point3f Pose,float* RelativePose){
    float ca = cos(Pose.z);
    float sa = sin(Pose.z);
    
    float px = pGlobal[0]-Pose.x;
    float py = pGlobal[1]-Pose.y;
    float pa = pGlobal[2]-Pose.z;
    
    RelativePose[0] = ca*px + sa*py;
    RelativePose[1] = -sa*px + ca*py;
    RelativePose[2] = mod_angle(pa);
}

float absolute(float x){
    return x * sign(x);
}

int validateLine(ActiveVision::Point A,ActiveVision::Point B){
    if (sqrt(pow(A.x-B.x,2)+pow(A.y-B.y,2))>=1)
        return 1;
    else
        return 0;
}

int sign(float x){
    if (x > 0)
	    return 1;
    else if (x < 0)
	    return -1;
    else
	    return 0;
}

void Detecteds(int ActionNumber,cv::Point3f Pose,string Name, double& ballX, double& ballY, double& ball_confidence){
    // cout<<"aaaa"<<endl;
    int temp,DetectedCounter=0;
    ActiveVision::Point StartPoint,EndPoint;
    int FeatureCounter;
    ActiveVision::Point BoundPoints[4];
    cv::Mat ShowMat;
    
    Field.copyTo(ShowMat);
    
    // Draw robot
    if (DrawRobot){
        cv::Point3f RobotPixels;
        RobotPixels.x=(FieldWidth/2+FieldPading+(Pose.x*10))*ScaleToPixel;
        RobotPixels.y=(FieldHeight/2+FieldPading-(Pose.y*10))*ScaleToPixel;
        RobotPixels.z=Pose.z;

        cv::circle(ShowMat,cv::Point(RobotPixels.x,RobotPixels.y),13,cv::Scalar(0,0,255),0.2*5);
        double r= 25;
        double x= r*cos(RobotPixels.z);
        double y= r*sin(RobotPixels.z);
        cv::line(ShowMat,cv::Point(RobotPixels.x+x,RobotPixels.y-y),cv::Point(RobotPixels.x,RobotPixels.y),cv::Scalar(0,0,255),0.5*5);
    }
    // Draw robot
    
    // Draw ball
    if (DrawBall){
        arma::mat ballxy(3,1);
        ballxy(0,0) = ballX;
        ballxy(1,0) = ballY;
        // cout<<"ballll"<<wcmBall.get<double>("GTX")[0]<<"    "<<wcmBall.get<double>("GTY")[0]<<endl;
        ballxy(2,0) = 0;
        arma::mat uPose(3,1);
        uPose(0,0) = Pose.x;
        uPose(1,0) = Pose.y;
        uPose(2,0) = Pose.z;
        arma::mat ballGlobal=pose_global(ballxy, uPose);
        // ballGlobal[0] = 0;//wcmBall.get<double>("GTX")[0];
        // ballGlobal[1] = 0;//wcmBall.get<double>("GTY")[0];
        ActiveVision::Point Gball;
        Gball.x = (FieldWidth/2+FieldPading+(ballGlobal[0]*10))*ScaleToPixel;
        Gball.y = (FieldHeight/2+FieldPading-(ballGlobal[1]*10))*ScaleToPixel;
        if (ball_confidence == 2)
            cv::circle(ShowMat, cv::Point(Gball.x,Gball.y), 7, cv::Scalar(0,0,0), -1);
        else
            cv::circle(ShowMat, cv::Point(Gball.x,Gball.y), 7, cv::Scalar(0,0,0), 0.2*5);
    }
    // Draw ball

    // Draw supposed field of view
    if (DrawSupposedFOV){
        FieldOfView(ActionNumber,Pose,BoundPoints);
        for (int w=0;w<4;w++){
            BoundPoints[w].x = (FieldWidth/2+FieldPading+( BoundPoints[w].x *10))*ScaleToPixel;
            BoundPoints[w].y = (FieldHeight/2+FieldPading-( BoundPoints[w].y *10))*ScaleToPixel;
        }
        for(int counter=0;counter<4;counter++){
            circle(ShowMat, cv::Point(BoundPoints[counter].x,BoundPoints[counter].y), 5, cv::Scalar(0,0,255), -1, 4, 0);
            line(ShowMat, cv::Point(BoundPoints[counter].x,BoundPoints[counter].y), cv::Point(BoundPoints[(counter+1)%4].x,BoundPoints[(counter+1)%4].y), cv::Scalar(0,0,255), 1, 4, 0);
        }
    }
    // Draw supposed field of view

    // Real Points online

    // Real Points online SHM
    if (DrawPointsOnlineSHM){
        ActiveVision::Point RealBoundPoints[4];
        float GlobalPose[3],Prelative[3];

        RealBoundPoints[0].x = 0;//wcmRobot.get<double>("UpperLeftX")[0];
        RealBoundPoints[0].y = 0;//wcmRobot.get<double>("UpperLeftY")[0];
        RealBoundPoints[1].x = 0;//wcmRobot.get<double>("UpperRightX")[0];
        RealBoundPoints[1].y = 0;//wcmRobot.get<double>("UpperRightY")[0];
        RealBoundPoints[2].x = 0;//wcmRobot.get<double>("LowerRightX")[0];
        RealBoundPoints[2].y = 0;//wcmRobot.get<double>("LowerRightY")[0];
        RealBoundPoints[3].x = 0;//wcmRobot.get<double>("LowerLeftX")[0];
        RealBoundPoints[3].y = 0;//wcmRobot.get<double>("LowerLeftY")[0];
        cout<<RealBoundPoints[2].x<<"    "<<RealBoundPoints[2].y<<"   "<<RealBoundPoints[3].x<<"   "<<RealBoundPoints[3].y<<endl;

        for(int w=0;w<4;w++){
	        Prelative[0]=RealBoundPoints[w].x;
	        Prelative[1]=RealBoundPoints[w].y;
	        // cout<<Prelative[0]<<" "<<Prelative[1]<<endl;
	        Prelative[2]=0;
	        pose_global(Prelative,Pose,GlobalPose);
	        RealBoundPoints[w].x = GlobalPose[0];
	        RealBoundPoints[w].y = GlobalPose[1];
        }

        for (int w=0;w<4;w++){
            RealBoundPoints[w].x = (FieldWidth/2+FieldPading+( RealBoundPoints[w].x *10))*ScaleToPixel;
            RealBoundPoints[w].y = (FieldHeight/2+FieldPading-( RealBoundPoints[w].y *10))*ScaleToPixel;
        }
        for(int counter=0;counter<4;counter++){
            circle(ShowMat, cv::Point(RealBoundPoints[counter].x,RealBoundPoints[counter].y), 5, cv::Scalar(0,0,255), -1, 4, 0);
            line(ShowMat, cv::Point(RealBoundPoints[counter].x,RealBoundPoints[counter].y), cv::Point(RealBoundPoints[(counter+1)%4].x,RealBoundPoints[(counter+1)%4].y), cv::Scalar(0,0,255), 1, 4, 0);
        }
    }
    // Real Points online SHM

    // Cpp from just here
    if (DrawPointsOnlineCpp){
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


        ActiveVision::Point *BoundPoints[4];
        for(int w=0;w<4;w++){
            Prelative[0]=RealBoundPoints[w].x;
            Prelative[1]=RealBoundPoints[w].y;
            Prelative[2]=0;
            pose_global(Prelative,Pose,GlobalPose);
            BoundPoints[w] = new ActiveVision::Point;
            BoundPoints[w]->x = GlobalPose[0];
            BoundPoints[w]->y = GlobalPose[1];
        }

        // Determine if true global ball is in the online field of view
        arma::mat ballxy(3,1);
        ballxy(0,0) = ballX;
        ballxy(1,0) = ballY;
        ballxy(2,0) = 0;

        arma::mat uPose(3,1);
        uPose(0,0) = Pose.x;
        uPose(1,0) = Pose.y;
        uPose(2,0) = Pose.z;

        arma::mat ballGlobal = pose_global(ballxy, uPose);
        
        ActiveVision::Point Gball;
        Gball.x = ballGlobal[0];
        Gball.y = ballGlobal[1];

        BallIsInOnlineFOV = 0;
        if (IsPointInPolygon(4, BoundPoints, Gball) == 2){
            BallIsInOnlineFOV = 1;
        }

        // cout<<BallIsInOnlineFOV<<endl;
        // wcmRobot.set("BallIsInOnlineFOV", BallIsInOnlineFOV);
        // Determine if true global ball is in the online field of view

        CurrentHeadPositionVisibilityCode = 0;
        for (int i=0; i<ObservationsNumber; i++){
            if (Visible(Actions[0], Observations[i], Pose, BoundPoints, false) == 1){
                // std::cout<<i<<"\t";
                CurrentHeadPositionVisibilityCode = CurrentHeadPositionVisibilityCode + pow(2, i);
            }
        }

        // cout<<BestActionVisibilityCode<<"    "<<CurrentHeadPositionVisibilityCode<<endl;
        CurrentHeadPositionVisibilityCode = (BestActionVisibilityCode&CurrentHeadPositionVisibilityCode);
        // if (hamming_distance(BestActionVisibilityCode, CurrentHeadPositionVisibilityCode) == 0){
        //     wcmRobot.set("Success", 1);
        // }else{
        //     wcmRobot.set("Success", 0);
        // }

        for(int w=0;w<4;w++){
	        Prelative[0]=RealBoundPoints[w].x;
	        Prelative[1]=RealBoundPoints[w].y;
	        // cout<<Prelative[0]<<" "<<Prelative[1]<<endl;
	        Prelative[2]=0;
	        pose_global(Prelative,Pose,GlobalPose);
	        RealBoundPoints[w].x = GlobalPose[0];
	        RealBoundPoints[w].y = GlobalPose[1];
        }

        for (int w=0;w<4;w++){
            RealBoundPoints[w].x = (FieldWidth/2+FieldPading+( RealBoundPoints[w].x *10))*ScaleToPixel;
            RealBoundPoints[w].y = (FieldHeight/2+FieldPading-( RealBoundPoints[w].y *10))*ScaleToPixel;
        }

        for(int counter=0;counter<4;counter++){
            circle(ShowMat, cv::Point(RealBoundPoints[counter].x, RealBoundPoints[counter].y), 5, cv::Scalar(255, 0, 0), -1, 4, 0);
            line(ShowMat, cv::Point(RealBoundPoints[counter].x, RealBoundPoints[counter].y), cv::Point(RealBoundPoints[(counter+1)%4].x, RealBoundPoints[(counter+1)%4].y), cv::Scalar(255, 0, 0), 1, 4, 0);
        }
        
    }
    // Cpp from just here
    // Real Points online
    
    // Show the supposed visible observations
    if (ActiveVisionBoundaryConfig){
        for (int i=0; i<NumOfFieldBoundaryLines; i++){
            temp = (ActionNumber * ObservationsNumber) + i;
            if (VisibilityStatus[temp]==1){
                DetectedCounter++;
                FeatureCounter = (ActionNumber* 4 * TotalLinesOfField) + (i * 4) ;
                // cout<<" AAAA "<<FeatureCounter<<endl;
                // cout<<ActionNumber<<" "<<i<<" "<<LineFeatures[FeatureCounter]<<" "<<LineFeatures[FeatureCounter+1]<<" "<<LineFeatures[FeatureCounter+2]<<" "<<LineFeatures[FeatureCounter+3]<<endl;
                StartPoint.x = (FieldWidth/2+FieldPading+( LineFeatures[FeatureCounter] *10))*ScaleToPixel;
                StartPoint.y = (FieldHeight/2+FieldPading-( LineFeatures[FeatureCounter+1] *10))*ScaleToPixel;
                EndPoint.x =   (FieldWidth/2+FieldPading+( LineFeatures[FeatureCounter+2] *10))*ScaleToPixel;
                EndPoint.y =   (FieldHeight/2+FieldPading-( LineFeatures[FeatureCounter+3] *10))*ScaleToPixel;
                line(ShowMat, cv::Point(StartPoint.x,StartPoint.y), cv::Point(EndPoint.x,EndPoint.y), cv::Scalar(255,0,0),5,4,0);
            }
        }
    }

    if (ActiveVisionLineConfig){
        for (int i=NumOfFieldBoundaryLines; i<TotalLinesOfField; i++){
            temp = (ActionNumber * ObservationsNumber) + i;
            if (VisibilityStatus[temp]==1){
                DetectedCounter++;
                FeatureCounter = (ActionNumber* 4 * TotalLinesOfField) + (i * 4) ;
                // cout<<" AAAA "<<FeatureCounter<<endl;
                // cout<<ActionNumber<<" "<<i<<" "<<LineFeatures[FeatureCounter]<<" "<<LineFeatures[FeatureCounter+1]<<" "<<LineFeatures[FeatureCounter+2]<<" "<<LineFeatures[FeatureCounter+3]<<endl;
                StartPoint.x = (FieldWidth/2+FieldPading+( LineFeatures[FeatureCounter] *10))*ScaleToPixel;
                StartPoint.y = (FieldHeight/2+FieldPading-( LineFeatures[FeatureCounter+1] *10))*ScaleToPixel;
                EndPoint.x =   (FieldWidth/2+FieldPading+( LineFeatures[FeatureCounter+2] *10))*ScaleToPixel;
                EndPoint.y =   (FieldHeight/2+FieldPading-( LineFeatures[FeatureCounter+3] *10))*ScaleToPixel;
                line(ShowMat, cv::Point(StartPoint.x,StartPoint.y), cv::Point(EndPoint.x,EndPoint.y), cv::Scalar(255,0,0),5,4,0);
            }
        }
    }

    if (ActiveVisionLCornerConfig){
        for (int j=TotalLinesOfField; j<TotalLinesOfField+NumOfLCorners; j++){
            temp = (ActionNumber * ObservationsNumber) + j;
            if (VisibilityStatus[temp]==1){
                DetectedCounter++;
                StartPoint.x = (FieldWidth/2+FieldPading+( Observations[j].startPoint.x *10))*ScaleToPixel;
                StartPoint.y = (FieldHeight/2+FieldPading-( Observations[j].startPoint.y *10))*ScaleToPixel;
                cv::circle(ShowMat, cv::Point(StartPoint.x,StartPoint.y), 5, cv::Scalar(147,20,255), -1, 4, 0);
            }
        }
    }

    if (ActiveVisionTCornerConfig){
        for (int j=TotalLinesOfField+NumOfLCorners; j<ObservationsNumber; j++){
            temp = (ActionNumber * ObservationsNumber) + j;
            if (VisibilityStatus[temp]==1){
                DetectedCounter++;
                StartPoint.x = (FieldWidth/2+FieldPading+( Observations[j].startPoint.x *10))*ScaleToPixel;
                StartPoint.y = (FieldHeight/2+FieldPading-( Observations[j].startPoint.y *10))*ScaleToPixel;
                cv::circle(ShowMat, cv::Point(StartPoint.x,StartPoint.y), 5, cv::Scalar(147,20,255), -1, 4, 0);
            }
        }
    }
    // Show the supposed visible observations
    
    if (WriteActionInfo){
        putText(ShowMat, to_string(ActionNumber), cv::Point(20,80), 1, 5, cv::Scalar(0,0,0), 5 );
        putText(ShowMat, to_string(int(Actions[ActionNumber].Yaw)), cv::Point(20,160), 1, 5, cv::Scalar(0,0,0), 5);
        putText(ShowMat, to_string(int(Actions[ActionNumber].Pitch)), cv::Point(20,240), 1, 5, cv::Scalar(0,0,0), 5);
    }

    // if (DetectedCounter>0){
    // cout<<"bbbb"<<endl;
    // cv::namedWindow(Name);
    cv::imshow(Name, ShowMat);
    cv::waitKey(1);
    // }
}

void FieldOfView(int ActionNumber,cv::Point3f Pose,ActiveVision::Point *BoundPoints){
    // cout<<ActionNumber<<endl;
    // cout<<"YAW PITCH : "<<Actions[ActionNumber].Yaw<<" "<<Actions[ActionNumber].Pitch<<endl;
    float GlobalPose[3],Prelative[3];
    for(int w=0;w<4;w++){
	    Prelative[0]=Actions[ActionNumber].BounPointz[w].x;
	    Prelative[1]=Actions[ActionNumber].BounPointz[w].y;
	    // cout<<Prelative[0]<<" "<<Prelative[1]<<endl;
	    Prelative[2]=0;
	    pose_global(Prelative,Pose,GlobalPose);
	    BoundPoints[w].x = GlobalPose[0];
	    BoundPoints[w].y = GlobalPose[1];
    }
}

int hamming_distance(unsigned x, unsigned y){
    int dist = 0;
    for (unsigned val = x ^ y; val > 0; val = val >> 1){
        if (val & 1)
            dist++;
    }
    return dist;
}

void CheckConditions(int ActionNumber, double BallX, double BallY, double PoseX, double PoseY, double PoseA, double* Conditions, int ForCheck){
    
    int temp,DetectedCounter=0;
    int FeatureCounter;
    ActiveVision::Point StartPoint,EndPoint;
    // cv::Mat ShowMat;
    Field.copyTo(ShowMat);
    cv::Point3f Pose;
    Pose.x = PoseX;
    Pose.y = PoseY;
    Pose.z = PoseA;
    
    // Draw robot
    if (DrawRobot){
        cv::Point3f RobotPixels;
        RobotPixels.x=(FieldWidth/2+FieldPading+(PoseX*10))*ScaleToPixel;
        RobotPixels.y=(FieldHeight/2+FieldPading-(PoseY*10))*ScaleToPixel;
        RobotPixels.z=PoseA;

        cv::circle(ShowMat,cv::Point(RobotPixels.x,RobotPixels.y),13,cv::Scalar(0,0,255),0.2*5);
        double r= 25;
        double x= r*cos(RobotPixels.z);
        double y= r*sin(RobotPixels.z);
        cv::line(ShowMat,cv::Point(RobotPixels.x+x,RobotPixels.y-y),cv::Point(RobotPixels.x,RobotPixels.y),cv::Scalar(0,0,255),0.5*5);
    }
    // Draw robot
    
    // Draw ball
    if (DrawBall){
        arma::mat ballxy(3,1);
        ballxy(0,0) = BallX;
        ballxy(1,0) = BallY;
        // cout<<"ballll"<<wcmBall.get<double>("GTX")[0]<<"    "<<wcmBall.get<double>("GTY")[0]<<endl;
        ballxy(2,0) = 0;
        arma::mat uPose(3,1);
        uPose(0,0) = PoseX;
        uPose(1,0) = PoseY;
        uPose(2,0) = PoseA;
        arma::mat ballGlobal=pose_global(ballxy, uPose);
        // ballGlobal[0] = 0;//wcmBall.get<double>("GTX")[0];
        // ballGlobal[1] = 0;//wcmBall.get<double>("GTY")[0];
        ActiveVision::Point Gball;
        Gball.x = (FieldWidth/2+FieldPading+(ballGlobal[0]*10))*ScaleToPixel;
        Gball.y = (FieldHeight/2+FieldPading-(ballGlobal[1]*10))*ScaleToPixel;
        cv::circle(ShowMat, cv::Point(Gball.x,Gball.y), 7, cv::Scalar(0,0,0), 0.2*5);
    }
    // Draw ball

    // Draw supposed field of view
    if (DrawSupposedFOV){
        ActiveVision::Point BoundPoints[4];
        FieldOfView(ActionNumber, Pose, BoundPoints);
        for (int w=0;w<4;w++){
            BoundPoints[w].x = (FieldWidth/2+FieldPading+( BoundPoints[w].x *10))*ScaleToPixel;
            BoundPoints[w].y = (FieldHeight/2+FieldPading-( BoundPoints[w].y *10))*ScaleToPixel;
        }
        for(int counter=0;counter<4;counter++){
            circle(ShowMat, cv::Point(BoundPoints[counter].x,BoundPoints[counter].y), 5, cv::Scalar(0,0,255), -1, 4, 0);
            line(ShowMat, cv::Point(BoundPoints[counter].x,BoundPoints[counter].y), cv::Point(BoundPoints[(counter+1)%4].x,BoundPoints[(counter+1)%4].y), cv::Scalar(0,0,255), 1, 4, 0);
        }
    }
    // Draw supposed field of view

    // Form the FOV polygon
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
    // Form the FOV polygon

    // Determine if true global ball is in the online field of view
    arma::mat ballxy(3,1);
    ballxy(0,0) = BallX;
    ballxy(1,0) = BallY;
    ballxy(2,0) = 0;
    arma::mat uPose(3,1);
    uPose(0,0) = Pose.x;
    uPose(1,0) = Pose.y;
    uPose(2,0) = Pose.z;
    arma::mat ballGlobal = pose_global(ballxy, uPose);
    ActiveVision::Point Gball;
    Gball.x = ballGlobal[0];
    Gball.y = ballGlobal[1];
    
    ActiveVision::Point *BoundPoints[4];
    for(int w=0; w<4; w++){
        Prelative[0] = RealBoundPoints[w].x;
        Prelative[1] = RealBoundPoints[w].y;
        Prelative[2] = 0;
        pose_global(Prelative, Pose, GlobalPose);
        BoundPoints[w] = new ActiveVision::Point;
        BoundPoints[w]->x = GlobalPose[0];
        BoundPoints[w]->y = GlobalPose[1];
    }
    
    Conditions[0] = 0;
    if (IsPointInPolygon(4, BoundPoints, Gball) == 2){
        Conditions[0] = 1;
    }
    // cout<<Conditions[0]<<endl;
    // Determine if true global ball is in the online field of view

    CurrentHeadPositionVisibilityCode = 0;
    for (int i=0; i<ObservationsNumber; i++){
        if (Visible(Actions[0], Observations[i], Pose, BoundPoints, false) == 1){
            // std::cout<<i<<"\t";
            CurrentHeadPositionVisibilityCode = CurrentHeadPositionVisibilityCode + pow(2, i);
        }
    }
    // cout<<BestActionVisibilityCode<<"    "<<CurrentHeadPositionVisibilityCode<<endl;
    // std::string binary1 = std::bitset<25>(CurrentHeadPositionVisibilityCode).to_string(); //to binary
    // std::string binary2 = std::bitset<25>(BestActionVisibilityCode).to_string(); //to binary

    // std::cout<<binary1<<"\t"<<countSetBits(CurrentHeadPositionVisibilityCode)<<"\n";
    CurrentHeadPositionVisibilityCode = (BestActionVisibilityCode&CurrentHeadPositionVisibilityCode);
    
    // std::string binary3 = std::bitset<25>(CurrentHeadPositionVisibilityCode).to_string(); //to binary
    // std::cout<<binary2<<"\t"<<countSetBits(BestActionVisibilityCode)<<"\n";
    // std::cout<<binary3<<"\t"<<countSetBits(CurrentHeadPositionVisibilityCode)<<"\n";
    // std::cout<<hamming_distance(BestActionVisibilityCode, CurrentHeadPositionVisibilityCode)<<"\n";
    // std::cout<<"-----------------------------------------"<<"\n";
    Conditions[1] = 0;
    if (hamming_distance(BestActionVisibilityCode, CurrentHeadPositionVisibilityCode) == 0){
        Conditions[1] = 1;
    }

    Conditions[2] = countSetBits(CurrentHeadPositionVisibilityCode);
    Conditions[3] = countSetBits(BestActionVisibilityCode);

    // Cpp from just here
    if (DrawPointsOnlineCpp){

        for(int w=0;w<4;w++){
	        Prelative[0]=RealBoundPoints[w].x;
	        Prelative[1]=RealBoundPoints[w].y;
	        Prelative[2]=0;
	        pose_global(Prelative, Pose, GlobalPose);
	        RealBoundPoints[w].x = GlobalPose[0];
	        RealBoundPoints[w].y = GlobalPose[1];
        }

        for (int w=0;w<4;w++){
            RealBoundPoints[w].x = (FieldWidth/2+FieldPading+( RealBoundPoints[w].x *10))*ScaleToPixel;
            RealBoundPoints[w].y = (FieldHeight/2+FieldPading-( RealBoundPoints[w].y *10))*ScaleToPixel;
        }

        for(int counter=0;counter<4;counter++){
            circle(ShowMat, cv::Point(RealBoundPoints[counter].x, RealBoundPoints[counter].y), 5, cv::Scalar(255, 0, 0), -1, 4, 0);
            line(ShowMat, cv::Point(RealBoundPoints[counter].x, RealBoundPoints[counter].y), cv::Point(RealBoundPoints[(counter+1)%4].x, RealBoundPoints[(counter+1)%4].y), cv::Scalar(255, 0, 0), 1, 4, 0);
        }
        
    }
    // Cpp from just here

    // Show the supposed visible observations
    if (ActiveVisionBoundaryConfig){
        for (int i=0; i<NumOfFieldBoundaryLines; i++){
            temp = (ActionNumber * ObservationsNumber) + i;
            if (VisibilityStatus[temp]==1){
                DetectedCounter++;
                FeatureCounter = (ActionNumber* 4 * TotalLinesOfField) + (i * 4) ;
                // cout<<" AAAA "<<FeatureCounter<<endl;
                // cout<<ActionNumber<<" "<<i<<" "<<LineFeatures[FeatureCounter]<<" "<<LineFeatures[FeatureCounter+1]<<" "<<LineFeatures[FeatureCounter+2]<<" "<<LineFeatures[FeatureCounter+3]<<endl;
                StartPoint.x = (FieldWidth/2+FieldPading+( LineFeatures[FeatureCounter] *10))*ScaleToPixel;
                StartPoint.y = (FieldHeight/2+FieldPading-( LineFeatures[FeatureCounter+1] *10))*ScaleToPixel;
                EndPoint.x =   (FieldWidth/2+FieldPading+( LineFeatures[FeatureCounter+2] *10))*ScaleToPixel;
                EndPoint.y =   (FieldHeight/2+FieldPading-( LineFeatures[FeatureCounter+3] *10))*ScaleToPixel;
                line(ShowMat, cv::Point(StartPoint.x,StartPoint.y), cv::Point(EndPoint.x,EndPoint.y), cv::Scalar(255,0,0),5,4,0);
            }
        }
    }

    if (ActiveVisionLineConfig){
        for (int i=NumOfFieldBoundaryLines; i<TotalLinesOfField; i++){
            temp = (ActionNumber * ObservationsNumber) + i;
            if (VisibilityStatus[temp]==1){
                DetectedCounter++;
                FeatureCounter = (ActionNumber* 4 * TotalLinesOfField) + (i * 4) ;
                // cout<<" AAAA "<<FeatureCounter<<endl;
                // cout<<ActionNumber<<" "<<i<<" "<<LineFeatures[FeatureCounter]<<" "<<LineFeatures[FeatureCounter+1]<<" "<<LineFeatures[FeatureCounter+2]<<" "<<LineFeatures[FeatureCounter+3]<<endl;
                StartPoint.x = (FieldWidth/2+FieldPading+( LineFeatures[FeatureCounter] *10))*ScaleToPixel;
                StartPoint.y = (FieldHeight/2+FieldPading-( LineFeatures[FeatureCounter+1] *10))*ScaleToPixel;
                EndPoint.x =   (FieldWidth/2+FieldPading+( LineFeatures[FeatureCounter+2] *10))*ScaleToPixel;
                EndPoint.y =   (FieldHeight/2+FieldPading-( LineFeatures[FeatureCounter+3] *10))*ScaleToPixel;
                line(ShowMat, cv::Point(StartPoint.x,StartPoint.y), cv::Point(EndPoint.x,EndPoint.y), cv::Scalar(255,0,0),5,4,0);
            }
        }
    }

    if (ActiveVisionLCornerConfig){
        for (int j=TotalLinesOfField; j<TotalLinesOfField+NumOfLCorners; j++){
            temp = (ActionNumber * ObservationsNumber) + j;
            if (VisibilityStatus[temp]==1){
                DetectedCounter++;
                StartPoint.x = (FieldWidth/2+FieldPading+( Observations[j].startPoint.x *10))*ScaleToPixel;
                StartPoint.y = (FieldHeight/2+FieldPading-( Observations[j].startPoint.y *10))*ScaleToPixel;
                cv::circle(ShowMat, cv::Point(StartPoint.x,StartPoint.y), 5, cv::Scalar(147,20,255), -1, 4, 0);
            }
        }
    }

    if (ActiveVisionTCornerConfig){
        for (int j=TotalLinesOfField+NumOfLCorners; j<ObservationsNumber; j++){
            temp = (ActionNumber * ObservationsNumber) + j;
            if (VisibilityStatus[temp]==1){
                DetectedCounter++;
                StartPoint.x = (FieldWidth/2+FieldPading+( Observations[j].startPoint.x *10))*ScaleToPixel;
                StartPoint.y = (FieldHeight/2+FieldPading-( Observations[j].startPoint.y *10))*ScaleToPixel;
                cv::circle(ShowMat, cv::Point(StartPoint.x,StartPoint.y), 5, cv::Scalar(147,20,255), -1, 4, 0);
            }
        }
    }
    // Show the supposed visible observations

    if (WriteActionInfo){
        putText(ShowMat, to_string(ActionNumber), cv::Point(20,80), 1, 5, cv::Scalar(0,0,0), 5 );
        putText(ShowMat, to_string(int(Actions[ActionNumber].Yaw)), cv::Point(20,160), 1, 5, cv::Scalar(0,0,0), 5);
        putText(ShowMat, to_string(int(Actions[ActionNumber].Pitch)), cv::Point(20,240), 1, 5, cv::Scalar(0,0,0), 5);
    }

    if (MyPlotConfig && ForCheck){
        cv::imshow("Check", ShowMat);
        cv::waitKey(20);
    }

}

unsigned int countSetBits(unsigned int n){ 
    unsigned int count = 0; 
    while (n){ 
        count += n & 1; 
        n >>= 1; 
    } 
    return count; 
}

void SaveShowMat(int ExpEpisodeNum, int step){
    cv::imwrite("./Directory/" + std::to_string(ExpEpisodeNum) + "/Field" + std::to_string(step) + ".png", ShowMat);
}