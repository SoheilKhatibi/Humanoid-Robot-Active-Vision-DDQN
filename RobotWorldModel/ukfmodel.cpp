#include "ukfmodel.h"

#include "shmProvider.h"

#include <boost/foreach.hpp>

#include "luatables.h"

double deltaWeight;
int n;
int nSigPoints;
int alpha;
int beta;
int kappa;
double lambda;
double Modelgamma;

int useOnlyOdometry = 0;
int landmark_liklihood_thrd = 100;
int countObserveErr=0;

double xMax;
double yMax;
double rErrBestFactor = 0.10;
double aErrBestFactor = 2 * M_PI/180;

double aErrBestGlobal = 5 * M_PI/180;


double goal_liklihood_thrd = 150;

arma::mat wm;
arma::mat wc;

std::vector<arma::mat> cornerT(6);
std::vector<arma::mat> circle(2);
std::vector<arma::mat> cornerL(8);
std::vector<arma::mat> penaltyArea(2);
std::vector<arma::mat> spot(2);




void init(){
  
    LuaTable input (LuaTable::fromFile("WorldConfigForCppModel.lua"));

    xMax = input["xMax"].getDefault<double>(false);
    yMax = input["yMax"].getDefault<double>(false);

    // cout<<"222"<<endl;
    cornerT[0] = arma::mat(3,1);
    cornerT[0](0,0) = input["cornerT"][1][1].getDefault<double>(false);
    cornerT[0](1,0) = input["cornerT"][1][2].getDefault<double>(false);
    cornerT[0](2,0) = input["cornerT"][1][3].getDefault<double>(false);

    cornerT[1] = arma::mat(3,1);
    cornerT[1](0,0) = input["cornerT"][2][1].getDefault<double>(false);
    cornerT[1](1,0) = input["cornerT"][2][2].getDefault<double>(false);
    cornerT[1](2,0) = input["cornerT"][2][3].getDefault<double>(false);

    cornerT[2] = arma::mat(3,1);
    cornerT[2](0,0) = input["cornerT"][3][1].getDefault<double>(false);
    cornerT[2](1,0) = input["cornerT"][3][2].getDefault<double>(false);
    cornerT[2](2,0) = input["cornerT"][3][3].getDefault<double>(false);

    cornerT[3] = arma::mat(3,1);
    cornerT[3](0,0) = input["cornerT"][4][1].getDefault<double>(false);
    cornerT[3](1,0) = input["cornerT"][4][2].getDefault<double>(false);
    cornerT[3](2,0) = input["cornerT"][4][3].getDefault<double>(false);

    cornerT[4] = arma::mat(3,1);
    cornerT[4](0,0) = input["cornerT"][5][1].getDefault<double>(false);
    cornerT[4](1,0) = input["cornerT"][5][2].getDefault<double>(false);
    cornerT[4](2,0) = input["cornerT"][5][3].getDefault<double>(false);

    cornerT[5] = arma::mat(3,1);
    cornerT[5](0,0) = input["cornerT"][6][1].getDefault<double>(false);
    cornerT[5](1,0) = input["cornerT"][6][2].getDefault<double>(false);
    cornerT[5](2,0) = input["cornerT"][6][3].getDefault<double>(false);

    cornerL[0] = arma::mat(3,1);
    cornerL[0](0,0) = input["cornerL"][1][1].getDefault<double>(false);
    cornerL[0](1,0) = input["cornerL"][1][2].getDefault<double>(false);
    cornerL[0](2,0) = input["cornerL"][1][3].getDefault<double>(false);

    cornerL[1] = arma::mat(3,1);
    cornerL[1](0,0) = input["cornerL"][2][1].getDefault<double>(false);
    cornerL[1](1,0) = input["cornerL"][2][2].getDefault<double>(false);
    cornerL[1](2,0) = input["cornerL"][2][3].getDefault<double>(false);

    cornerL[2] = arma::mat(3,1);
    cornerL[2](0,0) = input["cornerL"][3][1].getDefault<double>(false);
    cornerL[2](1,0) = input["cornerL"][3][2].getDefault<double>(false);
    cornerL[2](2,0) = input["cornerL"][3][3].getDefault<double>(false);

    cornerL[3] = arma::mat(3,1);
    cornerL[3](0,0) = input["cornerL"][4][1].getDefault<double>(false);
    cornerL[3](1,0) = input["cornerL"][4][2].getDefault<double>(false);
    cornerL[3](2,0) = input["cornerL"][4][3].getDefault<double>(false);

    cornerL[4] = arma::mat(3,1);
    cornerL[4](0,0) = input["cornerL"][5][1].getDefault<double>(false);
    cornerL[4](1,0) = input["cornerL"][5][2].getDefault<double>(false);
    cornerL[4](2,0) = input["cornerL"][5][3].getDefault<double>(false);

    cornerL[5] = arma::mat(3,1);
    cornerL[5](0,0) = input["cornerL"][6][1].getDefault<double>(false);
    cornerL[5](1,0) = input["cornerL"][6][2].getDefault<double>(false);
    cornerL[5](2,0) = input["cornerL"][6][3].getDefault<double>(false);

    cornerL[6] = arma::mat(3,1);
    cornerL[6](0,0) = input["cornerL"][7][1].getDefault<double>(false);
    cornerL[6](1,0) = input["cornerL"][7][2].getDefault<double>(false);
    cornerL[6](2,0) = input["cornerL"][7][3].getDefault<double>(false);

    cornerL[7] = arma::mat(3,1);
    cornerL[7](0,0) = input["cornerL"][8][1].getDefault<double>(false);
    cornerL[7](1,0) = input["cornerL"][8][2].getDefault<double>(false);
    cornerL[7](2,0) = input["cornerL"][8][3].getDefault<double>(false);

    penaltyArea[0] = arma::mat(3,1);
    penaltyArea[0](0,0) = input["penaltyArea"][1][1].getDefault<double>(false);
    penaltyArea[0](1,0) = input["penaltyArea"][1][2].getDefault<double>(false);
    penaltyArea[0](2,0) = input["penaltyArea"][1][3].getDefault<double>(false);
    
    penaltyArea[1] = arma::mat(3,1);
    penaltyArea[1](0,0) = input["penaltyArea"][2][1].getDefault<double>(false);
    penaltyArea[1](1,0) = input["penaltyArea"][2][2].getDefault<double>(false);
    penaltyArea[1](2,0) = input["penaltyArea"][2][3].getDefault<double>(false);

    spot[0] = arma::mat(3,1);
    spot[0](0,0) = input["spot"][1][1].getDefault<double>(false);
    spot[0](1,0) = input["spot"][1][2].getDefault<double>(false);
    spot[0](2,0) = 0;

    spot[1] = arma::mat(3,1);
    spot[1](0,0) = input["spot"][2][1].getDefault<double>(false);
    spot[1](1,0) = input["spot"][2][2].getDefault<double>(false);
    spot[1](2,0) = 0;

    circle[0] = arma::mat(3,1);
    circle[0](0,0) = input["circle"][1][1].getDefault<double>(false);
    circle[0](1,0) = input["circle"][1][2].getDefault<double>(false);
    circle[0](2,0) = input["circle"][1][3].getDefault<double>(false);


    circle[1] = arma::mat(3,1);
    circle[0](0,0) = input["circle"][2][1].getDefault<double>(false);
    circle[0](1,0) = input["circle"][2][2].getDefault<double>(false);
    circle[0](2,0) = input["circle"][2][3].getDefault<double>(false);

    deltaWeight = 0.03;
    n = input["ukf"]["dimention"].getDefault<double>(false);
    nSigPoints = (2 * n) + 1 ; 
    alpha = input["ukf"]["alpha"].getDefault<double>(false);
    beta = input["ukf"]["beta"].getDefault<double>(false);
    kappa = input["ukf"]["kappa"].getDefault<double>(false);
    lambda = (pow(alpha, 2)) * (n + kappa) - n;
    Modelgamma = n + lambda;

    wm = arma::mat(nSigPoints,1,arma::fill::zeros);
    wc = arma::mat(nSigPoints,1,arma::fill::zeros);
    
    wm(0,0) = (lambda / ( n + lambda));
    wc(0,0) = lambda / (n + lambda) + (1 - pow(alpha,2) + beta);

    for(int i = 1; i<nSigPoints; i++){
        wm(i,0) = 1 / (2 * (n + lambda));
        wc(i,0) = wm(i);
    }

}


UKFModel::UKFModel(){
    sigmaPoints = arma::cube(3,1,7);
    spWeight = arma::mat(7,1);
    spWeight.fill(1/7);
    spImportanceFac = arma::mat(7, 1, arma::fill::zeros);
    mean = arma::mat(3, 1, arma::fill::zeros);
    cov = arma::mat(3, 3, arma::fill::zeros);
    wSlow = 10;
    wFast = 0;
    modelWeight = 0.0005;
    only_update_weigth = false;

}

void UKFModel::initialize(arma::mat s,arma::mat p,double w){
    mean = s;
    cov = p;
    modelWeight = w;
}

// generate sigma points with cholesky decomposition method.
void UKFModel::generate_sigma_points(){
      
      arma::mat L(3,3);
      // cov->print();
      // mean->print();
      // cout<<"#####################################################"<<endl;
      L = compute_cholesky_decomposition(cov);
      // L.print();
      // cout<<"________________________________________________________________________"<<endl;
      L = L * sqrt(Modelgamma);
      // mean->print("m2");
      sigmaPoints.slice(0) = mean;
      for (int i = 1;i<nSigPoints;i++){
          // cout<<i<<"i"<<endl;
          if (i % 2 == 1){
              // L.col(((i+1)/2)-1).print();
              // cout<<"A"<<endl;
              sigmaPoints.slice(i) = mean + L.col(((i+1)/2)-1);
              // sigmaPoints->slice(i).print();
          }else{
            // L.col((i/2)-1).print();
            // cout<<"B"<<endl;
            sigmaPoints.slice(i) = mean - L.col((i/2)-1);
            // sigmaPoints->slice(i).print();
          }
      }

      // for (int i = 0;i<nSigPoints;i++){
      //   sigmaPoints->slice(i).print();
      // }




}

// return squre root of matrix A.
arma::mat compute_cholesky_decomposition(arma::mat A){
    arma::mat L(3,3,arma::fill::zeros);
    L(0,0) = sqrt(fmax( A(0,0) , 0));
    if(L(0,0) == 0)
        L(0,0) = 0.0000000001;
    L(1,0) = A(1,0) / L(0,0);
    L(1,1) = sqrt(fmax(A(1,1) - pow(L(1,0), 2), 0));
    if(L(1,1) == 0)
        L(1,1) = 0.0000000001;
    L(2,0) = A(2,0) / L(0,0);
    L(2,1) = (A(2,1) - L(2,0) * L(1,0)) / L(1,1);
    L(2,2) = sqrt(fmax(A(2,2) - pow(L(2,0), 2) - pow(L(2,1), 2), 0));
    return L;
}

//update mean and cov with respect to the odometry data
void UKFModel::predict_pos(double dx,double dy,double da){
    // mean->print("m1");
    // mean_of_sigmaPoints().print("m1");
    //     for(int i= 0; i< nSigPoints;i++){
    //       sigmaPoints->slice(i).print();
    //     }
    // cout<<"$##################################################"<<endl;
    generate_sigma_points();
    // mean->print("m3");
    // mean_of_sigmaPoints().print("m2");
    // for(int i= 0; i< nSigPoints;i++){
    //       sigmaPoints->slice(i).print();
    //     }

    // cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
    //propagate odometry on sigma points
    for(int i= 0; i< nSigPoints;i++){
        double ca = cos(sigmaPoints(2,0,i));
        double sa = sin(sigmaPoints(2,0,i));
        arma::mat transMat(3,1);
        transMat(0,0) = (dx) * ca - (dy) * sa;
        transMat(1,0) = (dx) * sa + (dy) * ca;
        transMat(2,0) = da ;




        // cout<<"AAA"<<" "<<transMat(0,0)<<" "<<transMat(1,0)<<" "<<transMat(2,0)<<endl;
        // sigmaPoints->slice(i).print();
        // cout<<da<<" "<<transMat(2,0)<<endl;

        // transMat.print();

        sigmaPoints.slice(i) = sigmaPoints.slice(i) + transMat;
    }
    // for(int i= 0; i< nSigPoints;i++){
    //   cout<<sigmaPoints->slice(i).at(2)<<endl;
      
    // }
    // cout<<dx<<" "<<dy<<" "<<da<<endl;
    // mean_of_sigmaPoints().print();
    mean = mean_of_sigmaPoints();
    //TODO : the noise params must be evaluated precisely
    double noiseX = 0.2 * dx;
    double noiseY = 0.2 * dy;
    double noiseA = 0.1 * da;
    //motion noise
    arma::mat R(3,3,arma::fill::zeros);
    //process noise
    arma::mat pNoise(3,3,arma::fill::zeros);
    if(useOnlyOdometry!=1){
        R(0,0) = pow(noiseX, 2);
        R(1,1) = pow(noiseY, 2);
        R(2,2) = pow(noiseA, 2);
        pNoise(0,0) = pow(0.006, 2);
        pNoise(1,1) = pow(0.006, 2);
        pNoise(2,2) = pow(0.006, 2);
    }
    // R.print("R");
    // pNoise.print("P");
    // cov_of_sigmaPoints().print("CS");
    cov = cov_of_sigmaPoints() + R + pNoise ;
    checkBoundary();
}

void UKFModel::checkBoundary(){
  double boundErr = fmax(mean(0,0)-xMax,0)+ fmax(-mean(0,0)-xMax,0)+
                    fmax(mean(1,0)-yMax,0)+ fmax(-mean(1,0)-yMax,0);

  mean(0,0) = fmax(fmin(mean(0,0), xMax), -xMax);
  mean(1,0) = fmax(fmin(mean(1,0), yMax), -yMax);
  mean(2,0) = mod_angle(mean(2,0));
}

arma::mat UKFModel::mean_of_sigmaPoints(){
  arma::mat m(3,1,arma::fill::zeros);
  double sum_sa = 0;
  double sum_ca = 0;
  for(int i = 0; i<nSigPoints; i++){
    m(0,0) = m(0,0) + wm(i,0) * sigmaPoints(0,0,i);
    m(1,0) = m(1,0) + wm(i,0) * sigmaPoints(1,0,i);
    sum_sa = sum_sa + wm(i,0) * sin(sigmaPoints(2,0,i));
    sum_ca = sum_ca + wm(i,0) * cos(sigmaPoints(2,0,i));
  }

  // cout<<sum_sa<<" "<<sum_ca<<endl;
  m(2,0) = atan2(sum_sa, sum_ca);
  return m;
}


arma::mat cov_of_predictedZ(arma::mat* Z , arma::mat& mZ){
   arma::mat s(3,3,arma::fill::zeros);
   for (int i = 0; i<nSigPoints;i++){
       arma::mat d(3,1);
       d = state_vector_diff(Z[i] , mZ);
       s = s + (wc(i,0) * d * d.t()) ;
   }
   return s;
}

arma::mat UKFModel::cov_of_sigmaPoints(){
    arma::mat c(3,3,arma::fill::zeros);
    arma::mat deviation(3,1);
    for (int i = 0; i<nSigPoints;i++){
        deviation =state_vector_diff(sigmaPoints.slice(i) , mean);
        c = c + wc(i,0) *  deviation * deviation.t();
    }
    return c;
}

arma::mat pose2d_vector_diff(arma::mat& v1,arma::mat& v2){
    arma::mat d(2,1);
    d(0,0) = v1(0,0) - v2(0,0);
    d(1,0) = v1(1,0) - v2(1,0);
    return d;
}

arma::mat state_vector_diff(arma::mat& v1 ,arma::mat& v2){
  arma::mat Diff = v1 - v2;
  Diff(2,0) = mod_angle(v1(2,0)-v2(2,0));
  // cout<<v1(2,0)<<" "<<v2(2,0)<<" "<<Diff(2,0)<<endl;
  return Diff;
}

void UKFModel::set_weight(double w){
  modelWeight = w;
}

void UKFModel::converge(double x,double y,double a){
  mean(0,0) = x;
  mean(1,0) = y;
  mean(2,0) = a;
  arma::mat init_cov = "0.01 0 0; 0 0.01 0; 0 0 0;";
  init_cov(2,2) = pow(5*M_PI/180,2);
  cov = init_cov;
}

arma::mat UKFModel::get_pose(){
    arma::mat Ar = mean;
    Ar(2,0) = mod_angle(mean(2,0));
    return Ar;
}

double UKFModel::get_model_weight(){
    return modelWeight;
}

double mod_angle(double a){
    if (a<0)
      a = a + 2*M_PI;
    
    a = fmod(a, (2*M_PI));
    if (a >= M_PI)
        a = a - 2*M_PI;
    return a;
}

void UKFModel::observe_line(Line& A){
    double distanceToLine , x , y;
    robot_to_line_distance(A.sp, A.ep, distanceToLine, x, y);
    //print("sp,ep",line.sp[1],line.ep[1],line.sp[2],line.ep[2] )
    //print("distanceToLine , x , y",distanceToLine , x , y);
    //print("d2", WorldUtil.distance_point_to_line({0,0},line.sp,line.ep) )
    double angle = atan2(y,x);
    bool horizental = is_horizental(A);
    std::vector<Line> lmPos;
    //print("is_horizental:",horizental);
    if(horizental){
        lmPos.resize(5);
        for (int k=0;k<lmPos.size();k++){
            lmPos[k].sp = arma::mat(3,1,arma::fill::zeros); //######################################################
            lmPos[k].ep = arma::mat(3,1,arma::fill::zeros);
        }
    // cout<<"ComeToSeeTheRelatedLine update2222222"<<endl;
        lmPos[0].sp = "4.5; 3; 0;";
        lmPos[0].ep = "4.5; -3; 0;";
        lmPos[1].sp = "3.5; 2.5; 0;";
        lmPos[1].ep = "3.5; -2.5; 0;";
        lmPos[2].sp = "0; 3; 0;";
        lmPos[2].ep = "0; -3; 0;";
        lmPos[3].sp = "-3.5; 2.5; 0;";
        lmPos[3].ep = "-3.5; -2.5; 0;";
    // cout<<"ComeToSeeTheRelatedLine update"<<endl;
        lmPos[4].sp = "-4.5; 3; 0;";
        lmPos[4].ep = "-4.5; -3; 0;";
    }else{
        lmPos.resize(2);
        for (int k=0;k<lmPos.size();k++){
            lmPos[k].sp = arma::mat(3,1,arma::fill::zeros);
            lmPos[k].ep = arma::mat(3,1,arma::fill::zeros);
        }
        // cout<<"ComeToSeeTheRelatedLine update666666666666666"<<endl;
        lmPos[0].sp = "4.5; 3; 0;";
        lmPos[0].ep = "-4.5; 3; 0;";
        lmPos[1].sp = "4.5; -3; 0;";
        // cout<<"ComeToSeeTheRelatedLine update7777777777777777777777"<<endl;
        lmPos[1].ep = "-4.5; -3; 0;";
    }

  line_update_2d(distanceToLine,angle,lmPos);
}

void UKFModel::line_update_2d(double distance,double angle, std::vector<Line> lmPos){
    generate_sigma_points();

    // mean->t().print();
    // cout<<"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaa"<<endl;

    // for(int ipos= 0; ipos<lmPos.size();ipos++){
    //     cout<<lmPos[ipos].sp->at(0,0)<<"      "<<lmPos[ipos].sp->at(1,0)<<"      "<<lmPos[ipos].ep->at(0,0)<<"      "<<lmPos[ipos].ep->at(1,0)<<endl;
    // }
        // sigmaPoints->slice(isp).t().print();
    
    arma::mat observedZ(2,1);
    observedZ(0,0) = distance;
    observedZ(1,0) = angle;
    //print("observed",observedZ[1][1],observedZ[2][1]*180/math.pi)
    arma::mat R(2,2,arma::fill::zeros);
    if (distance < 1.8) {
	    R(0,0) = pow((0.08 * distance), 2);
	    R(1,1) = pow((0.08 * distance), 2);
    }else if (distance <2.5) {
		R(0,0) = pow((0.11 * distance), 2);
	    R(1,1) = pow((0.11 * distance), 2);
	}else if (distance <4.5) {
		R(0,0) = pow((0.13 * distance), 2);
        R(1,1) = pow((0.13 * distance), 2);
    }else if (distance <6){
        R(0,0) = pow((0.15 * distance), 2);
        R(1,1) = pow((0.15 * distance), 2);
	}else{
		R(0,0) = pow((0.65 * distance), 2);
	    R(1,1) = pow((0.65 * distance), 2);
    }
    
    arma::mat predictedZ[lmPos.size()][nSigPoints];
    arma::mat S[lmPos.size()];
    arma::mat invS[lmPos.size()];
    arma::mat covXandZ[lmPos.size()];
    arma::mat meanPredictedZ[lmPos.size()];
    arma::mat likelihood(lmPos.size(),1);
    arma::mat err[lmPos.size()];
    arma::mat compensatedErr[lmPos.size()];

    double rSigma = .25 * distance + 0.20;
    double aSigma = 5 * M_PI / 180;
    double rErrBest = rErrBestFactor *distance;
    double aErrBest = aErrBestFactor *distance;
    double wBestDis = get_gussian_prob(rErrBest,rSigma);
    double wBestAngle = get_gussian_prob(aErrBest,aSigma);



    for(int ipos= 0;ipos<lmPos.size();ipos++ ){
        for(int isp= 0; isp<nSigPoints;isp++){
            Line relLmPos;
            relLmPos.sp = arma::mat(3,1);
            relLmPos.ep = arma::mat(3,1);
            // lmPos[ipos].sp->t().print("s");
            // lmPos[ipos].ep->t().print("s");
            // sigmaPoints->slice(isp).t().print("q");
            relLmPos.sp = pose_relative(lmPos[ipos].sp, sigmaPoints.slice(isp));
            relLmPos.ep = pose_relative(lmPos[ipos].ep, sigmaPoints.slice(isp));
            
            // sigmaPoints->slice(isp).t().print();
            // cout<<ipos<<"      "<<isp<<"      "<<lmPos[ipos].sp->at(0,0)<<"      "<<lmPos[ipos].sp->at(1,0)<<"      "<<relLmPos.sp->at(0,0)<<"      "<<relLmPos.sp->at(1,0)<<"      "<<relLmPos.ep->at(0,0)<<"      "<<relLmPos.ep->at(1,0)<<endl;
            predictedZ[ipos][isp] = arma::mat(2,1);
            double distanceToLine , x , y;
            robot_to_line_distance(relLmPos.sp, relLmPos.ep, distanceToLine, x, y);
            predictedZ[ipos][isp](0,0) = distanceToLine;
            predictedZ[ipos][isp](1,0) = atan2(y,x);
        }



        meanPredictedZ[ipos] = mean_of_predictedZ_dis_angle(predictedZ[ipos]);
        //print("",meanPredictedZ[ipos][1][1],meanPredictedZ[ipos][2][1]*180/math.pi)
        S[ipos] = arma::mat(2,2,arma::fill::zeros);
        S[ipos] = cov_of_predictedZ_2d(predictedZ[ipos],meanPredictedZ[ipos]);
        S[ipos] = S[ipos] + R;



        invS[ipos] = arma::mat(2,2,arma::fill::zeros);
        invS[ipos] = S[ipos].i();
        covXandZ[ipos] = arma::mat(3,2);
        covXandZ[ipos] = cov_of_sigmaPoint_and_predictedZ_2d(predictedZ[ipos],meanPredictedZ[ipos]);
        //print(ipos,"s",S[ipos]);
        //print("invs",invS[ipos]);
        err[ipos] = arma::mat(2,1);
        err[ipos](0,0) = observedZ(0,0) - meanPredictedZ[ipos](0,0);
        err[ipos](1,0) = mod_angle(observedZ(1,0) - meanPredictedZ[ipos](1,0));
        compensatedErr[ipos] = arma::mat(2,1);
        compensatedErr[ipos](0,0) = err[ipos](0,0);
        compensatedErr[ipos](1,0) = 20 * err[ipos](1,0);
        // cout<<ipos<<"   "<<err[ipos]->at(0,0)<<"   "<<err[ipos]->at(1,0)*180/M_PI<<endl;
        arma::mat a = compensatedErr[ipos].t() * invS[ipos] * compensatedErr[ipos];
        likelihood(ipos,0) = a(0,0);
    }


    int imin;
    double likelihoodMin= FindMin(likelihood, imin);
    // cout<<"cppline";
    // likelihood.t().print();
    double rErr = err[imin](0,0);
    double aErr = err[imin](1,0);
    double masurmentW = get_gussian_prob(rErr,rSigma)/wBestDis * get_gussian_prob(aErr,aSigma)/wBestAngle;
    //print("selected pose:",imin,rErr,aErr,likelihoodMin,masurmentW,wBestDis,wBestAngle)
    //update_weights(t,masurmentW);
    update_model_weight(masurmentW);
    //skip outlier
    if (!only_update_weigth){
        if (likelihoodMin <= landmark_liklihood_thrd){
            arma::mat kGain(3,2);
            kGain = covXandZ[imin] * invS[imin] ;
            mean = mean + kGain * err[imin];
            cov = cov - kGain * S[imin] * kGain.t();
            countObserveErr = 0;
            checkBoundary();
            //if(goal_liklihood_thrd > 5){
                //--landmark_liklihood_thrd = landmark_liklihood_thrd * 0.9;
            //}
        }
    }

}

bool UKFModel::is_horizental(Line& A){
    arma::mat startVec(3,1);
    startVec(0,0) = A.sp(0,0);
    startVec(1,0) = A.sp(1,0);
    startVec(2,0) = A.sp(2,0);
    arma::mat endVec(3,1);
    endVec(0,0) = A.ep(0,0);
    endVec(1,0) = A.ep(1,0);
    endVec(2,0) = A.ep(2,0);

    arma::mat P = get_pose();
    //print("robot_angle",pa*180/math.pi);

    arma::mat startOnField = pose_global(startVec,P);
    arma::mat endOnField = pose_global(endVec,P);

    return abs(startOnField(0,0) - endOnField(0,0)) < abs(startOnField(1,0) - endOnField(1,0));
}

arma::mat pose_global(arma::mat pRelative,arma::mat Pose){
    double ca = cos(Pose(2,0));
    double sa = sin(Pose(2,0));
    arma::mat GlobalPose(3,1);
    GlobalPose(0,0) = Pose(0,0) + ca*pRelative(0,0) - sa*pRelative(1,0);
    GlobalPose(1,0) = Pose(1,0) + sa*pRelative(0,0) + ca*pRelative(1,0);
    GlobalPose(2,0) = Pose(2,0) + pRelative(2,0);
    return GlobalPose;
}

void robot_to_line_distance(arma::mat& p1, arma::mat& p2, double& distanceToLine, double& x, double& y){
    double dx = p2(0,0) - p1(0,0);
    double dy = p2(1,0) - p1(1,0);
    double a = sqrt(pow(dx,2) + pow(dy,2))+ 0.0000001;
    double b = sqrt(pow(p1(0,0),2) + pow(p1(1,0),2));
    double c = sqrt(pow(p2(0,0),2) + pow(p2(1,0),2));
    double s = (a+b+c) / 2;
    double len = fmax(((s-a) * (s-b) * (s-c) * s),0);
    distanceToLine = (2 * sqrt(len)) / a;

    double m1= (p2(0,0)-p1(0,0)+0.0000001)/(p2(1,0)- p1(1,0)+ 0.0000001);
    b=p1(0,0)-(m1*p1(1,0));
    double m2=-1/m1;
    double diffSlop = m2-m1;
    if (diffSlop == 0)
        diffSlop = 0.0000001;
    y=b/(diffSlop);
    x=m2*b/(diffSlop);
}

double get_gussian_prob(double x,double s){
  double sqrt2p = sqrt(2*M_PI);
  return  fmax((1 / (s * sqrt2p)) * exp(-0.5 * pow(x/s,2)) ,0.0000001);
}

arma::mat pose_relative(arma::mat pGlobal, arma::mat Pose){
    double ca = cos(Pose(2,0));
    double sa = sin(Pose(2,0));
    double px = pGlobal(0,0)-Pose(0,0);
    double py = pGlobal(1,0)-Pose(1,0);
    double pa = pGlobal(2,0)-Pose(2,0);
    arma::mat RelativePose(3,1);
    RelativePose(0,0) = ca*px + sa*py;
    RelativePose(1,0) = -sa*px + ca*py;
    RelativePose(2,0) = mod_angle(pa);
    return RelativePose;
}

arma::mat mean_of_predictedZ_dis_angle(arma::mat* t){
    arma::mat m(2,1,arma::fill::zeros);
    double sum_sa = 0;
    double sum_ca = 0;
    for(int i= 0; i<nSigPoints; i++){
        m(0,0) += wm(i,0) * t[i](0,0);
        sum_sa += wm(i,0) * sin(t[i](1,0));
        sum_ca += wm(i,0) * cos(t[i](1,0));
    }
    m(1,0) = atan2(sum_sa, sum_ca);
  return m;
}

arma::mat cov_of_predictedZ_2d(arma::mat* Z,arma::mat& mZ){
  arma::mat s(2,2,arma::fill::zeros);
  for(int i= 0;i<nSigPoints;i++){
    arma::mat d(2,1);
    d = state_vector_diff_2d(Z[i] , mZ);
    s = s + (wc(i,0) * d * d.t()) ;
  }
  return s;
}

arma::mat state_vector_diff_2d(arma::mat& v1 , arma::mat& v2){
  arma::mat d(2,1);
  d(0,0) = v1(0,0) - v2(0,0);
  d(1,0) = mod_angle(v1(1,0) - v2(1,0));
  return d;
}

arma::mat UKFModel::cov_of_sigmaPoint_and_predictedZ_2d(arma::mat* Z,arma::mat& mZ){
  arma::mat c(3,2,arma::fill::zeros);
  for(int i = 0;i<nSigPoints;i++){
    c = c + wc(i,0) *  (state_vector_diff(sigmaPoints.slice(i) , mean)) * (state_vector_diff_2d(Z[i] , mZ)).t();
  }
  return c;
}

double FindMin(arma::mat t,int& imin){
  imin = 0;
  double tmin = HUGE_VALL;
  for(int i = 0;i<t.n_rows;i++){
    if (t(i,0) < tmin){
      tmin = t(i,0);
      imin = i;
    }
  }
  return tmin;
}

void UKFModel::update_model_weight(double masurmentW){
  if (masurmentW > modelWeight + deltaWeight){
      modelWeight += deltaWeight;
  }else if (masurmentW < modelWeight -deltaWeight){
      modelWeight -= deltaWeight;
  }else{
    modelWeight = masurmentW;
  }
  modelWeight = fmax(0,fmin(1,modelWeight));
}

void JustForTest(){
  std::cout<<" AAAA "<<std::endl;
}

void init_kalmanFilter(UKFModel& UKF,double x,double y,double a,arma::mat uncCovMat,double w){
  arma::mat initMean(3,1,arma::fill::zeros);
  initMean(0,0) = x;
  initMean(1,0) = y;
  initMean(2,0) = a;

  
  UKF.initialize(initMean,uncCovMat,w);
}

void get_odometry(std::vector<double>& uOdometry0 ,std::vector<double>& uOdometry,double* walkULeft,double* walkURight){
  double uFoot[3]; 
  se2_interpolate(0.5, walkULeft, walkURight,uFoot);

  // cout<<uFoot[0]<<" "<<uFoot[1]<<" "<<uFoot[2]<<endl;
  // cout<<uOdometry0[0]<<" "<<uOdometry0[1]<<" "<<uOdometry0[2]<<endl;
  arma::mat A(3,1);
  A(0,0) = uFoot[0];
  A(1,0) = uFoot[1];
  A(2,0) = uFoot[2];
  arma::mat B(3,1);
  B(0,0) = uOdometry0[0];
  B(1,0) = uOdometry0[1];
  B(2,0) = uOdometry0[2];
  
  arma::mat C = pose_relative(A, B);

  uOdometry0[0] = uFoot[0];
  uOdometry0[1] = uFoot[1];
  uOdometry0[2] = uFoot[2];

  // C.print("C");
  uOdometry[0] = C(0,0);
  // cout<<uOdometry[0]<<endl;
  uOdometry[1] = C(1,0);
  uOdometry[2] = C(2,0);
}

void se2_interpolate(double t,double* u1,double* u2,double* answer){
  answer[0] = u1[0] + t*(u2[0]-u1[0]);
  answer[1] = u1[1]+t*(u2[1]-u1[1]);
  answer[2] = u1[2]+t*mod_angle(u2[2]-u1[2]);
}

void UKFModel::observe_boundary_single_line(Line& A){
    double distanceToLine , x , y;
    robot_to_line_distance(A.sp, A.ep, distanceToLine, x, y);
    // print("sp,ep",line.sp[1],line.ep[1],line.sp[2],line.ep[2])
    // print("distanceToLine , x , y",distanceToLine , x , y);
    // print("d2", WorldUtil.distance_point_to_line({0,0},line.sp,line.ep))
    double angle = atan2(y,x);
    std::vector<Line> lmPos;
    bool horizental = is_horizental(A);
    // cout<<"is_horizental:  "<<horizental<<endl;
    lmPos.resize(2);
    for (int k=0;k<lmPos.size();k++){
            lmPos[k].sp = arma::mat(3,1,arma::fill::zeros);
            lmPos[k].ep = arma::mat(3,1,arma::fill::zeros);
    }

    if(horizental){
        lmPos[0].sp(0,0) = xMax;
        lmPos[0].sp(1,0) = yMax;
        lmPos[0].sp(2,0) = 0;

        lmPos[0].ep(0,0) = xMax;
        lmPos[0].ep(1,0) = -yMax;
        lmPos[0].ep(2,0) = 0;


        lmPos[1].sp(0,0) = -xMax;
        lmPos[1].sp(1,0) = yMax;
        lmPos[1].sp(2,0) = 0;

        lmPos[1].ep(0,0) = -xMax;
        lmPos[1].ep(1,0) = -yMax;
        lmPos[1].ep(2,0) = 0;

    }else{
        lmPos[0].sp(0,0) = xMax;
        lmPos[0].sp(1,0) = yMax;
        lmPos[0].sp(2,0) = 0;

        lmPos[0].ep(0,0) = -xMax;
        lmPos[0].ep(1,0) = yMax;
        lmPos[0].ep(2,0) = 0;


        lmPos[1].sp(0,0) = xMax;
        lmPos[1].sp(1,0) = -yMax;
        lmPos[1].sp(2,0) = 0;

        lmPos[1].ep(0,0) = -xMax;
        lmPos[1].ep(1,0) = -yMax;
        lmPos[1].ep(2,0) = 0;

    }

    // cout<<"CPP   "<<distanceToLine<<" "<<angle<<endl;
    // for(int ipos= 0; ipos<lmPos.size();ipos++){
    //     cout<<lmPos[ipos].sp->at(0,0)<<"      "<<lmPos[ipos].sp->at(1,0)<<"      "<<lmPos[ipos].ep->at(0,0)<<"      "<<lmPos[ipos].ep->at(1,0)<<endl;
    // }
  
    line_update_2d(distanceToLine,angle,lmPos);
}

void UKFModel::observe_boundary_two_line(Line& line1,Line& line2){
    std::vector<arma::mat> pos;
    pos.resize(4);
    for(int k=0;k<pos.size();k++){
        pos[k] = arma::mat(3,1,arma::fill::zeros);
    }

    pos[0](0,0) = xMax;
    pos[0](1,0) = yMax;
    pos[0](2,0) = -0.75*M_PI;

    pos[1](0,0) = xMax;
    pos[1](1,0) = -yMax;
    pos[1](2,0) = 0.75*M_PI;

    pos[2](0,0) = -xMax;
    pos[2](1,0) = yMax;
    pos[2](2,0) = -0.25*M_PI;

    pos[3](0,0) = -xMax;
    pos[3](1,0) = -yMax;
    pos[3](2,0) = 0.25*M_PI;

    arma::mat vCenter = line1.ep;
    double angle1 = atan2(line1.sp(1,0)-vCenter(1,0),line1.sp(0,0)-vCenter(0,0));
    double angle2 = atan2(line2.ep(1,0)-vCenter(1,0),line2.ep(0,0)-vCenter(0,0));
    double angle;
    // print("vcenter",vCenter[1],vCenter[2])
    if (angle1*angle2>=0 || fabs(angle2)+fabs(angle1)<=M_PI ){
        angle=(angle1+angle2)/2;
    }else{
        angle=(angle1+angle2)/2;
        if (angle>0){
            angle=angle-M_PI;
        }else{
            angle=angle+M_PI;
        }  
    }
    observe_corner(pos,angle,vCenter);
}

void UKFModel::observe_corner(std::vector<arma::mat>& pos,double angle,arma::mat& vCenter){
    std::vector<PossiblePosition> poses(pos.size());
    // cout<<vCenter(0,0)<<"    "<<vCenter(1,0)<<endl;
    double r = sqrt(pow(vCenter(0,0), 2) +pow(vCenter(1,0), 2));
    // cout<<"dis to center: "<<r<<endl;
    double arel = atan2(vCenter(1,0),vCenter(0,0));
    arma::mat myPos = get_pose();//------------------------------------------------------------------------
    // double posexya = {myPosX,myPosY,myPosA};-------------------------------------------------------------------
    // cout<<"angle: "<<angle*180/M_PI<<endl;
    for(int i = 0; i<pos.size();i++){
        double xpos = pos[i].at(0,0);
        double ypos = pos[i].at(1,0);
        double apos = pos[i].at(2,0);
        poses[i].a = mod_angle(apos - angle);
        double aObj = poses[i].a + arel;
        double xrel = r*cos(aObj);
        double yrel = r*sin(aObj);
    
        arma::mat poseRel = pose_relative(pos[i],myPos);
        double objAng = atan2(poseRel(1,0),poseRel(0,0));
        int inFront;
        if(fabs(objAng) < M_PI/2){
            inFront = 1; 
        }else{
            inFront = 0; 
        }// is object in front or behind
        poses[i].x = xpos - xrel;
        poses[i].y = ypos - yrel;
        poses[i].Front = inFront;
        // print("inFront"..inFront);
    }
    arma::mat Q(3,3,arma::fill::zeros);
    if (r < 1.8){
        Q(0,0) = pow((0.06 * r), 2);
        Q(1,1) = pow((0.06 * r), 2);
        Q(2,2) = pow((0.06), 2);
    }else if(r < 2.5 ){
        Q(0,0) = pow((0.08 * r), 2);
        Q(1,1) = pow((0.08 * r), 2);
        Q(2,2) = pow((0.08), 2);
    }else if(r < 4.5){
        Q(0,0) = pow((0.11 * r), 2);
        Q(1,1) = pow((0.11 * r), 2);
        Q(2,2) = pow((0.11), 2);
    }else if (r < 6){
        Q(0,0) = pow((0.13 * r), 2);
        Q(1,1) = pow((0.13 * r), 2);
        Q(2,2) = pow((0.13), 2);
    }else{
		Q(0,0) = pow((0.6 * r), 2);
        Q(1,1) = pow((0.6 * r), 2);
        Q(2,2) = pow((0.6), 2);
    }


    // for(int k=0;k<poses.size();k++){
    //     cout<<poses[k].x<<"    "<<poses[k].y<<"    "<<poses[k].a<<"    "<<poses[k].Front<<endl;
    //  }

    pose_update(poses, Q,r);
}

void UKFModel::pose_update(std::vector<PossiblePosition>& poses,arma::mat& Q,double r){
    generate_sigma_points();
    arma::mat predictedZ[nSigPoints];
    for(int isp= 0; isp<nSigPoints;isp++){
        predictedZ[isp] = arma::mat(3,1);
	    predictedZ[isp](0,0) = sigmaPoints.slice(isp)(0,0);
	    predictedZ[isp](1,0) = sigmaPoints.slice(isp)(1,0); 
    	predictedZ[isp](2,0) = sigmaPoints.slice(isp)(2,0);
    }
 
    arma::mat meanPredictedZ(3,1);
    meanPredictedZ = mean_of_sigmaPoints();

    // meanPredictedZ.t().print();
 
    arma::mat S(3,3,arma::fill::zeros);
    S = cov_of_predictedZ(predictedZ , meanPredictedZ) ;
    // S.print();
  
    goal_liklihood_thrd = r * 10;
    S = S + Q ;
    arma::mat invS(3,3,arma::fill::zeros);
    invS = S.i();
    arma::mat covXandZ(3,3,arma::fill::zeros);
    covXandZ = (cov_of_sigmaPoint_and_predictedZ (predictedZ , meanPredictedZ));
  
    arma::mat observedZ[poses.size()];
    for (int ipos = 0;ipos< poses.size();ipos++){   
        observedZ[ipos] = arma::mat(3,1);
        observedZ[ipos](0,0) = poses[ipos].x;
        observedZ[ipos](1,0) = poses[ipos].y;
        observedZ[ipos](2,0) = poses[ipos].a;
        // observedZ[ipos]->t().print();
    }

    arma::mat err[poses.size()];
    arma::mat likelihood(poses.size(),1);	
    for (int ipos = 0; ipos<poses.size();ipos++){
        err[ipos] = arma::mat(3,1);
        err[ipos](0,0) = observedZ[ipos](0,0) - meanPredictedZ(0,0);
        err[ipos](1,0) = observedZ[ipos](1,0) - meanPredictedZ(1,0);
        err[ipos](2,0) = mod_angle(observedZ[ipos](2,0) - meanPredictedZ(2,0));
        arma::mat a = err[ipos].t() *  invS * err[ipos];
        // err[ipos].t().print();
        // invS.print();
        // err[ipos](0,0).print();
        likelihood(ipos,0) = a(0,0);
        // print("possible pose:",poses[ipos].x,poses[ipos].y,poses[ipos].a)
    }
  
    int imin;
    double likelihoodMin= FindMin(likelihood, imin);
    // cout<<"cpp"<<imin<<endl;
    // cout<<"cpp ";
    // likelihood.t().print();
    arma::mat kGain(3,3);
    kGain = covXandZ * invS;
    //  std::cout<<"selected pose:"<<" "<<mean(0,0)<<" "<<mean(1,0)<<" "<<mean(2,0)<<" "<<imin<<" "<<err[imin](0,0)<<" "<<err[imin](1,0)<<" "<<err[imin](2,0)<<std::endl;
    double rSigma = 0.25*r + 0.20;
    double aSigma = 5*M_PI/180;
    double rErrBest = r * rErrBestFactor;
    double aErrBest = r * rErrBestFactor;
    double wBestDis = get_gussian_prob(rErrBest,rSigma);
    double wBestAngle = get_gussian_prob(aErrBest,aSigma);
    double rErr = sqrt(pow(err[imin](0,0), 2) + pow(err[imin](0,0), 2));
    double aErr = err[imin](2,0);
    double masurmentW = get_gussian_prob(rErr,rSigma)/wBestDis * get_gussian_prob(aErr,aSigma)/wBestAngle;
    update_model_weight(masurmentW);
    if (!only_update_weigth){
        if (likelihoodMin <= goal_liklihood_thrd){
            // std::cout<<" likelihoodMin : "<<likelihoodMin<<std::endl;
            mean = mean +  kGain * err[imin];
            cov = cov - kGain * S * kGain.t();
			checkBoundary();
        }
    }
}

arma::mat UKFModel::cov_of_sigmaPoint_and_predictedZ(arma::mat* Z,arma::mat& mZ){
    arma::mat c(3,3,arma::fill::zeros);
    for(int i = 0;i< nSigPoints;i++ ){
        c = c + wc(i,0) *  (state_vector_diff(sigmaPoints.slice(i) , mean)) * (state_vector_diff(Z[i] , mZ)).t();
    }
    return c;
}

void UKFModel::observe_cornerT(Line& line1,Line& line2,double angle,arma::mat& intersection){
    observe_corner(cornerT,angle,intersection);
}

void UKFModel::observe_cornerL(Line& line1,Line& line2,double angle,arma::mat& intersection){
    observe_corner(cornerL,angle,intersection);
}

void UKFModel::observe_centerT(arma::mat& intersection,double angle){
	std::vector<arma::mat> centerT;
    centerT.push_back(cornerT[4]);
    centerT.push_back(cornerT[5]);
	observe_corner(centerT,angle,intersection);
}

void UKFModel::observe_centerCircle(arma::mat& intersection,double angle){
    observe_corner(circle,angle,intersection);
}

void UKFModel::observe_field_cornerT(arma::mat& intersection,double angle){
	std::vector<arma::mat> CornerT;
    CornerT.push_back(cornerT[0]);
    CornerT.push_back(cornerT[1]);
    CornerT.push_back(cornerT[2]);
    CornerT.push_back(cornerT[3]);
	observe_corner(CornerT,angle,intersection);
}

void UKFModel::observe_penalty_area(Line& line1,Line& line2,double angle,arma::mat& intersection){
    observe_corner(penaltyArea,angle,intersection);
}

void UKFModel::observe_parallel_lines(Line& line){

    double distanceToLine , x , y;
    robot_to_line_distance(line.sp,line.ep,distanceToLine , x , y);
    double angle = atan2(y,x);
    std::vector<Line> lmPos;
    bool horizental = is_horizental(line);
    if(line.type==1){
        lmPos.resize(2);
        
        lmPos[0].sp = arma::mat(3,1);
        lmPos[0].sp(0,0) = 3.5;
        lmPos[0].sp(1,0) = 2.5;
        lmPos[0].sp(2,0) = 0;

        lmPos[0].ep = arma::mat(3,1);
        lmPos[0].ep(0,0) = 3.5;
        lmPos[0].ep(1,0) = -2.5;
        lmPos[0].ep(2,0) = 0;

        
        lmPos[1].sp = arma::mat(3,1);
        lmPos[1].sp(0,0) = -3.5;
        lmPos[1].sp(1,0) = 2.5;
        lmPos[1].sp(2,0) = 0;
        
        lmPos[1].ep = arma::mat(3,1);
        lmPos[1].ep(0,0) = -3.5;
        lmPos[1].ep(1,0) = -2.5;
        lmPos[1].sp(2,0) = 0;

        line_update_2d(distanceToLine,angle,lmPos);
    }
}

arma::mat UKFModel::get_mean(){
    return mean;
}

arma::mat UKFModel::get_cov(){
    return cov;
}

void UKFModel::observe_spot(double* v){
    landmark_update_2d(spot,v);
}

void UKFModel::landmark_update_2d(std::vector<arma::mat>& lmPos,double* v){
    double distance = sqrt( pow( v[0], 2) + pow( v[1], 2));
	double rSigma = .25 * distance + 0.20;
    double aSigma = 5 * M_PI / 180;
    double rErrBest = rErrBestFactor * distance;
    double wBestDis = get_gussian_prob(rErrBest,rSigma);
    double wBestAngle = get_gussian_prob(aErrBestGlobal,aSigma);
	
	int bestMatcheLandmarkIndex = get_associative_landmark(mean,v,lmPos);
	// print("imin",bestMatcheLandmarkIndex);
	generate_sigma_points();
	arma::mat observedZ(2,1);
	observedZ(0,0) = v[0];
	observedZ(1,0) = v[1];
	
	arma::mat R(2,2,arma::fill::zeros);

    if (distance < 1.8) {
	    R(0,0) = pow((0.06 * distance), 2);
	    R(1,1) = pow((0.06 * distance), 2);
    }else if (distance <2.5) {
		R(0,0) = pow((0.08 * distance), 2);
	    R(1,1) = pow((0.08 * distance), 2);
	}else if (distance <4.5) {
		R(0,0) = pow((0.13 * distance), 2);
        R(1,1) = pow((0.13 * distance), 2);
	}else{
		R(0,0) = pow((0.6 * distance), 2);
	    R(1,1) = pow((0.6 * distance), 2);
    }
	
	arma::mat predictedZ[nSigPoints];
    arma::mat landmarkInRobot(3,1);
    for(int isp= 0; isp<nSigPoints;isp++){
		landmarkInRobot = pose_relative(lmPos[bestMatcheLandmarkIndex],sigmaPoints.slice(isp));

        predictedZ[isp] = arma::mat(2,1);
		predictedZ[isp](0,0) = landmarkInRobot(0,0);
		predictedZ[isp](1,0) = landmarkInRobot(1,0);
    }
	
	arma::mat meanPredictedZ = mean_of_predictedZ(predictedZ);

	arma::mat S(2,2,arma::fill::zeros);
	S = cov_of_predictedZ_pose2d(predictedZ,meanPredictedZ);
	S = S + R;
	
	arma::mat invS(2,2,arma::fill::zeros);
	invS = S.i();
	arma::mat covXandZ(3,2);
	covXandZ = cov_of_sigmaPoint_and_predictedZ_pose2d(predictedZ,meanPredictedZ);
	
	arma::mat err = observedZ - meanPredictedZ;
	
	double rErr = sqrt( pow(err(0,0), 2) + pow(err(1,0), 2));
	double masurmentW = get_gussian_prob(rErr,rSigma)/wBestDis;
    update_model_weight(masurmentW);
	
    arma::mat a = err.t() *  invS * err;
	double likelihood = a(0,0);
	if (likelihood < landmark_liklihood_thrd) {
		arma::mat kGain(3,2);
		kGain = covXandZ * invS ;
		mean = mean + kGain * err;
		cov = cov - kGain * S * kGain.t();
		checkBoundary(); 
    }
	// print("t.cov",t.cov);
}

int get_associative_landmark(arma::mat& robotPose,double* detectedLandmark, std::vector<arma::mat>& landmarksInField){
	int bestMatchIndex = 0;
	double bestMatchError = HUGE_VALL;

    arma::mat detectedLandmarkInRobot(3,1);
    double dx,dy,da,error;
	for (int i=0; i<landmarksInField.size(); i++){
		detectedLandmarkInRobot = pose_relative( landmarksInField[i],robotPose);
		dx = detectedLandmark[0] - detectedLandmarkInRobot(0,0);
		dy = detectedLandmark[1] - detectedLandmarkInRobot(1,0);
		da = mod_angle(atan2(detectedLandmarkInRobot(1,0),detectedLandmarkInRobot(0,0)) - atan2(detectedLandmark[1],detectedLandmark[0]));
		// print(i,math.atan2(detectedLandmarkInRobot[2],detectedLandmarkInRobot[1])*180/math.pi, math.atan2(detectedLandmark[2],detectedLandmark[1])*180/math.pi,dx,dy,da*180/math.pi);
		error = pow(dx, 2) + pow(dy, 2) + 20*pow(da, 2);
		if(error < bestMatchError){
			bestMatchError = error;
			bestMatchIndex = i;
        }
    }
	return bestMatchIndex;
}

arma::mat mean_of_predictedZ(arma::mat* t){
    arma::mat m( t[0].n_rows, 1, arma::fill::zeros);
    for (int i=0; i<nSigPoints; i++){
        m = m + wm(i,0) * t[i];
    }
    return m;
}

arma::mat cov_of_predictedZ_pose2d(arma::mat* Z,arma::mat& mZ){
	arma::mat s(2,2,arma::fill::zeros);
    for (int i=0; i<nSigPoints; i++){
        arma::mat d(2,1);
        d = pose2d_vector_diff(Z[i] , mZ); 
        s = s + wc(i,0) * d * d.t();
    }
    return s;
}

arma::mat UKFModel::cov_of_sigmaPoint_and_predictedZ_pose2d(arma::mat* Z , arma::mat& mZ){
    arma::mat c(3,2,arma::fill::zeros);
    for (int i=0; i<nSigPoints; i++){
        c = c + wc(i,0) *  (state_vector_diff(sigmaPoints.slice(i) , mean)) * (pose2d_vector_diff(Z[i] , mZ)).t();
    }
    return c;
}
