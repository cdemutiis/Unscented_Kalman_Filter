#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;  

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.6; 
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.

  is_initialized_ = false;
  n_x_ = 5;
  n_aug_ = 7;
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  lambda_ = 3 - n_aug_;
  weights_ = VectorXd(2*n_aug_+1);
  // set weights
  double weight_0 = lambda_/(lambda_+n_aug_);
  weights_(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; i++) {  //2n+1 weights
    double weight = 0.5/(n_aug_+lambda_);
    weights_(i) = weight;
  }

  
  P_ << 1,0,0,0,0,
       0,1,0,0,0,
       0,0,1,0,0,
       0,0,0,1,0,
       0,0,0,0,1;

}

UKF::~UKF() {}

void UKF::initialise(MeasurementPackage meas_package) {
  // first measurement
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    /**
    Convert radar from polar to cartesian coordinates and initialize state.
    */
    float rho = meas_package.raw_measurements_[0]; //range 
    float phi = meas_package.raw_measurements_[1];  // bearing 
    float rho_dot = meas_package.raw_measurements_[2]; // radial velocity 
    // Conversion from polar to cartesian coordinates
    float x = rho*cos(phi);
    float y = rho*sin(phi);
    float vx = rho_dot*cos(phi);
    float vy = rho_dot*sin(phi);
    float v = sqrt(vx*vx + vy*vy);

    x_ << x,y,v,0,0;  
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    /**
    Initialize state.
    */
    x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;

    if (fabs(x_(0) < 0.0001) and fabs(x_(1) < 0.0001)) {
      x_(0) = 0.0001;
      x_(1) = 0.0001;
    }
  }

  time_us_ = meas_package.timestamp_;
  // done initializing, no need to predict or update
  cout << "UKF Initialisation: " << x_ << endl;
  is_initialized_ = true;
  return;
}

MatrixXd UKF::generate_sigma_points() {
  //create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++)
  {
    Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }

  return Xsig_aug;
}

void UKF::predict_sigma_points(MatrixXd Xsig_aug, double delta_t) {
  //predict sigma points
  for (int i = 0; i< 2*n_aug_+1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
}

void UKF::predict_state_mean() {
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ = x_+ weights_(i) * Xsig_pred_.col(i);
  }
}

void UKF::predict_state_covariance() {
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    x_diff(3) = normalise_angle(x_diff(3));

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }
}

double UKF::normalise_angle(double alpha) {
  while (alpha > M_PI) alpha-=2.*M_PI;
  while (alpha < -M_PI) alpha+=2.*M_PI;
  return alpha; 
}

MatrixXd UKF::state2measurement(MeasurementPackage meas_package) {
  MatrixXd Zsig;
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    int n_z = 3;
    //create matrix for sigma points in measurement space
    Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

      // extract values for better readibility
      double p_x = Xsig_pred_(0,i);
      double p_y = Xsig_pred_(1,i);
      double v  = Xsig_pred_(2,i);
      double yaw = Xsig_pred_(3,i);

      double v1 = cos(yaw)*v;
      double v2 = sin(yaw)*v;

      // measurement model
      Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
      Zsig(1,i) = atan2(p_y,p_x);                                 //phi
      Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
    } 
  }
  else {
    int n_z = 2;
    Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
      double p_x = Xsig_pred_(0,i);
      double p_y = Xsig_pred_(1,i);
      Zsig(0,i) = p_x;                       //p_x
      Zsig(1,i) = p_y;                       //p_y
    }
  }
  return Zsig;
}

VectorXd UKF::get_mean_predicted_measurement(MatrixXd Zsig, int n_z) {
  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }
  return z_pred;
}

MatrixXd UKF::get_measurement_covariance(MatrixXd Zsig, VectorXd z_pred, int n_z, MeasurementPackage meas_package) {
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
      //angle normalization
      z_diff(1) = normalise_angle(z_diff(1));
    }

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    R <<    std_radr_*std_radr_, 0, 0,
            0, std_radphi_*std_radphi_, 0,
            0, 0,std_radrd_*std_radrd_;
  } else {
    R << std_laspx_*std_laspx_, 0, 
         0, std_laspy_*std_laspy_;
  }
  S = S + R;
  return S;
}

MatrixXd UKF::get_cross_correlation(MatrixXd Zsig, VectorXd z_pred, int n_z, MeasurementPackage meas_package) {
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
      //angle normalization
      z_diff(1) = normalise_angle(z_diff(1));
    }

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    x_diff(3) = normalise_angle(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  return Tc;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

    /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    initialise(meas_package);
  }

  float delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  //cout << delta_t << endl;

  double v_current = x_[2];

  double yaw_rate_current = x_[4];

  Prediction(delta_t);

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    UpdateRadar(meas_package);
  }

  if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    UpdateLidar(meas_package);
  }

  double v_new = x_[2];

  double a = (v_new - v_current)/delta_t;

  double yaw_rate_new = x_[4];

  double yaw_acceleration = (yaw_rate_new - yaw_rate_current)/delta_t;

  //cout << a << endl;

  //cout << yaw_acceleration << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

  // Generate Sigma Points
  MatrixXd Xsig_aug = generate_sigma_points();
  //cout << Xsig_aug << endl;

  // Predict Sigma Points
  predict_sigma_points(Xsig_aug, delta_t);
  //cout << Xsig_pred_ << endl;

  // Predict State Mean
  predict_state_mean();
  //cout << x_ << endl;
  //cout << weights_ << endl;

  // Predict State Covariance Matrix
  predict_state_covariance();
  //cout << P_ << endl;

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  int n_z = 2;
  // Transform sigma points into measurement space
  MatrixXd Zsig = state2measurement(meas_package);

  VectorXd z_pred = get_mean_predicted_measurement(Zsig,n_z);

  MatrixXd S = get_measurement_covariance(Zsig, z_pred, n_z, meas_package);

  MatrixXd Tc = get_cross_correlation(Zsig, z_pred, n_z, meas_package);

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  VectorXd z = VectorXd(2);
  
  z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1];

  //residual
  VectorXd z_diff = z - z_pred;
  
  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

  int n_z = 3;
  // Transform sigma points into measurement space
  MatrixXd Zsig = state2measurement(meas_package);

  VectorXd z_pred = get_mean_predicted_measurement(Zsig,n_z);

  MatrixXd S = get_measurement_covariance(Zsig, z_pred, n_z, meas_package);

  MatrixXd Tc = get_cross_correlation(Zsig, z_pred, n_z, meas_package);

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  VectorXd z = VectorXd(3);

  z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], meas_package.raw_measurements_[2];

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  z_diff(1) = normalise_angle(z_diff(1));

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
}

