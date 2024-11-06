#include <ros/ros.h>
#include <eigen3/Eigen/Core>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>

#include <iostream>
#include <cmath>
#include <math.h>
#include <algorithm>

ros::Publisher *command_pub;

double s = 0.0; // A counter

// Keeps the angle within [-pi, pi]
static double smallestDeltaAngle(const double& x, const double& y) {
   return atan2(sin(x-y), cos(x-y));
}


void poseCallback(const geometry_msgs::Pose::ConstPtr& msg) {

// Pose message is divided in Point position and Quaternion orientation

  geometry_msgs::Twist tw;

  // Final destination
  double x_f = 1.0;  // Final x value
  double y_f = 1.0;  // Final y value

  double k = 5.0;        // Free parameter
  double theta_f = 0.0;  // Final theta value

  // Current state
  double x_p = msg->position.x;          // Current x value
  double y_p = msg->position.y;         // Current y value
  double theta_p = msg->orientation.w;  // Current orientation

  double a_x = k*cos(theta_f)-3*x_f; // alpha_x
  double b_x = k*cos(theta_p)+3*x_p; // beta_x

  double a_y = k*sin(theta_f)-3*y_f; // alpha_y
  double b_y = k*sin(theta_p)+3*y_p; // beta_y


  // First derivative of the cubic polynomials
  double x_s_1 = 3*s*s*x_f-3*(s-1)*(s-1)*x_p+a_x*s*s+2*a_x*s*(s-1)+b_x*(s-1)*(s-1)+2*b_x*s*(s-1);
  double y_s_1 = 3*s*s*y_f-3*(s-1)*(s-1)*y_p+a_y*s*s+2*a_y*s*(s-1)+b_y*(s-1)*(s-1)+2*b_y*s*(s-1);

  // Second derivative of the cubic polynomials
  double x_s_2 = 6*s*x_f-6*(s-1)*x_p+4*a_x*s+2*a_x*(s-1)+4*b_x*(s-1)+2*b_x*s;
  double y_s_2 = 6*s*y_f-6*(s-1)*y_p+4*a_y*s+2*a_y*(s-1)+4*b_y*(s-1)+2*b_y*s;

  // The geometrical inputs that drive the robot along the Cartesian path
  double lin_x = sqrt(x_s_1*x_s_1+y_s_1*y_s_1);
  double lin_y = (y_s_2*x_s_1-x_s_2*y_s_1)/(x_s_1*x_s_1+y_s_1*y_s_1);

  // Implementing a PD controller
  // Kp*e(t)+Kd*e_dot(t) = u
  double K_p = 5.0;
  double K_d = 4.0;

  double theta = smallestDeltaAngle(x_s_1, y_s_1);

  // rotational  matrix
  Eigen::MatrixXd rot(3,3);
  rot << cos(theta), sin(theta), 0.0,
  	 - sin(theta), cos(theta), 0.0,
  	 0.0, 0.0, 1.0;

  // The Cartesian error
  Eigen::MatrixXd e(3,1);
  e << x_f - x_p,
       y_f - y_p,
       theta_f - theta;

  // Tracking error
  Eigen::MatrixXd err(3,1);
  err << rot * e;

  // Calculating the tracking error dynamics

  // The reference inputs
  double v_d = 0.25;  // Desired forward velocity
  double w_d = 1.8;   // Desired rotational velocity

  // The input transformation
  double u_1 = v_d * cos(err(2,0)) - lin_x;
  double u_2 = w_d - lin_y;


  Eigen::MatrixXd err_dot(3,1);

  // First term
  Eigen::MatrixXd part_1(3,3);
  part_1 << 0.0, w_d, 0.0,
            -w_d, 0.0, 0.0,
            0.0, 0.0, 0.0;

  // Second term
  Eigen::MatrixXd part_2(3,1);
  part_2 << 0.0, sin(err(2,0)), 0.0;

  // Third term
  Eigen::MatrixXd part_3(3,2);
  part_3 << 1.0, -err(1,0),
            0.0, err(0,0),
            0.0, 1.0;

  Eigen::MatrixXd u(2,1);
  u << u_1, u_2;

  // The tracking error dynamics
  err_dot << part_1 * err + part_2 * v_d + part_3 * u;

  // The PD controller
  Eigen::MatrixXd controller(3,1);
  controller << K_p * err + K_d * err_dot;

  // The robot stops moving if s=1
  if (static_cast<int>(s)==1) {
  tw.linear.x = 0.0;   // The x position
  tw.angular.z = 0.0;  // The orientation
  } else {
  tw.linear.x = controller(0, 0);
  tw.angular.z = controller(2, 0);
  s = s+0.0001;
  }

  command_pub->publish(tw);
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "planner_node");
  ros::NodeHandle nh;

  // subscribing to the /pose topic and calling poseCallback when messages are published on this topic
  ros::Subscriber sub = nh.subscribe("/pose", 1000, poseCallback);
  ros::Publisher pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1000);
  command_pub = &pub;

  ros::spin();

  return 0;


}
