#include <ros/ros.h>
#include <../../home/claudaba/Desktop/ros_oblig2/devel/include/tek4030_visual_servoing_msgs/ImageFeaturePoints.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Point.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <math.h>


// holds the address for the publisher messages
ros::Publisher* camera_twist_pub_global;
ros::Publisher* points_setpoint_global;
ros::Publisher* points_error_global;

// the message here is s (feature vector) that is received when a node publishes on the points_normalized topic
void normalizedCallBack(const tek4030_visual_servoing_msgs::ImageFeaturePoints::ConstPtr& msg)
{
    
    // splitting the message to get X and Y separately for each feature vector s - just so we can easily to refer to
    geometry_msgs::Point s1;
    s1.x = msg->p[0].x;
    s1.y = msg->p[0].y;
    
    geometry_msgs::Point s2;
    s2.x = msg->p[1].x;
    s2.y = msg->p[1].y;
    
    geometry_msgs::Point s3;
    s3.x = msg->p[2].x;
    s3.y = msg->p[2].y;
    
    geometry_msgs::Point s4;
    s4.x = msg->p[3].x;
    s4.y = msg->p[3].y;
    
    
    // declaring the setpoint 
    tek4030_visual_servoing_msgs::ImageFeaturePoints sd;
    geometry_msgs::Point sd1;
    sd1.x = 0.15;
    sd1.y = 0.15;
    
    geometry_msgs::Point sd2;
    sd2.x = -0.15;
    sd2.y = 0.15;
    
    geometry_msgs::Point sd3;
    sd3.x = -0.15;
    sd3.y = -0.15;
    
    geometry_msgs::Point sd4;
    sd4.x = 0.15;
    sd4.y = -0.15;
   
    sd.p.push_back(sd1);
    sd.p.push_back(sd2);
    sd.p.push_back(sd3);
    sd.p.push_back(sd4);
    
    
    // calculating the error:
    tek4030_visual_servoing_msgs::ImageFeaturePoints es;
    geometry_msgs::Point es1;
    es1.x = msg->p[0].x - sd1.x;
    es1.y = msg->p[0].y - sd1.y;
    
    geometry_msgs::Point es2;
    es2.x = msg->p[1].x - sd2.x;
    es2.y = msg->p[1].y - sd2.y;
    
    geometry_msgs::Point es3;
    es3.x = msg->p[2].x - sd3.x;
    es3.y = msg->p[2].y - sd3.y;
    
    geometry_msgs::Point es4;
    es4.x = msg->p[3].x - sd4.x;
    es4.y = msg->p[3].y - sd4.y;
    

    es.p.push_back(es1);
    es.p.push_back(es2);
    es.p.push_back(es3);
    es.p.push_back(es4);
    
    
    // declaring the desired movement for the camera according to the control law
    geometry_msgs::Twist v_r_c;
    
    double zc = 1.0;
    
    Eigen::MatrixXd Ls(8,6); // interaction matrix
    
    Eigen::MatrixXd Ks(8,8); // Ks (assuming it's the identity matrix)
    Ks << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
          
    // defining the interaction matrix
    Ls << -1/zc, 0.0, s1.x/zc, s1.x*s1.y, -(1.0+(s1.x*s1.x)), s1.y,
          0.0, -1/zc, s1.y/zc, 1.0+(s1.y*s1.y), -s1.x*s1.y, -s1.x,
          -1/zc, 0.0, s2.x/zc, s2.x*s2.y, -(1.0+(s2.x*s2.x)), s2.y,
          0.0, -1/zc, s2.y/zc, 1.0+(s2.y*s2.y), -s2.x*s2.y, -s2.x, 
          -1/zc, 0.0, s3.x/zc, s3.x*s3.y, -(1.0+(s3.x*s3.x)), s3.y,
          0.0, -1/zc, s3.y/zc, 1.0+(s3.y*s3.y), -s3.x*s3.y, -s3.x,       
          -1/zc, 0.0, s4.x/zc, s4.x*s4.y, -(1.0+(s4.x*s4.x)), s4.y,
          0.0, -1/zc, s4.y/zc, 1.0+(s4.y*s4.y), -s4.x*s4.y, -s4.x;
           
    
    // making a vector from the error
    Eigen::MatrixXd es_vector(8,1);
    
    es_vector << es1.x, es1.y, es2.x, es2.y, es3.x, es3.y, es4.x, es4.y;
        
    // calculating v_r_c by using the specified formula
    Eigen::MatrixXd v_r_c_vector(6,1);
    v_r_c_vector<<(Ls.transpose()*Ls).inverse()*(Ls.transpose())*Ks*es_vector;
        
    //
    v_r_c.linear.x = -v_r_c_vector(0,0);
    v_r_c.linear.y = v_r_c_vector(1,0);
    v_r_c.linear.z = v_r_c_vector(2,0);
        
    v_r_c.angular.x = -v_r_c_vector(3,0);
    v_r_c.angular.y = v_r_c_vector(4,0);
    v_r_c.angular.z = v_r_c_vector(5,0);

    
    // publishing everything
    camera_twist_pub_global->publish(v_r_c);
    points_setpoint_global->publish(sd);
    points_error_global->publish(es);
}


int main(int argc, char **argv) 
{

    ros::init(argc, argv, "node");
    ros::NodeHandle nh;
    
    // adding the topics that the node should publish to
    ros::Publisher camera_twist_pub = nh.advertise<geometry_msgs::Twist>("camera_twist", 1000);
    
    ros::Publisher points_setpoint_pub = nh.advertise<tek4030_visual_servoing_msgs::ImageFeaturePoints>("/imgproc/points_setpoint", 1000);

    ros::Publisher points_error_pub = nh.advertise<tek4030_visual_servoing_msgs::ImageFeaturePoints>("/imgproc/points_error", 1000);
    
    // setting the value to the global variables to the addresses of the publishers
    camera_twist_pub_global = &camera_twist_pub;
    points_setpoint_global = &points_setpoint_pub;
    points_error_global = &points_error_pub;

    // adding the topics that this node should subscribe to -whenever a node publishes in this node, the normalizedCallBack function is called
    ros::Subscriber points_normalized_sub = nh.subscribe("/imgproc/points_normalized", 1000, normalizedCallBack);
    
    ros::spin(); // node waiting for incoming topics or services
    return 0;
}
