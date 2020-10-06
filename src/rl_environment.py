#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys, time
import rospy
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32
from std_srvs.srv import Empty, EmptyRequest
import tf

imu_topic = "/imu"

class RLEnvironment:
    def __init__(self):
        self.imu_subscriber = rospy.Subscriber(imu_topic,Imu,self.callback)

    def callback(self,data):
        quaternion = (data.orientation.x, data.orientation.y, data.orientation.z, data.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        roll, pitch, yaw = euler

        pause_physics_client=rospy.ServiceProxy('/gazebo/pause_physics',Empty)
        pause_physics_client(EmptyRequest())
        
        rospy.loginfo("roll: %.4f pitch: %.4f yaw: %.4f", roll, pitch, yaw)
        
        if pitch > 0.4:
            reset_physics_client=rospy.ServiceProxy('/gazebo/reset_simulation',Empty)
            reset_physics_client(EmptyRequest())
        
        start_physics_client=rospy.ServiceProxy('/gazebo/unpause_physics',Empty)
        start_physics_client(EmptyRequest())

def main(args):
    '''Initializes and cleanup ros node'''
    rospy.init_node('RLEnvironment', anonymous=True)
    env = RLEnvironment()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS "
    

if __name__ == '__main__':
    main(sys.argv)
