import time
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from gazebo_connection import GazeboConnection

class RobotEnvironment:

    def __init__(self):
        # before initialising the environment
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        
        # stablishes connection with simulator
        self.gazebo = GazeboConnection()

    def check_topic_publishers_connection(self):
        
        rate = rospy.Rate(10) # 10hz
        while(self.cmd_vel_pub.get_num_connections() == 0):
            rospy.loginfo("No subscribers to cmd_vel yet so we wait and try again")
            rate.sleep()
        rospy.loginfo("cmd_vel publisher connected")
        
    def reset_cmd_vel_commands(self):
        # We send an empty null Twist
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.linear.y = 0.0
        cmd_vel.linear.z = 0.0

        cmd_vel.angular.x = 0.0
        cmd_vel.angular.y = 0.0
        cmd_vel.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd_vel)

    def takeoff_sequence(self, seconds_taking_off=1):
        # Before taking off be sure that cmd_vel value there is is null to avoid drifts        
        rospy.loginfo( "Reset sequence start")
        self.reset_cmd_vel_commands()
        time.sleep(seconds_taking_off)
        rospy.loginfo( "Reset sequence completed")


    def take_observation (self):
        pitch_data = None
        while pitch_data is None:
            try:
                raw_pitch_data = rospy.wait_for_message('/curr_pitch', Float32, timeout=5)
                pitch_data = raw_pitch_data.data
            except:
                rospy.loginfo("Current pitch_data not ready yet, retrying for getting pitch_data")
        
        return pitch_data

    # Resets the state of the environment and returns an initial observation.
    def reset(self):
        
        # 1st: resets the simulation to initial values
        self.gazebo.reset_sim()

        # 2nd: Unpauses simulation
        self.gazebo.unpause_sim()

        # 3rd: resets the robot to initial conditions
        self.check_topic_publishers_connection()
        self.takeoff_sequence()

        # 4th: takes an observation of the initial condition of the robot
        pitch_data = self.take_observation()
        observation = [pitch_data,]
        
        # 5th: pauses simulation
        self.gazebo.pause_sim()

        return observation

    def process_data(self, observation):

        done = False
        reward = 1

        pitch_data = observation[0]

        if pitch_data > 0.4:
            done = True
            reward = 0

        return reward, done

    def step(self, action, running_step=0.3):
        # Given the action selected by the learning algorithm,
        # we perform the corresponding movement of the robot
        actions = {
            1: 0.1,
            0: 0,
            -1: -0.1,
        }
        if not action in actions:
            raise Exception("Choose actions -1, 0, or 1")
        
        # 1st, we decide which velocity command corresponds
        cmd_vel = Twist()
        cmd_vel.linear.x = actions[action]

        # Then we send the command to the robot and let it go
        # for running_step seconds
        self.gazebo.unpause_sim()
        self.cmd_vel_pub.publish(cmd_vel)
        time.sleep(running_step)
        pitch_data = self.take_observation()
        self.gazebo.pause_sim()

        observation = [pitch_data,]

        # finally we get an evaluation based on what happened in the sim
        reward, done = self.process_data(observation)

        state = observation
        return state, reward, done, {}

    def run(self):
        self.reset()
        is_running = True
        episode = 1

        action = 0
        rospy.loginfo("Episode %d is starting", episode)
        while is_running:
            state, reward, done, _, = self.step(action) 
            rospy.loginfo("Current pitch is %f", state[0])
            rospy.loginfo("Reward for action is %d", reward)

            if done:
                is_running = False

        rospy.loginfo("Episode %d is completed", episode)

if __name__ == "__main__":
    '''Initializes and cleanup ros node'''
    rospy.init_node('robot_environment_node', anonymous=True)
    try:
        env = RobotEnvironment()
        env.run()
    except KeyboardInterrupt:
        print "Shutting down ROS "
    

