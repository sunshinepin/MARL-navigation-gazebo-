import math
import os
import random
import subprocess
import time
from os import path

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
# from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState  # Make sure to import ModelState


GOAL_REACHED_DIST = 0.7
COLLISION_DIST = 0.35
TIME_DELTA = 0.1

# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y):
    goal_ok = True

    if -2.3 > x > -3.7 and 4.7 > y > 3.3:
        goal_ok = False

    if -2.3 > x > -3.7 and -3.3 > y > -4.7:
        goal_ok = False

    if -8.3 > x > -9.7 and 4.7 > y > 3.3:
        goal_ok = False

    if -8.3 > x > -9.7 and -3.3 > y > -4.7:
        goal_ok = False

    if 6.7 > x > 5.3 and 0.7 > y > -0.7:
        goal_ok = False

    if -5.3 > x > -6.7 and 0.7 > y > -0.7:
        goal_ok = False

    if 3.7 > x > 2.3 and 4.7 > y > 3.3:
        goal_ok = False

    if 3.7 > x > 2.3 and -3.3 > y > -4.7:
        goal_ok = False

    if 9.7 > x > 8.3 and 4.7 > y > 3.3:
        goal_ok = False

    if 9.7 > x > 8.3 and -3.3 > y > -4.7:
        goal_ok = False

    if 0.7 > x > -0.7 and 0.7 > y > -0.7:
        goal_ok = False

    if x > 13.0 or x < -13.0 or y > 7.0 or y < -7.0:
        goal_ok = False

    return goal_ok


class GazeboEnv:
    """Superclass for all Gazebo environments."""
    def __init__(self, environment_dim):
        self.environment_dim = environment_dim
        # Initialize for multiple robots
        self.num_robots = 2  # Assuming two robots
        self.robot_state_dims = 24
        self.robot_action_dims = 2
        # The observation for each robot
        self.robot_observations = [np.zeros(self.robot_state_dims) for _ in range(self.num_robots)]

        rospy.init_node("gym", anonymous=True)
        print("Roscore launched!")

        # Setup Publishers
        self.vel_pub_car1 = self.setup_publisher("/car1/ackermann_steering_controller/cmd_vel", Twist, 1)
        self.vel_pub_car2 = self.setup_publisher("/car2/ackermann_steering_controller/cmd_vel", Twist, 1)
        # self.set_state_pub = self.setup_publisher("gazebo/set_model_state", ModelState, 10)
        self.goal_pub = self.setup_publisher("goal_point", MarkerArray, 3)
        self.lin_vel_pub = self.setup_publisher("linear_velocity", MarkerArray, 1)
        self.ang_vel_pub = self.setup_publisher("angular_velocity", MarkerArray, 1)
        # Setup Publishers and Subscribers as previously defined
        self.set_state_pub = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10)

        # Setup Subscribers
        self.velodyne_sub_car1 = self.setup_subscriber("/car1/velodyne/velodyne_points", PointCloud2, self.velodyne_callback, 1)
        self.odom_sub_car1 = self.setup_subscriber("/car1/odom_gazebo", Odometry, self.odom_callback, 1)
        self.velodyne_sub_car2 = self.setup_subscriber("/car2/velodyne/velodyne_points", PointCloud2, self.velodyne_callback, 1)
        self.odom_sub_car2 = self.setup_subscriber("/car2/odom_gazebo", Odometry, self.odom_callback, 1)

        # Setup Services
        self.unpause_srv = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause_srv = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_world_srv = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        # Initialize the service proxy for resetting Gazebo
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        # Initialize ModelState for setting the state of objects in Gazebo
        self.set_self_state = ModelState()

    def setup_publisher(self, topic, data_type, queue_size):
        return rospy.Publisher(topic, data_type, queue_size=queue_size)

    def setup_subscriber(self, topic, data_type, callback, queue_size):
        return rospy.Subscriber(topic, data_type, callback, queue_size=queue_size)


    def velodyne_callback(self, v):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data = np.ones(self.environment_dim) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                        break

    def odom_callback(self, od_data):
        self.last_odom = od_data
def step(self, actions):
    # 确保 actions 是一个由元组组成的列表 [(lin_vel1, ang_vel1), (lin_vel2, ang_vel2)]
    states, rewards, dones, targets = [], [], [], []
    for i, action in enumerate(actions):
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.publish_vel(i, vel_cmd)  # 自定义方法来处理不同的机器人

        self.simulate_motion()  # 通用方法来取消暂停，等待，然后暂停模拟

        state, reward, done, target = self.process_robot_state(i)  # 在运动后处理每个机器人的状态
        states.append(state)
        rewards.append(reward)
        dones.append(done)
        targets.append(target)

    return states, rewards, dones, targets

def reset(self):
    rospy.wait_for_service("/gazebo/reset_world")
    try:
        self.reset_proxy()
    except rospy.ServiceException as e:
        print("/gazebo/reset_world service call failed: %s" % e)
    self.initialize_robots()  # 自定义方法来放置机器人在初始位置

    return [self.get_robot_state(i) for i in range(self.num_robots)]  # 获取每个机器人的初始状态

def simulate_motion(self):
    rospy.wait_for_service("/gazebo/unpause_physics")
    try:
        self.unpause_srv()
    except rospy.ServiceException as e:
        print("Error unpausing Gazebo: %s" % e)
    time.sleep(TIME_DELTA)
    rospy.wait_for_service("/gazebo/pause_physics")
    try:
        self.pause_srv()
    except rospy.ServiceException as e:
        print("Error pausing Gazebo: %s" % e)

def process_robot_state(self, robot_index):
    angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_self_state

        x = 0
        y = 0
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            position_ok = check_pos(x, y)
        # 下面通过 /gazebo/set_model_state 话题更改移动机器人的初始位置
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        object_state.pose.position.z = 0.3
        # 应用朝向
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        # 发布新的模型状态
        self.set_state.publish(object_state)
        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y

        # 更改一个目标点
        # set a random goal in empty space in environment
        self.change_goal()
        # randomly scatter boxes in the environment# 更改环境中的小方块障碍物
        self.random_box()
        self.publish_markers([0.0, 0.0])
        # 开始仿真
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")
        # 执行一段时间
        time.sleep(TIME_DELTA)
        # 停止仿真，获得目前的状态
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [distance, theta, 0.0, 0.0]
        state = np.append(laser_state, robot_state)
        # 返回状态值
    pass
    

    def reset_robot(self, robot_index):
        # Reset logic for each robot
        return np.zeros(self.robot_state_dims) 


    def change_goal(self):
        # Place a new goal and check if its location is not on one of the obstacles
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        goal_ok = False

        while not goal_ok:
            self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
            self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
            goal_ok = check_pos(self.goal_x, self.goal_y)

    def random_box(self):
        # Randomly change the location of the boxes in the environment on each reset to randomize the training
        # environment
        for i in range(4):
            name = "cardboard_box_" + str(i)

            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y)
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < 1.5 or distance_to_goal < 1.5:
                    box_ok = False
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

    def publish_markers(self, action):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "world"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "world"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(action[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "world"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(action[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    # 奖励部分
    @staticmethod
    def get_reward(target, collision, action, min_laser):
        if target:
            return 100.0
        elif collision:
            return -100.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2
