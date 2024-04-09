import math
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
import random
import time
 
GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.35
TIME_DELTA = 0.1

class SimplifiedGazeboEnv:
    """Simplified superclass for Gazebo environments with fixed goal positions."""

    def __init__(self, launchfile, environment_dim):
        self.environment_dim = environment_dim
        self.odom_x = 0
        self.odom_y = 0

        # Fixed goal positions
        self.goal_positions = [(-10, 0), (0, 5), (10, 0)]
        self.goal_x, self.goal_y = random.choice(self.goal_positions)

        self.velodyne_data = np.ones(self.environment_dim) * 10
        self.last_odom = None
        self.gaps = self.initialize_gaps()  # 假设你添加了一个方法来初始化 gaps
        self.set_self_state = ModelState()
        self.set_self_state.model_name = "car1"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        rospy.init_node("gym", anonymous=True)

        # ROS publishers and subscribers
        self.vel_pub = rospy.Publisher("/car1/ackermann_steering/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher("/gazebo/model_state", ModelState, queue_size=10)

        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)

        self.velodyne = rospy.Subscriber("/car1/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1)
        self.odom = rospy.Subscriber("/car1/ackermann_steering/odom", Odometry, self.odom_callback, queue_size=1)


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

    def reset(self):
        # Resets the state of the environment to an initial state
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        # Randomly select a new goal position
        self.goal_x, self.goal_y = random.choice(self.goal_positions)
        # No need to place random boxes as there are no random obstacles

        # Your existing reset code here, modified to not place obstacles

        return self._get_state()

    def step(self, action):
        target = False
        # 发布机器人的动作，这里的step输入是action，action[0]是线速度，action[1]是角速度
        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)
        # 首先unpause，发布消息后，开始仿真
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")
        # 接下来需要停一小段时间，让小车以目前的速度行驶一段时间，TIME_DELTA = 0.1
        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)
        # 然后我们停止仿真，开始观测目前的状态
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # read velodyne laser state# 检测碰撞
        done, collision, min_laser = self.observe_collision(self.velodyne_data)
         # 雷达状态
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]
        # 从里程计信息中获取小车朝向
        # Calculate robot heading from odometry data
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)
        # 计算到目标点的距离
        # Calculate distance to the goal from the robot
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )
        # 计算偏离角度
        # Calculate the relative angle between the robots heading and heading toward the goal
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
        # 判断是否到达了目标点
        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            target = True
            done = True
        # 输出机器人状态
        robot_state = [distance, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)
        # 计算奖励  
        reward = self.get_reward(target, collision, action, min_laser)
         # 返回状态、奖励、是否结束、是否到达目标点
        return state, reward, done, target


    def _get_state(self):
        # 你可能需要根据你的实际情况调整这个方法来获取当前的激光雷达数据和机器人状态
        laser_state = self.velodyne_data  # 假设已经是20维的数据
        robot_state = np.array([self.calculate_distance_to_goal(), self.calculate_angle_difference_to_goal(), self.current_linear_velocity, self.current_angular_velocity])
        state = np.concatenate((laser_state, robot_state), axis=None)
        return state
    def get_reward(self, target, collision, action, min_laser):
        # 简单的奖励计算：到达目标给予正奖励，碰撞给予负奖励
        if target:
            return 100  # 到达目标的奖励
        if collision:
            return -100  # 碰撞的惩罚
        # 可以添加更多基于动作或当前状态的奖励逻辑
        # 例如，鼓励机器人保持一定的速度或者接近目标
        return -1  # 默认的步骤消耗，鼓励更快到达目标
    def observe_collision(self, laser_data):
        collision = np.any(laser_data < COLLISION_DIST)  # 假设碰撞发生在任何激光雷达读数小于设定的碰撞距离
        min_laser = np.min(laser_data)  # 获取激光雷达数据中的最小值，可能用于其他计算
        done = collision  # 如果发生碰撞，认为当前episode结束
        return done, collision, min_laser
    def initialize_gaps(self):
        num_segments = 20
        gap_width = 180.0 / num_segments  # 视场宽度除以段数得到每个段的宽度
        return [(-90 + i * gap_width, -90 + (i + 1) * gap_width) for i in range(num_segments)]

# Example usage
if __name__ == "__main__":
    # Assuming a predefined launchfile and environment dimension
    env = SimplifiedGazeboEnv("your_launchfile.launch", 180)
    state = env.reset()
    done = False
    while not done:
        action = [0.5, 0]  # Example action
        state, reward, done, _ = env.step(action)
        print(f"State: {state}, Reward: {reward}, Done: {done}")
