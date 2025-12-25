from y1_msg.msg import ArmJointState
from y1_msg.msg import ArmJointPositionControl
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np
import torch
from typing import Union
import rclpy
from rclpy.node import Node

class RealRobotEnv(Node):
  def __init__(self, single_arm: bool, cam_names: list):
    # 确保rclpy已初始化
    if not rclpy.ok():
      rclpy.init()
    
    super().__init__('eval_real_robot')
    self.single_arm = single_arm
    self.camera_names = cam_names

    self.bridge = CvBridge()
    self.right_puppet_arm_state = None
    self.left_puppet_arm_state = None
    self.img_dict = {}
    self.left_arm_joint_position_control_pub_ = None
    self.right_arm_joint_position_control_pub_ = None
    # 初始化订阅
    self.init_subscriptions()

  def destroy(self):
    self.destroy_node()
    rclpy.shutdown()
    
  def init_subscriptions(self):
    # subscribe
    # robotic arm data
    if self.single_arm:
      # one arm, default right arm
      self.create_subscription(
            ArmJointState,
            "/puppet_arm_right/joint_states",
            self.puppet_arm_right_callback,
            1)
      
      # control right arm
      self.right_arm_joint_position_control_pub_ = self.create_publisher(
            ArmJointPositionControl,
            '/master_arm_right/joint_states',
            1)
      
    else:
      # dual arm
      self.create_subscription(
            ArmJointState,
            "/puppet_arm_right/joint_states",
            self.puppet_arm_right_callback,
            1)
      
      self.create_subscription(
          ArmJointState,
          "/puppet_arm_left/joint_states",
          self.puppet_arm_left_callback,
          1)
      
      # control left and right arm
      self.left_arm_joint_position_control_pub_ = self.create_publisher(
          ArmJointPositionControl,
          '/master_arm_left/joint_states',
          1)
      self.right_arm_joint_position_control_pub_ = self.create_publisher(
          ArmJointPositionControl,
          '/master_arm_right/joint_states',
          1)  
  
    # subscribe camera rgb data
    for cam_name in self.camera_names:
      if cam_name == "cam_right_wrist":
        # right arm wrist camera rgb image
        self.create_subscription(
            Image, "/camera_right/color/image_raw", self.img_right_callback, 1)
      elif cam_name == "cam_left_wrist":
        # left arm wrist camera rgb image
        self.create_subscription(
            Image, "/camera_left/color/image_raw", self.img_left_callback, 1)
      elif cam_name == "cam_high":
        # high camera rgb image
        self.create_subscription(
            Image, "/camera_high/color/image_raw", self.img_high_callback, 1)
      elif cam_name == "cam_front":
        # front camera rgb image
        self.create_subscription(
            Image, "/camera_front/color/image_raw", self.img_front_callback, 1)
      else:
        raise Exception(f"camera name {cam_name} not found")

  def puppet_arm_right_callback(self, msg: ArmJointState):
    """right arm"""
    self.right_puppet_arm_state = msg 
    
  def puppet_arm_left_callback(self, msg: ArmJointState):
    """left arm"""
    self.left_puppet_arm_state = msg 
    
  def img_right_callback(self, msg: Image):
    """right arm wrist camera rgb image"""
    self.img_dict["cam_right_wrist"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    
  def img_left_callback(self, msg: Image):
    """left arm wrist camera rgb image"""
    self.img_dict["cam_left_wrist"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    
  def img_high_callback(self, msg: Image):
    """high camera rgb image"""
    self.img_dict["cam_high"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

  def img_front_callback(self, msg: Image):
    """front camera rgb image"""
    self.img_dict["cam_front"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    
  def get_observation(self):
    observation = {}

    # state
    if self.single_arm:
      # single arm
      if self.right_puppet_arm_state is None:
        print("not receive right arm data")
        return None
      else:
        joint_state = np.array(self.right_puppet_arm_state.joint_position)
        observation["state"] = joint_state
    else:
      # double arm
      if self.right_puppet_arm_state is None:
        print("not receive right arm data")
        return None

      if self.left_puppet_arm_state is None:
        print("not receive left arm data")
        return None
      
      observation["state"] = np.concatenate([self.right_puppet_arm_state.joint_position,
                                 self.left_puppet_arm_state.joint_position])
    
    # image
    images = {}
    for cam_name in self.camera_names:
      if cam_name not in  self.img_dict:
        print(f"not receive {cam_name} image data")
        return None
      images[cam_name] = np.transpose(self.img_dict[cam_name], (2, 0, 1))
      
    observation["images"] = images
    
    return observation
    
  def step(self, action: Union[list, np.ndarray, torch.Tensor]):
    action = action.tolist()
    if self.single_arm:
      assert len(action) >= 7

      # single arm, default right arm
      joint_control_msg = ArmJointPositionControl()
      joint_control_msg.header.stamp = self.get_clock().now().to_msg()
      joint_control_msg.joint_position = action[0:6]
      joint_control_msg.joint_velocity = 3
      joint_control_msg.gripper_stroke = action[6]
      joint_control_msg.gripper_velocity = 3
      self.right_arm_joint_position_control_pub_.publish(joint_control_msg)

    else:
      assert len(action) >= 14

      # action[0:6]  -> left arm control
      left_arm_control_msg = ArmJointPositionControl()
      left_arm_control_msg.header.stamp = self.get_clock().now().to_msg()
      left_arm_control_msg.joint_position = action[0:6]
      left_arm_control_msg.joint_velocity = 3
      left_arm_control_msg.gripper_stroke = action[6]
      left_arm_control_msg.gripper_velocity = 3
      self.left_arm_joint_position_control_pub_.publish(left_arm_control_msg)

      # action[7:13] -> right arm control
      right_arm_control_msg = ArmJointPositionControl()
      right_arm_control_msg.header.stamp = self.get_clock().now().to_msg()
      right_arm_control_msg.joint_position = action[7:13]
      right_arm_control_msg.joint_velocity = 3
      right_arm_control_msg.gripper_stroke = action[13]
      right_arm_control_msg.gripper_velocity = 3
      self.right_arm_joint_position_control_pub_.publish(right_arm_control_msg)
