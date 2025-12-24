from y1_msg.msg import ArmJointState
from y1_msg.msg import ArmJointPositionControl
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rospy
import numpy as np
from typing import Union

class RealRobotEnv:
  def __init__(self, single_arm: bool, cam_names: list):
    self.single_arm = single_arm
    self.camera_names = cam_names

    self.bridge = CvBridge()
    self.right_puppet_arm_state = None
    self.left_puppet_arm_state = None
    self.img_dict = {}
    self.left_arm_joint_position_control_pub_ = None
    self.right_arm_joint_position_control_pub_ = None
    self.init_topic()
    
  def init_topic(self):
    rospy.init_node("eval_real_robot")
    
    # subscribe robotic arm data
    # right arm
    rospy.Subscriber("/puppet_arm_right/joint_states",
          ArmJointState, self.puppet_arm_right_callback, queue_size=1, tcp_nodelay=True)
    # left arm
    rospy.Subscriber("/puppet_arm_left/joint_states",
          ArmJointState, self.puppet_arm_left_callback, queue_size=1, tcp_nodelay=True)
    
    # control left and right arm
    self.left_arm_joint_position_control_pub_ = rospy.Publisher('/master_arm_left/joint_states', 
                                                                ArmJointPositionControl, queue_size=1)
    self.right_arm_joint_position_control_pub_ = rospy.Publisher('/master_arm_right/joint_states', 
                                                                  ArmJointPositionControl, queue_size=1)    
  
    # subscribe camera rgb data
    # right arm wrist camera rgb image
    rospy.Subscriber("/camera_right/color/image_raw", 
      Image, self.img_right_callback, queue_size=1, tcp_nodelay=True)

    # left arm wrist camera rgb image
    rospy.Subscriber("/camera_left/color/image_raw", 
      Image, self.img_left_callback, queue_size=1, tcp_nodelay=True)

    # front camera rgb image
    rospy.Subscriber("/camera_high/color/image_raw", 
      Image, self.img_high_callback, queue_size=1, tcp_nodelay=True)

  def puppet_arm_right_callback(self, msg: ArmJointState):
    """right arm"""
    self.right_puppet_arm_state = msg 
    
  def puppet_arm_left_callback(self, msg: ArmJointState):
    """left arm"""
    self.left_puppet_arm_state = msg 
    
  def img_right_callback(self, msg: Image):
    """right arm wrist camera rgb image"""
    self.img_dict["cam_right_wrist"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    # print("cam_right_wrist image shape: ", self.img_dict["cam_right_wrist"].shape)
    
  def img_left_callback(self, msg: Image):
    """left arm wrist camera rgb image"""
    self.img_dict["cam_left_wrist"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    # print("cam_left_wrist image shape: ", self.img_dict["cam_left_wrist"].shape)
    
  def img_high_callback(self, msg: Image):
    """high camera rgb image"""
    self.img_dict["cam_high"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    # print("cam_high image shape: ", self.img_dict["cam_high"].shape)
    
  def get_observation(self):
    observation = {}

    # state
    if self.single_arm:
      # default right arm
      if self.right_puppet_arm_state is None:
        print("not receive right arm data")
        return None
      else:
        observation["state"] = self.right_puppet_arm_state.joint_position
    else:
      # dual arm
      if self.left_puppet_arm_state is None:
        print("not receive left arm data")
        return None
      
      if self.right_puppet_arm_state is None:
        print("not receive right arm data")
        return None
      
      observation["state"] = np.concatenate([self.left_puppet_arm_state.joint_position,
                                self.right_puppet_arm_state.joint_position])
                                           
    # image
    images = {}
    for cam_name in self.camera_names:
      if cam_name not in self.img_dict:
        print(f"not receive {cam_name} image data")
        return None
      images[cam_name] = np.transpose(self.img_dict[cam_name], (2, 0, 1))
      
    observation["images"] = images
    
    return observation
    
  def step(self, action: Union[list, np.ndarray]):
    if self.single_arm:
      assert len(action) >= 7
      
      # single arm, default right arm
      joint_control_msg = ArmJointPositionControl()
      joint_control_msg.header.stamp = rospy.Time.now()
      joint_control_msg.joint_position = action[0:6]
      joint_control_msg.joint_velocity = 3
      joint_control_msg.gripper_stroke = action[6]
      joint_control_msg.gripper_velocity = 3
      self.right_arm_joint_position_control_pub_.publish(joint_control_msg)

    else:
      assert len(action) >= 14

      # action[0:6]  -> left arm control
      left_arm_control_msg = ArmJointPositionControl()
      left_arm_control_msg.header.stamp = rospy.Time.now()
      left_arm_control_msg.joint_position = action[0:6]
      left_arm_control_msg.joint_velocity = 3
      left_arm_control_msg.gripper_stroke = action[6]
      left_arm_control_msg.gripper_velocity = 3
      self.left_arm_joint_position_control_pub_.publish(left_arm_control_msg)

      # action[7:13] -> right arm control
      right_arm_control_msg = ArmJointPositionControl()
      right_arm_control_msg.header.stamp = rospy.Time.now()
      right_arm_control_msg.joint_position = action[7:13]
      right_arm_control_msg.joint_velocity = 3
      right_arm_control_msg.gripper_stroke = action[13]
      right_arm_control_msg.gripper_velocity = 3
      self.right_arm_joint_position_control_pub_.publish(right_arm_control_msg)

    # TODO: add mobile_base control


if __name__ == "__main__":
    env = RealRobotEnv()

    while True:
      continue