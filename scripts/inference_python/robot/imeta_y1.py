'''
code reference from:
https://github.com/IMETA-Robotics/y1_sdk_python/blob/noetic/y1_sdk/example/single_arm_control.py
'''
import sys
sys.path.append("./")

from y1_sdk import Y1SDKInterface, ControlMode
import numpy as np
import time
import os
# 获取当前脚本文件所在目录
assets_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/assets"

class Y1Controller():
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.controller = None
    
    def set_up(self, can:str, arm_end_type=3, teleop=False):
        if arm_end_type == 0:
            urdf_path = f"{assets_path}/imeta_y1/y1_no_gripper.urdf"
        elif arm_end_type == 1:
            urdf_path = f"{assets_path}/imeta_y1/y1_with_gripper.urdf"
        elif arm_end_type == 2:
            urdf_path = f"{assets_path}/imeta_y1/y1_with_gripper.urdf"
        elif arm_end_type == 3:
            urdf_path = f"{assets_path}/imeta_y1/y1_with_gripper.urdf"

        self.controller = Y1SDKInterface(
            can_id=can,
            urdf_path=urdf_path,
            arm_end_type=arm_end_type,
            enable_arm=True,
        )

        if not self.controller.Init():
            print(f"{self.name} initialization failed!")
            exit()
        if teleop:
            self.controller.SetArmControlMode(ControlMode.GRAVITY_COMPENSATION)
        else:
            self.controller.SetArmControlMode(ControlMode.NRT_JOINT_POSITION)
        time.sleep(1)

    def get_state(self):
        state = {}
        
        eef = self.controller.GetArmEndPose()
        joint_position = self.controller.GetJointPosition()
        # vel = self.controller.GetJointVelocity()

        state["end_pose"] = eef
        state["joint_position"] = joint_position[:6]
        state["gripper"] = joint_position[6]

        return state

    def set_end_pose(self, end_pose):
        self.controller.SetArmEndPose(list(end_pose))
    
    def set_joint_position(self, joint_position, velocity: int = 3):
        self.controller.SetArmJointPosition(list(joint_position), velocity)

    # The input gripper value is in the range [0, 1], representing the degree of opening.
    def set_gripper(self, gripper, velocity: int = 3):
        # gripper = gripper * 84
        self.controller.SetGripperStroke(gripper, velocity)

if __name__=="__main__":
    # left_controller = Y1Controller("left_arm")
    right_controller = Y1Controller("right_arm")
    # left_controller.set_up("can0", 3, False)
    right_controller.set_up("can1", 3, False)

    input("Press key [enter] to start control robot arm!")
    # left_controller.set_joint(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
    right_controller.set_joint_position(np.array([0.6, -0.6, 0.6, 0.5, 0.4, 0]))
    right_controller.set_gripper(80)
    time.sleep(2)
    print(right_controller.get_state())
    # left_controller.set_joint(np.array([0.1,0.1,-0.2,0.3,-0.2,0.5]))
    right_controller.set_joint_position(np.array([0, 0, 0, 0, 0, 0]))
    right_controller.set_gripper(0)
    time.sleep(2)
    print(right_controller.get_state())

    right_controller.set_end_pose(np.array([0.05, -0.04, 0.4, 0.2, -0.5, -1]))
    time.sleep(2)

    print(right_controller.get_state())
    right_controller.set_joint_position(np.array([0, 0, 0, 0, 0, 0]))
    right_controller.set_gripper(0)
    
    while True:
        try:
            time.sleep(0.1)
        except KeyboardInterrupt:
            break