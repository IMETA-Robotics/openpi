import sys
sys.path.append("./")

import numpy as np
import time
from typing import Union
from robot.imeta_y1 import Y1Controller
from camera.orbbec_camera import OrbbecCamera

# setting your camera serial number
CAMERA_SERIALS = {
    # Replace with actual serial number
    'cam_high': 'CH8R554001M',
    'cam_left_wrist': 'CH8G65200Z9',
    'cam_right_wrist': 'CH8G65200Y4',
}

class RealRobotEnv:
    def __init__(self, single_arm: bool, cam_names: list):
        self.single_arm = single_arm
        self.camera_names = cam_names

        self.controllers = {
            "arm":{
                # "left_arm": Y1Controller("left_arm"),
                "right_arm": Y1Controller("right_arm"),
            }
        }

        self.cameras = {
            "images": {
                "cam_high": OrbbecCamera("cam_high"),
                # "cam_left_wrist": OrbbecCamera("cam_left_wrist"),
                "cam_right_wrist": OrbbecCamera("cam_right_wrist"),
            },
        }
        
    def set_up(self, teleop=False):
        # self.controllers["arm"]["left_arm"].set_up("can0", teleop=teleop)
        self.controllers["arm"]["right_arm"].set_up("can1", teleop=teleop)

        self.cameras["images"]["cam_high"].set_up(CAMERA_SERIALS['cam_high'])
        # self.sensors["images"]["cam_left_wrist"].set_up(CAMERA_SERIALS['cam_left_wrist'])
        self.cameras["images"]["cam_right_wrist"].set_up(CAMERA_SERIALS['cam_right_wrist'])
        
        print("set up success!")
    
    def get_observation(self):
        observation = {}
    
        # state
        if "arm" in self.controllers:
            arm_controller = self.controllers["arm"]
            if len(arm_controller) == 1:
               # single arm, default right arm
               right_arm_state = arm_controller["right_arm"].get_state()
               observation["state"] = np.concatenate([right_arm_state["joint_position"], 
                                                    [right_arm_state["gripper"]]])
               print("observation shape: ", observation["state"].shape)

            elif len(arm_controller) == 2:
                # dual arm
                left_arm_state = arm_controller["left_arm"].get_state()
                right_arm_state = arm_controller["right_arm"].get_state()
                
                # 合并左臂的关节位置和夹爪
                left_joint_and_gripper = np.concatenate([left_arm_state["joint_position"], 
                                                        [left_arm_state["gripper"]]])
                # 合并右臂的关节位置和夹爪
                right_joint_and_gripper = np.concatenate([right_arm_state["joint_position"], 
                                                        [right_arm_state["gripper"]]])
                # 连接双臂的数据
                observation["state"] = np.concatenate([left_joint_and_gripper, 
                                                        right_joint_and_gripper])
                print("observation shape: ", observation["state"].shape)
            else:
                raise RuntimeError(f"arm controller size is {len(arm_controller)}")
                           
        else:
            raise RuntimeError("Not find arm controller!")
                                           
        # image
        camere_images = self.cameras["images"]
        images = {}
        for cam_name in self.camera_names:
            if cam_name not in camere_images:
                raise RuntimeError(f"Not find camera {cam_name}!")
        
            image = camere_images[cam_name].get_image()
            if image is None:
                print(f"not receive {cam_name} image data")
                return None
            
            images[cam_name] = np.transpose(image, (2, 0, 1))
        
        observation["images"] = images
        
        return observation
    
    def step(self, action: Union[list, np.ndarray]):
        if self.single_arm:
            assert len(action) >= 7
            
            # single arm, default right arm
            right_arm_controller = self.controllers["arm"]["right_arm"]
            right_arm_controller.set_joint_position(action[0:6])
            right_arm_controller.set_gripper(action[6])

        else:
            assert len(action) >= 14

            # action[0:6]  -> left arm control
            left_arm_controller = self.controllers["arm"]["left_arm"]
            left_arm_controller.set_joint_position(action[0:6])
            left_arm_controller.set_gripper(action[6])

            # action[7:13] -> right arm control
            right_arm_controller = self.controllers["arm"]["right_arm"]
            right_arm_controller.set_joint_position(action[7:13])
            right_arm_controller.set_gripper(action[13])

if __name__ == "__main__":
    env = RealRobotEnv(single_arm=True, cam_names=["cam_high", "cam_right_wrist"])
    env.set_up()

    while True:
      obs = env.get_observation()
      print(f"observation : {obs}")
      time.sleep(1.0 / 30)

      