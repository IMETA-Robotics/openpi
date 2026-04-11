import sys
sys.path.append("./")

import cv2
import numpy as np
import rerun as rr
import threading
import time
from typing import Union
from robot.imeta_y1 import Y1Controller
from camera.orbbec_camera import OrbbecCamera
from camera.v4l2_camera import V4l2Camera

# setting your camera serial number
Orbbec_CAMERA_SERIALS = {
    # Replace with actual serial number
    'cam_high': 'CH8R554001M',
    'cam_left_wrist': 'CH8G65200Z9',
    'cam_right_wrist': 'CH8G65200Y4',
}

V4l2_CAMERA_DEV = {
    "cam_high": "/dev/cam_high",
    "cam_left_wrist": "/dev/cam_left_wrist",
    "cam_right_wrist": "/dev/cam_right_wrist",
}

# fix can id 
ARM_CAN_DEV = {
    "left_arm": "can0",
    "right_arm": "can1",
}

class RealRobotEnv:
    def __init__(
        self,
        single_arm: bool,
        cam_names: list,
        visual: bool = False,
        camera_type: str = "v4l2",
        visual_fps: float = 30.0,
    ):
        self.single_arm = single_arm
        self.camera_names = cam_names
        self.camera_type = camera_type
        self.visual = visual
        self.visual_fps = visual_fps
        self._visualizer_running = False
        self._visualizer_thread = None
        self._visualizer_step = 0

        arm_names = ["right_arm"] if self.single_arm else ["left_arm", "right_arm"]
        self.controllers = {
            "arm": {arm_name: Y1Controller(arm_name) for arm_name in arm_names}
        }

        if self.camera_type == "orbbec":
            camera_cls = OrbbecCamera
            self.camera_devices = Orbbec_CAMERA_SERIALS
        elif self.camera_type == "v4l2":
            camera_cls = V4l2Camera
            self.camera_devices = V4l2_CAMERA_DEV
        else:
            raise ValueError(f"Unsupported camera_type: {self.camera_type}")

        self.cameras = {
            # Keep OpenCV preview inside camera classes for manual debugging only.
            "images": {cam_name: camera_cls(cam_name, visual=False) for cam_name in self.camera_names}
        }

    def _compose_display_image(self, frames: dict[str, np.ndarray]):
        display_order = ["cam_left_wrist", "cam_high", "cam_right_wrist"]
        ordered_frames = []

        for cam_name in display_order:
            if cam_name in self.camera_names and cam_name in frames:
                ordered_frames.append(frames[cam_name])

        for cam_name in self.camera_names:
            if cam_name not in display_order and cam_name in frames:
                ordered_frames.append(frames[cam_name])

        if not ordered_frames:
            return None

        target_height = min(frame.shape[0] for frame in ordered_frames)
        resized_frames = []
        for frame in ordered_frames:
            height, width = frame.shape[:2]
            if height != target_height:
                target_width = max(1, int(round(width * target_height / height)))
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            resized_frames.append(frame)

        return np.concatenate(resized_frames, axis=1)
        
    def set_up(self, teleop=False):
        for arm_name, controller in self.controllers["arm"].items():
            controller.set_up(ARM_CAN_DEV[arm_name], teleop=teleop)

        for cam_name, camera in self.cameras["images"].items():
            if cam_name not in self.camera_devices:
                raise RuntimeError(f"Not find device config for camera {cam_name}!")
            camera.set_up(self.camera_devices[cam_name])

        if self.visual:
            self.start_visualizer()
        print("set up success!")

    def start_visualizer(self):
        if self._visualizer_running:
            return

        rr.init("openpi_real_robot_inference", spawn=True)
        self._visualizer_running = True
        self._visualizer_thread = threading.Thread(
            target=self._visualizer_loop,
            name="real_robot_visualizer",
            daemon=True,
        )
        self._visualizer_thread.start()
        print(f"rerun visualizer started at {self.visual_fps:.1f} FPS")

    def _visualizer_loop(self):
        sleep_time = 1.0 / self.visual_fps if self.visual_fps > 0 else 0.0
        while self._visualizer_running:
            try:
                frame_bundle = {}
                for cam_name in self.camera_names:
                    image = self.cameras["images"][cam_name].get_image(timeout=0.0)
                    if image is not None:
                        frame_bundle[cam_name] = image

                if frame_bundle:
                    rr.set_time("frame", sequence=self._visualizer_step)
                    for cam_name, image in frame_bundle.items():
                        rr.log(f"camera/{cam_name}", rr.Image(image))

                    display_image = self._compose_display_image(frame_bundle)
                    if display_image is not None:
                        rr.log("camera/stitched", rr.Image(display_image))

                    self._visualizer_step += 1

                if sleep_time > 0:
                    time.sleep(sleep_time)
            except KeyboardInterrupt:
                self._visualizer_running = False
                break
            except Exception as exc:
                self._visualizer_running = False
                print(f"visualizer stopped: {exc}")
                break
    
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

            image_start = time.time()
            image = camere_images[cam_name].get_image(timeout=0.0)
            if image is None:
                image = camere_images[cam_name].get_image()
            if image is None:
                print(
                    f"not receive {cam_name} image data "
                    f"after {(time.time() - image_start) * 1000:.1f} ms"
                )
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
            # right_arm_controller.set_joint_position_control(action[0:7])

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

    def stop(self):
        self._visualizer_running = False
        if self._visualizer_thread is not None:
            self._visualizer_thread.join(timeout=1.0)
            self._visualizer_thread = None

        for camera in self.cameras["images"].values():
            camera.stop()

if __name__ == "__main__":
    env = RealRobotEnv(single_arm=False, cam_names=["cam_high", "cam_right_wrist", "cam_left_wrist"], visual=True)
    env.set_up()

    try:
        while True:
            obs = env.get_observation()
            print(f"observation : {obs}")
            time.sleep(1.0 / 30)
    finally:
        env.stop()
