# ******************************************************************************
#  Copyright (c) 2023 Orbbec 3D Technology, Inc
#  
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.  
#  You may obtain a copy of the License at
#  
#      http:# www.apache.org/licenses/LICENSE-2.0
#  
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ******************************************************************************
import cv2
import threading
import time
import numpy as np

try:
    from camera.base_camera import BaseCamera
except ModuleNotFoundError:
    from base_camera import BaseCamera

from pyorbbecsdk import *
from pyorbbecsdk import Config
from pyorbbecsdk import OBError
from pyorbbecsdk import OBSensorType, OBFormat
from pyorbbecsdk import Pipeline, FrameSet
from pyorbbecsdk import VideoStreamProfile

def get_device_by_serial(context: Context, target_serial: str):
    """
    根据序列号获取设备
    """
    device_list = context.query_devices()
    for i in range(device_list.get_count()):
        device = device_list.get_device_by_index(i)

        serial_number = device.get_device_info().get_serial_number()
        if serial_number == target_serial:
            return device
    return None

class OrbbecCamera(BaseCamera):
    def __init__(self, name, visual: bool = False):
        super().__init__(name, visual)
        self.latest_frame = None
        self.lock = threading.Lock()
    
    def set_up(self, camera_serial):
        # 创建上下文
        self.context = Context()
        # 根据序列号获取设备
        self.device = get_device_by_serial(self.context, camera_serial)
        if self.device is None:
            print(f"Cannot find device with serial number: {camera_serial}")
            exit()

        self.config = Config()
        self.pipeline = Pipeline(self.device)

        # rgb stream
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            try:
                color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
            except OBError as e:
                print(e)
                exit()
            self.config.enable_stream(color_profile)
        except Exception as e:
            print(e)
            exit()

        def frame_callback(frame_set):
            # 只保存最新帧
            with self.lock:
                self.latest_frame = frame_set

        self.pipeline.start(self.config, frame_callback)
        print(f"Started camera: {self.name} (SN: {camera_serial})")
        
    def get_latest_frame(self):
        """获取最新帧"""
        with self.lock:
            return self.latest_frame
    def get_image(self):
        start_time = time.time()
        frame = self.get_latest_frame()
        # print(f"wait frame time :{(time.time() - start_time) * 1000}")
        if not frame:
            # print("No frame")
            return None
        
        color_frame = frame.get_color_frame()
        if not color_frame:
            raise RuntimeError("Failed to get color frame.")
        
        # RGB
        width = color_frame.get_width()
        height = color_frame.get_height()
        data = np.asanyarray(color_frame.get_data()).copy()
        image = np.resize(data, (height, width, 3))
        # print(f"image shape : {image.shape} , image dtype : {image.dtype}")

        self.show_rgb_image(image)
        
        return image

    def stop(self):
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()


if __name__ == "__main__":
    cam = OrbbecCamera("test", True)
    cam.set_up("CH8XA52010Y")

    while True:
        try:
            data = cam.get_image()
            time.sleep(1.0 / 30)
        except KeyboardInterrupt:
            cam.stop()    
            break

