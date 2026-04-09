import os
import sys
import fcntl
import mmap
import select
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from camera.base_camera import BaseCamera
except ModuleNotFoundError:
    from base_camera import BaseCamera


def load_v4l2_module():
    try:
        import v4l2 as module
        return module
    except TypeError as exc:
        if "unsupported operand type(s) for +: 'range' and 'list'" not in str(exc):
            raise

    candidates = [Path(entry) / "v4l2.py" for entry in sys.path if entry]
    source_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if source_path is None:
        raise ModuleNotFoundError("Unable to locate installed v4l2.py")

    source = source_path.read_text()
    source = source.replace("range(1, 9) + [0x80]", "list(range(1, 9)) + [0x80]")
    source = source.replace("range(0, 4) + [2]", "list(range(0, 4)) + [2]")

    namespace = {"__file__": str(source_path), "__name__": "v4l2", "__package__": ""}
    exec(compile(source, str(source_path), "exec"), namespace)

    class V4L2Module:
        pass

    module = V4L2Module()
    module.__dict__.update(namespace)
    return module


v4l2 = load_v4l2_module()


class V4l2Camera(BaseCamera):
    def __init__(self, name, width: int = 640, height: int = 480, visual: bool = False):
        super().__init__(name, visual)
        self.width = width
        self.height = height

        self.fd = None
        self.buffers = []

    def set_up(self, device: str):
        if self.fd is not None:
            self.stop()

        self.fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)

        fmt = v4l2.v4l2_format()
        fmt.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        fmt.fmt.pix.width = self.width
        fmt.fmt.pix.height = self.height
        fmt.fmt.pix.pixelformat = v4l2.V4L2_PIX_FMT_MJPEG
        fmt.fmt.pix.field = v4l2.V4L2_FIELD_NONE
        fcntl.ioctl(self.fd, v4l2.VIDIOC_S_FMT, fmt)

        req = v4l2.v4l2_requestbuffers()
        req.count = 4
        req.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        req.memory = v4l2.V4L2_MEMORY_MMAP
        fcntl.ioctl(self.fd, v4l2.VIDIOC_REQBUFS, req)

        self.buffers = []
        for index in range(req.count):
            buf = v4l2.v4l2_buffer()
            buf.type = req.type
            buf.memory = v4l2.V4L2_MEMORY_MMAP
            buf.index = index
            fcntl.ioctl(self.fd, v4l2.VIDIOC_QUERYBUF, buf)

            mm = mmap.mmap(
                self.fd,
                buf.length,
                mmap.PROT_READ | mmap.PROT_WRITE,
                mmap.MAP_SHARED,
                offset=buf.m.offset,
            )
            self.buffers.append(mm)
            fcntl.ioctl(self.fd, v4l2.VIDIOC_QBUF, buf)

        buf_type = v4l2.v4l2_buf_type(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE)
        fcntl.ioctl(self.fd, v4l2.VIDIOC_STREAMON, buf_type)
        print(f"Started camera: {self.name} ({device})")

    def get_image(self):
        if self.fd is None:
            raise RuntimeError("Camera is not set up.")

        ready, _, _ = select.select([self.fd], [], [], 2.0)
        if not ready:
            return None

        buf = v4l2.v4l2_buffer()
        buf.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = v4l2.V4L2_MEMORY_MMAP
        fcntl.ioctl(self.fd, v4l2.VIDIOC_DQBUF, buf)

        try:
            data = self.buffers[buf.index][:buf.bytesused]
            bgr_image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        finally:
            fcntl.ioctl(self.fd, v4l2.VIDIOC_QBUF, buf)

        if bgr_image is None:
            raise RuntimeError("Failed to decode MJPEG frame.")

        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        self.show_rgb_image(image)

        return image

    def stop(self):
        if self.fd is None:
            return

        try:
            buf_type = v4l2.v4l2_buf_type(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE)
            fcntl.ioctl(self.fd, v4l2.VIDIOC_STREAMOFF, buf_type)
        except Exception:
            pass

        for mm in self.buffers:
            try:
                mm.close()
            except Exception:
                pass
        self.buffers.clear()

        try:
            os.close(self.fd)
        finally:
            self.fd = None

if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "/dev/video0"

    cam = V4l2Camera("test_v4l2", visual=True)
    cam.set_up(device)

    try:
        while True:
            cam.get_image()
            time.sleep(1.0 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
