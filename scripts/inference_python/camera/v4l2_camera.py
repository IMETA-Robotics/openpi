import os
import sys
import errno
import fcntl
import mmap
import select
import threading
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
        self.lock = threading.Lock()
        self.latest_image = None
        self.latest_metadata = None
        self._running = False
        self._reader_thread = None

    def _buffer_timestamp_sec(self, buf):
        if hasattr(buf.timestamp, "tv_sec") and hasattr(buf.timestamp, "tv_usec"):
            return float(buf.timestamp.tv_sec) + float(buf.timestamp.tv_usec) / 1_000_000.0
        return None

    def _dequeue_buffer(self):
        buf = v4l2.v4l2_buffer()
        buf.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = v4l2.V4L2_MEMORY_MMAP

        try:
            fcntl.ioctl(self.fd, v4l2.VIDIOC_DQBUF, buf)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                return None
            raise

        data = bytes(self.buffers[buf.index][:buf.bytesused])
        metadata = {
            "index": int(buf.index),
            "bytesused": int(buf.bytesused),
            "sequence": int(getattr(buf, "sequence", -1)),
            "timestamp": self._buffer_timestamp_sec(buf),
        }
        fcntl.ioctl(self.fd, v4l2.VIDIOC_QBUF, buf)
        return data, metadata

    def _decode_image(self, data: bytes):
        bgr_image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr_image is None:
            raise RuntimeError("Failed to decode MJPEG frame.")
        return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    def _reader_loop(self):
        while self._running and self.fd is not None:
            try:
                ready, _, _ = select.select([self.fd], [], [], 0.1)
                if not ready:
                    continue

                latest_payload = None
                while True:
                    frame = self._dequeue_buffer()
                    if frame is None:
                        break
                    latest_payload = frame

                    ready, _, _ = select.select([self.fd], [], [], 0.0)
                    if not ready:
                        break

                if latest_payload is None:
                    continue

                data, metadata = latest_payload
                image = self._decode_image(data)
                with self.lock:
                    self.latest_image = image
                    self.latest_metadata = metadata
            except Exception as exc:
                print(f"camera reader stopped for {self.name}: {exc}")
                self._running = False
                break

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
        self._running = True
        self._reader_thread = threading.Thread(target=self._reader_loop, name=f"{self.name}_reader", daemon=True)
        self._reader_thread.start()
        print(f"Started camera: {self.name} ({device})")

    def get_frame(self, timeout: float = 2.0):
        if self.fd is None:
            raise RuntimeError("Camera is not set up.")

        deadline = time.monotonic() + max(timeout, 0.0)
        while True:
            with self.lock:
                if self.latest_image is not None:
                    return self.latest_image.copy(), dict(self.latest_metadata or {})
            if time.monotonic() >= deadline:
                break
            time.sleep(0.005)
        return None

    def get_image(self, timeout: float = 2.0):
        frame = self.get_frame(timeout=timeout)
        if frame is None:
            return None
        image, _ = frame
        return image

    def stop(self):
        if self.fd is None:
            return

        self._running = False
        if self._reader_thread is not None:
            try:
                self._reader_thread.join(timeout=1.0)
            except BaseException:
                pass
            self._reader_thread = None

        try:
            buf_type = v4l2.v4l2_buf_type(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE)
            fcntl.ioctl(self.fd, v4l2.VIDIOC_STREAMOFF, buf_type)
        except BaseException:
            pass

        for mm in self.buffers:
            try:
                mm.close()
            except BaseException:
                pass
        self.buffers.clear()

        try:
            os.close(self.fd)
        finally:
            self.fd = None
            with self.lock:
                self.latest_image = None
                self.latest_metadata = None

if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "/dev/video0"

    cam = V4l2Camera("test_v4l2", visual=True)
    cam.set_up(device)

    try:
        stream_duration = 10.0
        pause_duration = 3.0
        next_pause_time = time.monotonic() + stream_duration

        while True:
            image = cam.get_image(timeout=0.0)
            if image is not None:
                cam.show_rgb_image(image)
            time.sleep(1.0 / 30)

            # if time.monotonic() >= next_pause_time:
            #     print(f"pause capture for {pause_duration:.1f}s")
            #     time.sleep(pause_duration)
            #     print("resume capture")
            #     next_pause_time = time.monotonic() + stream_duration
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
