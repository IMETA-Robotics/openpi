from abc import ABC, abstractmethod

import cv2


class BaseCamera(ABC):
    def __init__(self, name: str, visual: bool = False):
        self.name = name
        self.visual = visual

    @abstractmethod
    def set_up(self, device_id: str):
        pass

    @abstractmethod
    def get_image(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    def show_rgb_image(self, image):
        if not self.visual:
            return

        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.name, bgr_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("User interrupted preview")

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass
