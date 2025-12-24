"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.1 format.

Example usage: 
    ## single arm
    uv run examples/imeta_y1/convert_h5_to_lerobot.py \
        --config.h5-raw-dir /path/to/raw/data \
        --config.repo-id openpi/<dataset-name>

    ## dual arm
    uv run examples/imeta_y1/convert_h5_to_lerobot.py \
        --config.h5-raw-dir /path/to/raw/data \
        --config.repo-id openpi/<dataset-name> \
        --config.no-single-arm
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro
import cv2


@dataclass(frozen=True)
class DatasetConfig:
    h5_raw_dir: Path
    repo_id: str
    # True: single arm, False: dual arm
    single_arm: bool = True
    # fix camera names use your camera config
    cam_names: List[str] = field(default_factory=lambda: ["cam_high", "cam_right_wrist", "cam_left_wrist"])
    # cam_names: List[str] = field(default_factory=lambda: ["cam_high", "cam_right_wrist"])
    has_velocity: bool = False
    has_effort: bool = False
    # if not None, use this task
    task : str = None

    use_videos: bool = True
    fps: int = 30
    robot_type: str = "IMETA_Y1"
    push_to_hub: bool = False
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


def create_empty_dataset(
    dataset_config: DatasetConfig,
) -> LeRobotDataset:
    if dataset_config.single_arm:
      motors = ["left_joint1", 
                "left_joint2", 
                "left_joint3", 
                "left_joint4", 
                "left_joint5", 
                "left_joint6", 
                "left_gripper"]
    else:
      motors = ["left_joint1", 
                "left_joint2", 
                "left_joint3", 
                "left_joint4", 
                "left_joint5", 
                "left_joint6", 
                "left_gripper",
                "right_joint1", 
                "right_joint2", 
                "right_joint3", 
                "right_joint4", 
                "right_joint5", 
                "right_joint6", 
                "right_gripper"]
    
    cameras = dataset_config.cam_names

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if dataset_config.has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if dataset_config.has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video" if dataset_config.use_videos else "image",
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    return LeRobotDataset.create(
        repo_id=dataset_config.repo_id,
        fps=dataset_config.fps,
        robot_type=dataset_config.robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        img_bytes_seq = ep[f"/observation/images/{camera}"][()]
        # img_bytes_seq 可能是一个包含多帧字节的 NumPy 数组
        frames = []
        for frame_bytes in img_bytes_seq:
            buf = np.frombuffer(frame_bytes, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to decode frame from camera '{camera}'.")
            frames.append(img)

        imgs_per_cam[camera] = frames

    return imgs_per_cam

def load_raw_episode_data(
    ep_path: Path,
    cameras: list[str]
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observation/state"][:])
        action = torch.from_numpy(ep["/action"][:])

        velocity = None
        if "/observation/velocity" in ep:
            velocity = torch.from_numpy(ep["/observation/velocity"][:])

        effort = None
        if "/observation/effort" in ep:
            effort = torch.from_numpy(ep["/observation/effort"][:])

        imgs_per_cam = load_raw_images_per_camera(ep, cameras)

        task_description = None
        if "task" in ep.attrs:
            task_description = ep.attrs["task"]

    return imgs_per_cam, state, action, velocity, effort, task_description


def populate_dataset(
    dataset_config: DatasetConfig,
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
) -> LeRobotDataset:
    episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        imgs_per_cam, state, action, velocity, effort, task_description = load_raw_episode_data(ep_path, dataset_config.cam_names)
        num_frames = state.shape[0]

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            if velocity is not None and dataset_config.has_velocity:
                frame["observation.velocity"] = velocity[i]
            if effort is not None and dataset_config.has_effort:
                frame["observation.effort"] = effort[i]

            if dataset_config.task:
                frame["task"] = dataset_config.task
            else:
                frame["task"] = task_description
            dataset.add_frame(frame)

        dataset.save_episode()

    return dataset


def port_aloha(
    config: DatasetConfig,
):
    if (HF_LEROBOT_HOME / config.repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / config.repo_id)

    raw_data_dir  = config.h5_raw_dir.resolve()
    if not raw_data_dir.exists():
        raise ValueError("h5_raw_dir does not exist")

    hdf5_files = sorted(raw_data_dir.glob("episode_*.hdf5"))

    dataset = create_empty_dataset(
        dataset_config=config,
    )
    dataset = populate_dataset(
        config,
        dataset,
        hdf5_files,
    )

    if config.push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_aloha)