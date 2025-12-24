
import dataclasses
import logging
import time
import tyro
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 在导入 pyplot 之前设置
import matplotlib.pyplot as plt
from pprint import pformat
from dataclasses import asdict
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.policies import policy as _policy

@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str

@dataclasses.dataclass
class Args:
    """Arguments for the eval_dataset script."""

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | None

    # evaluate a specific episode from the dataset. default 0
    episode_idx: int = 0

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Record the policy's behavior for debugging.
    record: bool = False

def eval_policy(
    policy,
    dataset: LeRobotDataset,
    episode_idx: int,
):
    # init pose
    start_idx = dataset.episode_data_index["from"][episode_idx].item()
    step = dataset[start_idx]
    end_idx = dataset.episode_data_index["to"][episode_idx].item()

    cam_names = []
    for key, _ in step.items():
        if key.startswith("observation.images."):
            cam_name = key[len("observation.images."):]
            cam_names.append(cam_name)
    print(f"cam_names: {cam_names}")

    ground_truth_actions = []
    predicted_actions = []
        
    input("Press [Enter] key to start eval dataset):")

    step_idx = start_idx
    while step_idx < end_idx:
        step = dataset[step_idx]

        # images
        images_dict = {}
        for cam_name in cam_names:
            images_dict[cam_name] = step[f"observation.images.{cam_name}"]

        observation = {
            "state": step["observation.state"],
            "images": images_dict,
            "prompt": step["task"],
        }

        start_time = time.time()  # 记录循环开始时间
        action_chunk = policy.infer(observation)["actions"]
        print(f"policy time: {(time.time() - start_time) * 1000} ms")

        action_chunk = action_chunk[:30]  # 取前30个chunk
        for action in action_chunk:
            if step_idx >= end_idx:
                break
            ground_truth_actions.append(dataset[step_idx]["action"].numpy())
            predicted_actions.append(action[:30])
            
            step_idx += 1
            time.sleep(1/30)

    ground_truth_actions = np.array(ground_truth_actions)
    predicted_actions = np.array(predicted_actions)

    # Get the number of timesteps and action dimensions
    _, n_dims = ground_truth_actions.shape
    print("n_dims: ", n_dims)

    # Create a figure with subplots for each action dimension
    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 4*n_dims), sharex=True)
    fig.suptitle('Ground Truth vs Predicted Actions')

    # Plot each dimension
    for i in range(n_dims):
        ax = axes[i] if n_dims > 1 else axes

        ax.plot(ground_truth_actions[:, i], label='Ground Truth', color='blue')
        ax.plot(predicted_actions[:, i], label='Predicted', color='red', linestyle='--')
        ax.set_ylabel(f'Dim {i+1}')
        ax.legend()

    # Set common x-label
    axes[-1].set_xlabel('Timestep')

    plt.tight_layout()
    # plt.show()

    time.sleep(1)
    plt.savefig('eval_dataset.png')

def main(args: Args) -> None:
    logging.info(pformat(asdict(args)))

    train_config = _config.get_config(args.policy.config)
    # load trained policy
    policy = _policy_config.create_trained_policy(
        train_config, args.policy.dir
    )

    # Record the policy's behavior
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")
    
    # load dataset
    dataset = LeRobotDataset(repo_id = train_config.data.repo_id)

    eval_policy(policy, dataset, args.episode_idx)

    logging.info("End of eval")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))