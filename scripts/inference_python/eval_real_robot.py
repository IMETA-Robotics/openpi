import sys
sys.path.append("./")

import dataclasses
import logging
import time
import tyro
from pprint import pformat
from dataclasses import asdict
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from real_robot_env import RealRobotEnv
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

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Record the policy's behavior for debugging.
    record: bool = False

def eval_policy(
    policy,
    dataset: LeRobotDataset,
):
    # init pose
    start_idx = dataset.episode_data_index["from"][9].item()
    step = dataset[start_idx]

    init_state = step["observation.state"]
    if init_state.shape[-1] >=14:
        # dual arm
        single_arm = False
    elif init_state.shape[-1] >= 7:
        # single arm
        single_arm = True
    else :
        raise ValueError(f"Invalid state shape {init_state.shape[-1]}")
    print("single_arm: ", single_arm)

    cam_names = []
    for key, _ in step.items():
        if key.startswith("observation.images."):
            cam_name = key[len("observation.images."):]
            cam_names.append(cam_name)
    print(f"cam_names: {cam_names}")

    # real robot environment
    env = RealRobotEnv(single_arm, cam_names)
    env.set_up()

    # language
    lerobot_task = step["task"]
    print("lerobot_task: ", lerobot_task)
        
    input("Press key [enter] control robot to init position: ")
    # robot go to dataset init position
    print("wait robot to init joint position")
    init_state = step["observation.state"]
    env.step(init_state)
    time.sleep(3)

    input("Press key [enter] to start model inference: ")

    while True:
        try:
            observation = env.get_observation()
            # wait input data
            if observation is None:
                time.sleep(1/30)
                continue

            observation["prompt"] = lerobot_task

            start_time = time.time()  # 记录循环开始时间
            action_chunk = policy.infer(observation)["actions"]
            print(f"model inference time: {(time.time() - start_time) * 1000} ms")

            action_chunk = action_chunk[:30]  # 取前30个chunk
            print("action_chunk shape: ", action_chunk.shape)
            for action in action_chunk:
                env.step(action[:30])
                time.sleep(1/30)

        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            break

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

    eval_policy(policy, dataset)
    # eval_policy(policy)
    logging.info("End of eval")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))