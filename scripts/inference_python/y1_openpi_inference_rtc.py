#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from openpi_client import websocket_client_policy

FILE_DIR = Path(__file__).resolve().parent
TRAIN_DEPLOY_ALIGNMENT_DIR = FILE_DIR.parents[1]
COMMON_DIR = TRAIN_DEPLOY_ALIGNMENT_DIR / "common"
COMMON_ENV_DIR = COMMON_DIR / "env"

for path in (FILE_DIR, COMMON_DIR, COMMON_ENV_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from real_robot_env import RealRobotEnv


DEFAULT_CAMERA_NAMES = ["cam_high", "cam_right_wrist", "cam_left_wrist"]


shutdown_event = threading.Event()
observation_ready = threading.Event()
delay_buffer: deque[float] = deque(maxlen=20)

latest_observation_lock = threading.Lock()
latest_observation: dict[str, object] | None = None

rtc_prev_chunk_lock = threading.Lock()
rtc_prev_chunk: np.ndarray | None = None


class StreamActionBuffer:
    """Action chunk buffer with optional overlap smoothing."""

    def __init__(self, *, max_chunks: int = 10, state_dim: int = 14, smooth_method: str = "temporal"):
        self.max_chunks = max(1, int(max_chunks))
        self.state_dim = int(state_dim)
        self.smooth_method = str(smooth_method)
        self.lock = threading.Lock()
        self.cur_chunk: deque[np.ndarray] = deque()
        self.last_action: np.ndarray | None = None
        self.k = 0

    def integrate_new_chunk(self, actions_chunk: np.ndarray, *, max_k: int, min_m: int = 8) -> None:
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return

            max_k = max(0, int(max_k))
            min_m = max(1, int(min_m))
            drop_n = min(self.k, max_k)
            if drop_n >= len(actions_chunk):
                return

            new_chunk = [np.asarray(a, dtype=float).copy() for a in actions_chunk[drop_n:]]

            if self.smooth_method.lower() == "raw":
                self.cur_chunk = deque(new_chunk, maxlen=None)
                self.k = 0
                return

            if len(self.cur_chunk) == 0 and self.last_action is not None:
                old_list = [np.asarray(self.last_action, dtype=float).copy() for _ in range(min_m)]
                self.last_action = None
            else:
                old_list = list(self.cur_chunk)
                if len(old_list) > 0 and len(old_list) < min_m:
                    tail = np.asarray(old_list[-1], dtype=float).copy()
                    old_list.extend([tail.copy() for _ in range(min_m - len(old_list))])
                elif len(old_list) == 0:
                    self.cur_chunk = deque(new_chunk, maxlen=None)
                    self.k = 0
                    return

            new_list = list(new_chunk)
            overlap_len = min(len(old_list), len(new_list))
            if overlap_len <= 0:
                self.cur_chunk = deque(new_list, maxlen=None)
                self.k = 0
                return

            if len(old_list) > len(new_list):
                old_list = old_list[: len(new_list)]
                overlap_len = len(new_list)

            if overlap_len == 1:
                w_old = np.array([1.0], dtype=float)
            else:
                w_old = np.linspace(1.0, 0.0, overlap_len, dtype=float)
            w_new = 1.0 - w_old

            smoothed = [
                (
                    w_old[i] * np.asarray(old_list[i], dtype=float)
                    + w_new[i] * np.asarray(new_list[i], dtype=float)
                )
                for i in range(overlap_len)
            ]
            combined = smoothed + new_list[overlap_len:]
            self.cur_chunk = deque([a.copy() for a in combined], maxlen=None)
            self.k = 0

    def pop_next_action(self) -> np.ndarray | None:
        with self.lock:
            if len(self.cur_chunk) == 0:
                return None
            if len(self.cur_chunk) == 1:
                self.last_action = np.asarray(self.cur_chunk[0], dtype=float).copy()
            action = np.asarray(self.cur_chunk.popleft(), dtype=float)
            self.k += 1
            return action


def install_signal_handler() -> None:
    def _on_sigint(sig, frame):
        del sig, frame
        shutdown_event.set()

    signal.signal(signal.SIGINT, _on_sigint)


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="fold the towel")
    parser.add_argument("--single_arm", action="store_true")
    parser.add_argument("--camera_type", choices=["v4l2", "orbbec"], default="v4l2")
    parser.add_argument("--camera_names", nargs="+", default=DEFAULT_CAMERA_NAMES)
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--visual_fps", type=float, default=30.0)
    parser.add_argument("--control_frequency", type=float, default=30.0)
    parser.add_argument("--inference_rate", type=float, default=4.0)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--max_publish_step", type=int, default=10000000)
    parser.add_argument("--init_state", nargs="*", type=float, help="Optional 7D/14D init state")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument("--interp_steps", type=int, default=10)
    parser.add_argument("--interp_frequency", type=float, default=300.0)
    parser.add_argument("--latency_k", type=int, default=8)
    parser.add_argument("--min_smooth_steps", type=int, default=8)
    parser.add_argument("--buffer_max_chunks", type=int, default=10)
    parser.add_argument("--rtc_disable_smoothing", action="store_true")
    parser.add_argument("--rtc_model_chunk_size", type=int, default=50)
    parser.add_argument("--rtc_execute_horizon", type=int, default=None)
    parser.add_argument("--rtc_max_guidance_weight", type=float, default=0.5)
    parser.add_argument("--rtc_mask_prefix_delay", action="store_true")
    return parser


def copy_observation(observation: dict[str, object] | None) -> dict[str, object] | None:
    if observation is None:
        return None
    copied: dict[str, object] = {"state": np.asarray(observation["state"], dtype=np.float32).copy()}
    images = observation.get("images", {})
    copied["images"] = {
        name: np.asarray(image, dtype=np.uint8).copy() for name, image in dict(images).items()
    }
    return copied


def set_latest_observation(observation: dict[str, object] | None) -> None:
    global latest_observation
    copied = copy_observation(observation)
    if copied is None:
        return
    with latest_observation_lock:
        latest_observation = copied
    observation_ready.set()


def get_latest_observation() -> dict[str, object] | None:
    with latest_observation_lock:
        return copy_observation(latest_observation)


def update_delay_steps(rtt_sec: float, control_frequency: float) -> int:
    if rtt_sec is None or not np.isfinite(rtt_sec):
        return 0
    delay_buffer.append(float(rtt_sec))
    if not delay_buffer:
        return 0
    median_rtt = float(np.median(np.asarray(delay_buffer, dtype=float)))
    return int(max(0, round(median_rtt * float(control_frequency))))


def throttle_loop(loop_start: float, rate_hz: float) -> None:
    if rate_hz <= 0:
        return
    remaining = 1.0 / float(rate_hz) - (time.perf_counter() - loop_start)
    if remaining > 0:
        shutdown_event.wait(remaining)


def build_payload(
    observation: dict[str, object],
    prompt: str,
    *,
    prev_action_chunk: np.ndarray | None,
    inference_delay: int,
    execute_horizon: int,
    rtc_mask_prefix_delay: bool,
    rtc_max_guidance_weight: float,
) -> dict[str, object]:
    payload = copy_observation(observation)
    if payload is None:
        raise ValueError("Observation is required for RTC payload.")
    payload["prompt"] = prompt
    payload["enable_rtc"] = True
    payload["execute_horizon"] = int(max(1, execute_horizon))
    payload["mask_prefix_delay"] = bool(rtc_mask_prefix_delay)
    payload["max_guidance_weight"] = float(rtc_max_guidance_weight)
    payload["inference_delay"] = int(max(0, inference_delay))
    if prev_action_chunk is not None:
        payload["prev_action_chunk"] = np.asarray(prev_action_chunk, dtype=float).tolist()
    return payload


def inference_fn(
    policy,
    observation: dict[str, object],
    prompt: str,
    *,
    prev_action_chunk: np.ndarray | None,
    inference_delay: int,
    execute_horizon: int,
    rtc_mask_prefix_delay: bool,
    rtc_max_guidance_weight: float,
):
    payload = build_payload(
        observation,
        prompt,
        prev_action_chunk=prev_action_chunk,
        inference_delay=inference_delay,
        execute_horizon=execute_horizon,
        rtc_mask_prefix_delay=rtc_mask_prefix_delay,
        rtc_max_guidance_weight=rtc_max_guidance_weight,
    )

    infer_start = time.perf_counter()
    out = policy.infer(payload)
    infer_elapsed_ms = (time.perf_counter() - infer_start) * 1000.0
    server_timing = out.get("server_timing", {}) if isinstance(out, dict) else {}
    infer_ms = server_timing.get("infer_ms")
    prev_total_ms = server_timing.get("prev_total_ms")
    if infer_ms is not None:
        if prev_total_ms is not None:
            print(
                f"[client] recv action in {infer_elapsed_ms:.2f} ms "
                f"(server infer {infer_ms:.2f} ms, prev total {prev_total_ms:.2f} ms)"
            )
        else:
            print(f"[client] recv action in {infer_elapsed_ms:.2f} ms (server infer {infer_ms:.2f} ms)")
    else:
        print(f"[client] recv action in {infer_elapsed_ms:.2f} ms")

    actions = out.get("actions", None) if isinstance(out, dict) else None
    if actions is None or len(actions) == 0:
        return None, infer_elapsed_ms / 1000.0
    return np.asarray(actions, dtype=float), infer_elapsed_ms / 1000.0


def execute_action(
    env: RealRobotEnv,
    observation: dict[str, object],
    action: np.ndarray,
    prev_action: np.ndarray | None,
    args,
) -> np.ndarray:
    action = np.asarray(action, dtype=float)
    state = np.asarray(observation["state"], dtype=float)
    target = np.asarray(action[: len(state)], dtype=float)

    if args.interpolation:
        interp_steps = max(1, int(args.interp_steps))
        interp_dt = 1.0 / max(float(args.interp_frequency), 1e-6)
        start_state = state if prev_action is None else np.asarray(prev_action[: len(state)], dtype=float)
        for interpolated in np.linspace(start_state, target, interp_steps):
            env.step(interpolated)
            time.sleep(interp_dt)
    else:
        env.step(action)
        time.sleep(1.0 / max(float(args.control_frequency), 1e-6))

    return action.copy()


def inference_thread_fn(policy, args, execute_horizon: int, stream_buffer: StreamActionBuffer) -> None:
    global rtc_prev_chunk

    pred_delay_steps = 0
    while not shutdown_event.is_set():
        loop_start = time.perf_counter()
        observation = get_latest_observation()
        if observation is None:
            observation_ready.wait(timeout=0.1)
            continue

        with rtc_prev_chunk_lock:
            prev_chunk = None if rtc_prev_chunk is None else np.asarray(rtc_prev_chunk, dtype=float).copy()

        try:
            action_chunk, rtt_sec = inference_fn(
                policy,
                observation,
                args.prompt,
                prev_action_chunk=prev_chunk,
                inference_delay=pred_delay_steps,
                execute_horizon=execute_horizon,
                rtc_mask_prefix_delay=args.rtc_mask_prefix_delay,
                rtc_max_guidance_weight=args.rtc_max_guidance_weight,
            )
        except Exception as exc:
            print(f"[WARN] RTC inference failed: {exc}")
            if shutdown_event.is_set():
                break
            try:
                shutdown_event.wait(1.0)
                if shutdown_event.is_set():
                    break
                policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
                print("[client] reconnected to policy server")
            except Exception as reconnect_exc:
                print(f"[WARN] RTC reconnect failed: {reconnect_exc}")
                shutdown_event.wait(1.0)
            continue

        pred_delay_steps = update_delay_steps(rtt_sec, args.control_frequency)

        if action_chunk is not None and len(action_chunk) > 0:
            action_chunk = np.asarray(action_chunk, dtype=float)
            with rtc_prev_chunk_lock:
                rtc_prev_chunk = action_chunk.copy()
            stream_buffer.integrate_new_chunk(
                action_chunk[: args.chunk_size],
                max_k=int(args.latency_k),
                min_m=int(args.min_smooth_steps),
            )

        throttle_loop(loop_start, float(args.inference_rate))


def main():
    global rtc_prev_chunk

    parser = argparse.ArgumentParser(
        description="Y1 RTC inference with async chunk inference and buffered real-time execution."
    )
    add_args(parser)
    args = parser.parse_args()
    print("args:", vars(args))
    
    install_signal_handler()

    runtime = RealRobotEnv(
        single_arm=args.single_arm,
        cam_names=args.camera_names,
        visual=args.visual,
        visual_fps=args.visual_fps,
        camera_type=args.camera_type,
    )
    runtime.set_up()
    if args.init_state is not None:
        target = np.asarray(args.init_state, dtype=float)
        expected_dim = 7 if args.single_arm else 14
        if target.shape[0] != expected_dim:
            raise ValueError(f"--init_state expects {expected_dim} values, got {target.shape[0]}")
        runtime.step(target)
        time.sleep(3.0)

    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print("Server metadata:", policy.get_server_metadata())

    execute_horizon = args.chunk_size if args.rtc_execute_horizon is None else args.rtc_execute_horizon
    execute_horizon = max(1, min(int(execute_horizon), int(args.rtc_model_chunk_size)))

    state_dim = 7 if args.single_arm else 14
    stream_buffer = StreamActionBuffer(
        max_chunks=args.buffer_max_chunks,
        state_dim=state_dim,
        smooth_method="raw" if args.rtc_disable_smoothing else "temporal",
    )
    inference_thread: threading.Thread | None = None

    try:
        initial_observation = None
        while initial_observation is None and not shutdown_event.is_set():
            initial_observation = runtime.get_observation()
            if initial_observation is None:
                shutdown_event.wait(1.0 / max(float(args.control_frequency), 1e-6))
        if initial_observation is None:
            raise RuntimeError("Failed to get initial observation before shutdown.")

        set_latest_observation(initial_observation)
        initial_state = np.asarray(initial_observation["state"], dtype=float)

        warmup_prev_chunk = np.tile(initial_state[None, :], (int(args.rtc_model_chunk_size), 1))

        try:
            _ = policy.infer(
                build_payload(
                    initial_observation,
                    args.prompt,
                    prev_action_chunk=warmup_prev_chunk,
                    inference_delay=0,
                    execute_horizon=execute_horizon,
                    rtc_mask_prefix_delay=args.rtc_mask_prefix_delay,
                    rtc_max_guidance_weight=args.rtc_max_guidance_weight,
                )
            )
            print("Warmup done.")
        except Exception as exc:
            print(f"[WARN] warmup failed: {exc}")

        with rtc_prev_chunk_lock:
            rtc_prev_chunk = warmup_prev_chunk.copy()

        inference_thread = threading.Thread(
            target=inference_thread_fn,
            args=(policy, args, execute_horizon, stream_buffer),
            daemon=True,
        )
        inference_thread.start()

        published_step = 0
        prev_action: np.ndarray | None = None
        waiting_logged = False
        while published_step < args.max_publish_step and not shutdown_event.is_set():
            observation = runtime.get_observation()
            if observation is None:
                shutdown_event.wait(1.0 / max(float(args.control_frequency), 1e-6))
                continue

            set_latest_observation(observation)
            action = stream_buffer.pop_next_action()
            if action is None:
                if not waiting_logged:
                    print("[client] waiting for RTC action chunk...")
                    waiting_logged = True
                shutdown_event.wait(1.0 / max(float(args.control_frequency), 1e-6))
                continue

            waiting_logged = False
            prev_action = execute_action(runtime, observation, action, prev_action, args)
            published_step += 1
            if published_step > 0 and published_step % 50 == 0:
                print(f"Published step {published_step}")
    finally:
        shutdown_event.set()
        if inference_thread is not None:
            inference_thread.join(timeout=1.0)
        runtime.stop()
        print("Real robot RTC inference exited.")


if __name__ == "__main__":
    main()
