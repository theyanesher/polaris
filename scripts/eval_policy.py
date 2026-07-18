import os
import tyro
import mediapy

# import wandb
import tqdm
import gymnasium as gym
import torch
import argparse
import pandas as pd


from pathlib import Path
from isaaclab.app import AppLauncher

from polaris.config import EvalArgs
from polaris.utils_.eval_utils import randomize_object_poses, set_seed
import numpy as np

SUMMARY_ROW = "summary"


def _episode_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rollout rows, ignoring the aggregate summary footer."""
    if df.empty or "episode" not in df.columns:
        return df
    return df[df["episode"].astype(str) != SUMMARY_ROW].copy()


def _write_eval_results_with_summary(df: pd.DataFrame, csv_path: Path) -> None:
    episode_df = _episode_rows(df)
    summary_df = episode_df.copy()
    summary_success = summary_df["success"].map(
        lambda val: str(val).strip().lower() == "true"
    )

    num_episodes = len(summary_df)
    num_success = int(summary_success.sum()) if num_episodes else 0
    success_rate = num_success / num_episodes if num_episodes else 0.0
    avg_progress = float(summary_df["progress"].mean()) if num_episodes else 0.0
    avg_episode_length = (
        float(summary_df["episode_length"].mean()) if num_episodes else 0.0
    )

    for col in ["num_success", "success_rate", "avg_progress", "avg_episode_length"]:
        if col not in episode_df.columns:
            episode_df[col] = pd.NA

    summary_row = {
        "episode": SUMMARY_ROW,
        "episode_length": avg_episode_length,
        "success": num_success,
        "progress": avg_progress,
        "num_success": num_success,
        "success_rate": success_rate,
        "avg_progress": avg_progress,
        "avg_episode_length": avg_episode_length,
    }
    episode_df = pd.concat([episode_df, pd.DataFrame([summary_row])], ignore_index=True)
    episode_df.to_csv(csv_path, index=False)


def _as_bool(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        return bool(value.item())
    if isinstance(value, (list, tuple)):
        return bool(value[0]) if value else False
    return bool(value)


def _video_frame_from_obs(obs: dict, fallback: np.ndarray | None = None) -> np.ndarray | None:
    splat = obs.get("splat", {})
    if not isinstance(splat, dict) or not splat:
        return fallback

    preferred_pairs = [
        ("cam1", "wrist_cam"),
        ("external_cam", "wrist_cam"),
        ("front_img_1", "wrist_img"),
    ]
    for left_key, right_key in preferred_pairs:
        if left_key in splat and right_key in splat:
            left = np.asarray(splat[left_key])
            right = np.asarray(splat[right_key])
            if left.ndim == right.ndim == 3 and left.shape[0] == right.shape[0]:
                return np.ascontiguousarray(np.concatenate([left, right], axis=1))

    for key in ("cam1", "external_cam", "front_img_1", "wrist_cam", "wrist_img"):
        if key in splat:
            return np.ascontiguousarray(np.asarray(splat[key]))

    return np.ascontiguousarray(np.asarray(next(iter(splat.values()))))


def main(eval_args: EvalArgs):
    set_seed(eval_args.seed)
    # This must be done before importing anything from IsaacLab
    # Inside main function to avoid launching IsaacLab in global scope
    # >>>> Isaac Sim App Launcher <<<<
    parser = argparse.ArgumentParser()
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = eval_args.headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    # >>>> Isaac Sim App Launcher <<<<

    from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
    from polaris.environments.manager_based_rl_splat_environment import (
        ManagerBasedRLSplatEnv,
    )
    from polaris.utils import load_eval_initial_conditions, load_task_config
    from polaris.client import InferenceClient
    # from real2simeval.autoscoring import TASK_TO_SUCCESS_CHECKER

    if eval_args.env_folder is None:
        eval_args.env_folder = os.path.dirname(gym.spec(eval_args.environment).kwargs["usd_file"])

    env_cfg = parse_env_cfg(
        eval_args.environment,
        device="cuda",
        num_envs=1,
        use_fabric=True,
    )

    if eval_args.control_frequency_hz is not None:
        if eval_args.control_frequency_hz <= 0:
            raise ValueError("control_frequency_hz must be positive")
        decimation = 1.0 / (env_cfg.sim.dt * eval_args.control_frequency_hz)
        rounded_decimation = round(decimation)
        if not np.isclose(decimation, rounded_decimation):
            raise ValueError(
                f"Requested {eval_args.control_frequency_hz} Hz cannot be represented "
                f"exactly with sim.dt={env_cfg.sim.dt}; decimation={decimation}"
            )
        env_cfg.decimation = int(rounded_decimation)
        env_cfg.sim.render_interval = env_cfg.decimation
        print(
            f"Overriding environment control rate to {eval_args.control_frequency_hz:g} Hz "
            f"(sim.dt={env_cfg.sim.dt}, decimation={env_cfg.decimation})"
        )

    env_cfg.episode_length_s = eval_args.max_episode_length * (env_cfg.sim.dt * env_cfg.decimation)
    
    env: ManagerBasedRLSplatEnv = gym.make(eval_args.environment, cfg=env_cfg)  # type: ignore

    language_instruction, initial_conditions = load_eval_initial_conditions(
        usd=env.usd_file,
        initial_conditions_file=eval_args.initial_conditions_file,
        rollouts=eval_args.rollouts,
    )
    task_config_path = eval_args.task_config or os.path.join(
        eval_args.env_folder, "task_config.yaml"
    )
    object_randomization, _ = load_task_config(task_config_path)

    # Randomise object poses
    ic = randomize_object_poses(object_randomization, initial_conditions)[0]

    rollouts = eval_args.rollouts
    # Resume CSV logging
    run_folder = Path(eval_args.run_folder)
    run_folder.mkdir(parents=True, exist_ok=True)
    csv_path = run_folder / "eval_results.csv"
    if csv_path.exists():
        episode_df = _episode_rows(pd.read_csv(csv_path))
    else:
        episode_df = pd.DataFrame(
            {
                "episode": pd.Series(dtype="int"),
                "episode_length": pd.Series(dtype="int"),
                "success": pd.Series(dtype="bool"),
                "progress": pd.Series(dtype="float"),
            }
        )
    episode = len(episode_df)
    if episode >= rollouts:
        print("All rollouts have been evaluated. Exiting.")
        env.close()
        simulation_app.close()
        return

    policy_client: InferenceClient = InferenceClient.get_client(eval_args.policy)

    video = []
    horizon = eval_args.max_episode_length
    bar = tqdm.tqdm(range(horizon))
    obs, info = env.reset(
        object_positions=ic, expensive=True
    )
    policy_client.reset()

    print(f" >>> Starting eval job from episode {episode + 1} of {rollouts} <<< ")
    while True:
        action, viz = policy_client.infer(obs, language_instruction, return_viz=True)
        obs, rew, term, trunc, info = env.step(
            torch.tensor(action).reshape(1, -1), expensive=True
        )
        if eval_args.save_video:
            frame = _video_frame_from_obs(obs, fallback=viz)
            if frame is not None:
                video.append(frame)

        bar.update(1)
        success = _as_bool(info["rubric"]["success"])
        if term[0] or trunc[0] or success or bar.n >= horizon:
            policy_client.reset()

            if eval_args.save_video:
                filename = run_folder / f"episode_{episode}.mp4"
                mediapy.write_video(filename, video, fps=15)

            # Log episode results to CSV
            episode_data = {
                "episode": episode,
                "episode_length": bar.n,
                "success": success,
                "progress": info["rubric"]["progress"],
            }
            episode_df = pd.concat(
                [episode_df, pd.DataFrame([episode_data])], ignore_index=True
            )
            _write_eval_results_with_summary(episode_df, csv_path)

            bar.close()
            print(f"Episode {episode} finished. Episode length: {bar.n}")
            episode += 1
            bar = tqdm.tqdm(range(horizon))
            ic = randomize_object_poses(object_randomization, initial_conditions)[0]
            obs, info = env.reset(
                object_positions=ic, expensive = True
            )

            video = []
            if episode >= rollouts:
                break

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    args: EvalArgs = tyro.cli(EvalArgs)
    main(args)
