"""Append or refresh aggregate summary footer rows in eval_results.csv files."""

from pathlib import Path

import pandas as pd
import tyro

SUMMARY_ROW = "summary"


def _episode_rows(df: pd.DataFrame) -> pd.DataFrame:
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

    grasp_summary = {}
    for col in (
        "grasp_succeeded",
        "grasp_succeeded_first_attempt",
        "pick_left_side",
        "pick_right_side",
        "pick_through_handle",
    ):
        if col in summary_df:
            measured = summary_df[col].dropna()
            values = measured.map(lambda val: str(val).strip().lower() == "true")
            grasp_summary[f"{col}_rate"] = float(values.mean()) if len(values) else 0.0
    if "grasp_attempts" in summary_df:
        attempts = pd.to_numeric(summary_df["grasp_attempts"], errors="coerce").dropna()
        grasp_summary["avg_grasp_attempts"] = (
            float(attempts.mean()) if len(attempts) else 0.0
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
        **grasp_summary,
    }
    episode_df = pd.concat([episode_df, pd.DataFrame([summary_row])], ignore_index=True)
    episode_df.to_csv(csv_path, index=False)


def main(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    _write_eval_results_with_summary(df, csv_path)
    print(f"Wrote summary footer to {csv_path}")


if __name__ == "__main__":
    tyro.cli(main)
