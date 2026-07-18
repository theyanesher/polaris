from pathlib import Path

from huggingface_hub import snapshot_download


# Hugging Face dataset repositories that make up this dataset.
REPOSITORIES = (
    "Kovavavvavava/pick_place_red_mug_20260327_1",
    "Kovavavvavava/pick_place_red_mug_20260327_2",
    "xiaochyVera/pick_red_mug_human",
    "xiaochyVera/pick_red_mug_human_1",
    "xiaochyVera/pick_red_mug_human_2",
    "xiaochyVera/pick_red_mug_human_3",
    "xiaochyVera/pick_red_mug_human_4",
)

DATASET_NAME = "pick_red_mug_human"
DATA_ROOT = Path("/data/madhavan") / DATASET_NAME


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    for i, repo_id in enumerate(REPOSITORIES):
        # Use only a numeric part directory; omit the Hugging Face username.
        local_dir = DATA_ROOT / f"{i}"

        print(f"Downloading {repo_id} to {local_dir}...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            max_workers=4,
        )

    print(f"All downloads complete: {DATA_ROOT}")


if __name__ == "__main__":
    main()
