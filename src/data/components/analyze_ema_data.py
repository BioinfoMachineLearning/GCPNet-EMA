import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import rootutils
import seaborn as sns
from omegaconf import DictConfig
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.ema_dataset import generate_lddt_score


@hydra.main(
    version_base="1.3", config_path="../../../configs/data", config_name="analyze_ema_data.yaml"
)
def main(cfg: DictConfig):
    """Calculates and plots the lDDT scores for the EMA test dataset.

    :param cfg: Hydra config object.
    """
    test_file_path = Path(cfg.splits_dir) / "test.lst"
    global_scores = []

    with open(test_file_path) as test_file:
        for line in tqdm(test_file):
            test_name = line.strip()
            decoy_model_path = os.path.join(cfg.decoy_dir, f"{test_name}.pdb")
            true_model_path = os.path.join(cfg.true_dir, f"{test_name}.pdb")
            lddt_scores = generate_lddt_score(
                decoy_model_path, true_model_path, lddt_exec_path=cfg.lddt_exec_path
            )
            global_scores.append(lddt_scores.mean())

    # plot the distribution of global lDDT scores
    plt.clf()
    sns.kdeplot(global_scores, fill=True)
    plt.xlabel("Global lDDT Score")
    plt.ylabel("Density")
    plt.title("Density Estimation of Global lDDT Scores for EMA Test Dataset")

    # add vertical lines for thresholds with different colors and labels
    thresholds = [1.0, 0.9, 0.7, 0.5]
    threshold_colors = ["darkblue", "cyan", "gold", "orange"]
    threshold_labels = ["Very High", "High", "Low", "Very Low"]

    for th, color, label in zip(thresholds, threshold_colors, threshold_labels):
        plt.axvline(x=th, linestyle="--", color=color, label=f"{label}: {th}", linewidth=1.5)

    # count data points between thresholds and add captions
    thresholds.sort(reverse=True)
    for i in range(len(thresholds) - 1):
        lower_threshold = thresholds[i]
        upper_threshold = thresholds[i + 1]
        points_in_region = sum(
            1 for score in global_scores if lower_threshold >= score > upper_threshold
        )
        plt.text(
            (lower_threshold + upper_threshold) / 2,
            0.05,
            f"{points_in_region} decoys",
            ha="center",
            fontsize=8,
        )

    plt.legend()
    plt.show()
    plt.savefig("ema_test_dataset_global_lddt_scores.png")


if __name__ == "__main__":
    main()
