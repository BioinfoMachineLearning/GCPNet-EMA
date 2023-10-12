import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import rootutils
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

    # plot the distribution of global plDDT scores
    plt.hist(global_scores, bins=20)
    plt.xlabel("Global plDDT Score")
    plt.ylabel("Count")
    plt.title("Distribution of Global plDDT Scores")
    plt.show()
    plt.savefig("global_plddt_scores.png")


if __name__ == "__main__":
    main()
