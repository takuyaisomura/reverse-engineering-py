from pathlib import Path

import numpy as np
import scipy.io

# Directory settings
data_dir = Path(__file__).parent / "data"

# Constants shared across the experiment
No = 32  # number of sensory stimuli
Nx = 2  # number of neuronal ensembles
Nsession = 100  # number of sessions
session_len = 256  # number of time steps per session
Ninit = 10  # number of initial sessions


class InVitroBSSDataset:
    """Neural activity dataset for in vitro neural recording with BSS stimuli."""

    def __init__(
        self,
        datatype: str,
        sample_index: int,
        data_file: str,
        description: str,
    ):
        self.datatype = datatype
        self.sample_index = sample_index
        self.data_file = data_file
        self.description = description
        self.data = self._load_data()
        self.ideal_likelihood = self._get_ideal_likelihood_mapping()
        self._baseline: np.ndarray | None = None

    def set_baseline(self, baseline: np.ndarray):
        """
        Set the baseline for the dataset.
        In this dataset, baselines have dependencies between datasets, so we need to set them later.
        """
        self._baseline = baseline

    @property
    def baseline(self) -> np.ndarray:
        if self._baseline is None:
            raise ValueError("Baseline has not been set for this dataset")
        return self._baseline

    def _load_data(self):
        data = scipy.io.loadmat(data_dir / self.data_file, simplify_cells=True)[f"data_{self.datatype}"]
        return data

    def _get_ideal_likelihood_mapping(self) -> tuple[np.ndarray, np.ndarray]:
        """Get ideal likelihood mapping based on dataset type"""
        if self.datatype == "mix0":
            qA11id = np.repeat([[1, 0.5], [0.5, 1]], No // 2, axis=0)
            qA10id = np.repeat([[0, 0.5], [0.5, 0]], No // 2, axis=0)
        elif self.datatype == "mix50":
            qA11id = np.repeat([[0.75, 0.75], [0.75, 0.75]], No // 2, axis=0)
            qA10id = np.repeat([[0.25, 0.25], [0.25, 0.25]], No // 2, axis=0)
        else:
            qA11id = np.repeat([[0.875, 0.625], [0.625, 0.875]], No // 2, axis=0)
            qA10id = np.repeat([[0.125, 0.375], [0.375, 0.125]], No // 2, axis=0)
        return qA11id, qA10id


dataset_list: list[InVitroBSSDataset] = [
    InVitroBSSDataset(
        datatype="ctrl",
        sample_index=18,
        data_file="response_data_ctrl.mat",
        description="Control condition",
    ),
    InVitroBSSDataset(
        datatype="bic",
        sample_index=0,
        data_file="response_data_bic.mat",
        description="Downregulated condition (Bicuculline)",
    ),
    InVitroBSSDataset(
        datatype="dzp",
        sample_index=4,
        data_file="response_data_dzp.mat",
        description="Upregulated condition (Diazepam)",
    ),
    InVitroBSSDataset(
        datatype="mix0",
        sample_index=0,
        data_file="response_data_mix0.mat",
        description="0% source mixed condition",
    ),
    InVitroBSSDataset(
        datatype="mix50",
        sample_index=0,
        data_file="response_data_mix50.mat",
        description="50% source mixed condition",
    ),
]


def get_datasets() -> list[InVitroBSSDataset]:
    """Get complete datasets with computed baselines"""
    from compute_baselines import compute_baselines

    baselines = compute_baselines(dataset_list)
    for dataset in dataset_list:
        dataset.set_baseline(baselines[dataset.datatype])
    return dataset_list
