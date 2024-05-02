# Register dataset classes here
# from .robomimic_dataset import RobomimicDataset # Awaiting Numba Release
from wiserl.dataset.d4rl_dataset import D4RLOfflineDataset
from wiserl.dataset.ipl_dataset import IPLComparisonOfflineDataset
from wiserl.dataset.metaworld_dataset import MetaworldComparisonOfflineDataset
from wiserl.dataset.metaworld_offline_dataset import (
    MetaworldComparisonDataset,
    MetaworldOfflineDataset,
)
from wiserl.dataset.mismatched_mujoco_dataset import (
    MismatchedComparisonDataset,
    MismatchedOfflineDataset,
)

from .replay_buffer import ReplayBuffer

try:
    from .robomimic_dataset import RobomimicDataset
except ImportError:
    print("Warning: Could not import RobomimicDataset")
