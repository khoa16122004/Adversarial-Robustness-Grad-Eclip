from .utils import set_device, use_device
from .imputed_dataset import ImputedDataset


def run_road(*args, **kwargs):
	from .road import run_road as _run_road
	return _run_road(*args, **kwargs)