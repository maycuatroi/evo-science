import numpy as np
import torch
import cv2
from os import environ
from platform import system


def wh2xy(x) -> np.ndarray:
    y = np.copy(x) if isinstance(x, np.ndarray) else x.clone()
    y[:, 0:2] = x[:, 0:2] - x[:, 2:4] / 2
    y[:, 2:4] = x[:, 0:2] + x[:, 2:4] / 2
    return y


def setup_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    _set_multiprocessing_method()
    _disable_opencv_multithreading()
    _set_thread_count("OMP_NUM_THREADS")
    _set_thread_count("MKL_NUM_THREADS")


def _set_multiprocessing_method():
    if system() != "Windows":
        torch.multiprocessing.set_start_method("fork", force=True)


def _disable_opencv_multithreading():
    cv2.setNumThreads(0)


def _set_thread_count(env_var, count="1"):
    if env_var not in environ:
        environ[env_var] = count
