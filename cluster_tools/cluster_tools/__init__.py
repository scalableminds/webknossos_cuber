import logging
import multiprocessing
import os
import tempfile
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from shutil import rmtree
from typing import Union

from cluster_tools._utils.warning import enrich_future_with_uncaught_warning
from cluster_tools.executors.debug_sequential import DebugSequentialExecutor
from cluster_tools.executors.multiprocessing import MultiprocessingExecutor
from cluster_tools.executors.pickle import PickleExecutor
from cluster_tools.executors.sequential import SequentialExecutor
from cluster_tools.schedulers.cluster_executor import (
    ClusterExecutor,
    RemoteOutOfMemoryException,
)
from cluster_tools.schedulers.kube import KubernetesExecutor
from cluster_tools.schedulers.pbs import PBSExecutor
from cluster_tools.schedulers.slurm import SlurmExecutor

# For backwards-compatibility:
WrappedProcessPoolExecutor = MultiprocessingExecutor


def _noop():
    return True


did_start_test_multiprocessing = False


def _test_valid_multiprocessing():
    msg = """
    ###############################################################
    An attempt has been made to start a new process before the
    current process has finished its bootstrapping phase.

    This probably means that you are not using fork to start your
    child processes and you have forgotten to use the proper idiom
    in the main module:

        if __name__ == '__main__':
            main()
            ...
    ###############################################################
    """

    with get_executor("multiprocessing") as executor:
        try:
            res_fut = executor.submit(_noop)
            assert res_fut.result() == True, msg
        except RuntimeError as exc:
            raise Exception(msg) from exc
        except EOFError as exc:
            raise Exception(msg) from exc


def get_executor(environment, **kwargs):

    if environment == "slurm":
        return SlurmExecutor(**kwargs)
    elif environment == "pbs":
        return PBSExecutor(**kwargs)
    elif environment == "kubernetes":
        return KubernetesExecutor(**kwargs)
    elif environment == "multiprocessing":
        global did_start_test_multiprocessing
        if not did_start_test_multiprocessing:
            did_start_test_multiprocessing = True
            _test_valid_multiprocessing()

        return MultiprocessingExecutor(**kwargs)
    elif environment == "sequential":
        return SequentialExecutor(**kwargs)
    elif environment == "debug_sequential":
        return DebugSequentialExecutor(**kwargs)
    elif environment == "test_pickling":
        return PickleExecutor(**kwargs)
    raise Exception("Unknown executor: {}".format(environment))


Executor = Union[ClusterExecutor, MultiprocessingExecutor]
