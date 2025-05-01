import os
from pathlib import Path
import sys


class ConsoleOutputSaver:
    def __init__(self, log_path: str, stream):
        """Hook that gets all stdout and stderr messages, and passes them on
        to both the stream given (typically the console) an saves them to the log_file.
        """
        os.makedirs(Path(log_path).parent, exist_ok=True)
        self.log_file = open(log_path, "a")
        self.stream = stream

    def write(self, message):
        self.log_file.write(message)
        self.stream.write(message)

    def flush(self):
        self.log_file.flush()
        self.stream.flush()

    def isatty(self):
        return False


def save_outputs(path: os.PathLike = "outputs.log"):
    """Make sure to save all stdout and stderr to file in addition to printing. 
    
    :param path: path to log to save outputs in. 
        Default is in the current working directory in a file called "outputs.log".

    Similar to tee, but works for k8s cluster without log accesses.
    """
    sys.stdout = ConsoleOutputSaver(path, sys.__stdout__)
    sys.stderr = ConsoleOutputSaver(path, sys.__stderr__)
