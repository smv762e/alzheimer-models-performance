# type: ignore
import sys

class Tee:
    def __init__(self, file):
        self.terminal = sys.__stdout__  # Terminal output
        self.log_file = file  # File output

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()