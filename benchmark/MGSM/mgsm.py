import re
from datasets import load_dataset
from ..base_benchmark import BaseBenchmark

class MGSMBenchmark(BaseBenchmark):
    def __init__(self, split="test"):

        super().__init__()
        self.dataset_name = "juletxara/mgsm"
        self.split = split
        self.data = self.load_data()

    def load_data(self):

        print(f"Loading {self.dataset_name} ({self.lang})...")
        return load_dataset(self.dataset_name, self.lang, split=self.split)