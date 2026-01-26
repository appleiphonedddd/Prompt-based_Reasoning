import re
import ast
from typing import List, Dict
from datasets import load_dataset
from ..base_benchmark import BaseBenchmark

class GameOf24Benchmark(BaseBenchmark):

    def __init__(self, dataset_name="nlile/24-game", split="train"):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.data = self.load_data()
    
    def load_data(self):
        """
        Loads the dataset from Hugging Face.
        """
        print(f"Loading {self.dataset_name}...")
        return load_dataset(self.dataset_name, split=self.split)