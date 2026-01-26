from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

class BaseBenchmark(ABC):
    @abstractmethod
    def load_data(self) -> List[Dict]:
        pass

    @abstractmethod
    def evaluate(self, prediction: str, ground_truth: str) -> bool:
        pass