from typing import Any
from tqdm import tqdm

def range_with_timer(num_rounds: int, desc: str = 'Training AI') -> Any:
    return tqdm(range(num_rounds), desc=desc, bar_format='{l_bar}{bar}')
