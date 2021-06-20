from typing import NewType
import numpy as np

StateVector = NewType('StateVector', np.ndarray)
ActionMask = NewType('ActionMask', np.ndarray)
ActionVector = NewType('ActionVector', np.ndarray)
