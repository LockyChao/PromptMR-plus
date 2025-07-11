import numpy as np
from typing import Sequence

def calculate_center_mask(shape: Sequence[int], num_low_freqs = 1)-> np.ndarray:
        """
        Build center mask based on number of low frequencies.
        Args:
            shape: Shape of k-space to mask.
            num_low_freqs: Number of low-frequency lines to sample.
        Returns:
            A mask for hte low spatial frequencies of k-space.
        """
        num_cols = shape[-2]
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad: pad + num_low_freqs] = 1
        assert mask.sum() == num_low_freqs
        return mask.shape

print(calculate_center_mask((9, 5, 176, 224, 2)))
array1 = np.random.rand(2, 2, 3, 3, 2)
array2 = np.random.rand(3, 3, 2)

array3 = array1 * array2
print(array3.shape)
