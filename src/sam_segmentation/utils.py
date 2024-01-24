import numpy as np


def unite_masks(masks):
    """
    Combines multiple binary masks into a single mask by performing logical OR operation.

    Args:
        masks (list of numpy.ndarray): A list of binary masks where each element represents
            a segmented region (1 for foreground, 0 for background).

    Returns:
        numpy.ndarray: A binary mask representing the union of all input masks.
            The result is obtained by performing a logical OR operation.

    Notes:
        The input masks should have the same dimensions.

    Example:
        mask1 = np.array([[0, 1, 0],
                          [1, 1, 0],
                          [0, 0, 1]])

        mask2 = np.array([[1, 0, 1],
                          [0, 1, 1],
                          [1, 0, 0]])

        result = unite_masks([mask1, mask2])
        # Output:
        # [[1, 1, 1],
        #  [1, 1, 1],
        #  [1, 0, 1]]
    """
    combined_mask = np.zeros_like(masks[0])

    for mask in masks:
        combined_mask = np.logical_or(combined_mask, mask)

    return combined_mask.astype(np.uint8) * 255
