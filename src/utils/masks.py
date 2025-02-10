import numpy as np
import cv2


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


def suppress_watershed_with_yolosam(mask_yolo_sam, mask_watershed):
    contours, _ = cv2.findContours(mask_yolo_sam, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    suppression_mask = np.zeros_like(mask_watershed)
    cv2.drawContours(suppression_mask, contours, -1, 255, thickness=cv2.FILLED)
    suppression_mask = cv2.dilate(suppression_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 5)),
                           iterations=3)

    updated_watershed = cv2.bitwise_and(mask_watershed, cv2.bitwise_not(suppression_mask))

    return updated_watershed


def masks_narrowing(masks, narrowing=0.1):
    """
    Narrows the provided binary masks by applying distance transformation and thresholding.
    Args:
        masks (list of numpy.ndarray): A list of binary masks where each element represents
            a segmented region (1 for foreground, 0 for background).
        narrowing (float, optional): The factor by which the masks should be narrowed.
            Defaults to 0.1, representing 10% narrowing.

    Returns:
        list of numpy.ndarray: A list of narrowed binary masks.

    Notes:
        The input masks should have the same dimensions.
        The narrowing factor determines the threshold applied to the distance-transformed masks.

    Example:
        mask1 = np.array([[0, 1, 0],
                          [1, 1, 0],
                          [0, 0, 1]])

        mask2 = np.array([[1, 0, 1],
                          [0, 1, 1],
                          [1, 0, 0]])

        narrowed_masks = masks_narrowing([mask1, mask2], narrowing=0.2)
        # Output: List of narrowed masks based on the 20% narrowing factor.
    """
    masks_narrowed = []
    for mask in masks:
        _, thresh = cv2.threshold(mask, 127, 255, 0)
        dist_expanded = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, mask_narrowed = cv2.threshold(dist_expanded, narrowing * dist_expanded.max(), 255, cv2.THRESH_BINARY)
        masks_narrowed.append(mask_narrowed)

    return masks_narrowed


def apply_correction(target, correction_path):
    correction = cv2.imread(
        correction_path,
        cv2.IMREAD_GRAYSCALE
    )
    cond = correction == 255
    mask = np.where(cond, cond, target)
    return mask


def binarize_mask(mask, threshold=127):
    return (np.where(mask > threshold, 255, 0)).astype(np.uint8)
