import os
import matplotlib.pyplot as plt
from PIL import Image

import torch.nn as nn


def draw_results(model, images_dir='.\\data\\test_data\\bubbles', show_plot=False):
    images_list = os.listdir(images_dir)
    figs = {image_name: [] for image_name in images_list}
    for image_name in images_list:
        image = Image.open(os.path.join(images_dir, image_name))
        predicted_segmentation_map = model.predict(image)
        predicted_segmentation_map = predicted_segmentation_map.squeeze().cpu().detach().numpy()
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 9))

        ax[0].imshow(image)
        ax[1].imshow(predicted_segmentation_map, cmap="gray")
        figs[image_name] = fig
        if show_plot:
            plt.show()
    return figs


def draw_history(history, metrics_num, show_plot=False, width=8, fontsize=20):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 9))

    ax[0].plot(
        range(len(history['train'])), history['train'],
        label="train", linewidth=width
    )
    ax[0].plot(
        range(len(history['val'])), history['val'], 'r--',
        label="val", linewidth=width
    )
    ax[0].set_xlabel("epochs")
    ax[0].set_title("history")
    ax[0].tick_params(axis='both', which='major')
    ax[0].tick_params(axis='both', which='minor')
    ax[0].grid(True)
    ax[0].legend(fontsize=fontsize)

    for i, (metric_name, metric_value) in enumerate(metrics_num.items()):
        ax[1].plot(
            range(len(metric_value)), metric_value,
            label=metric_name, linewidth=width
        )
    ax[1].set_xlabel("epochs")
    ax[1].set_title("metrics")
    ax[1].tick_params(axis='both', which='major')
    ax[1].tick_params(axis='both', which='minor')
    ax[1].grid(True)
    ax[1].legend(fontsize=fontsize)
    if show_plot:
        plt.show()
    return fig

