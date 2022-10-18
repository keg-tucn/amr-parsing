from typing import List

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def plot_scores(title: str,
                input_sequence: List[str],
                output_sequence: List[str],
                text_scores: np.array,
                color_scores: np.array,
                on_training_flow: bool = False,
                show_numbers: bool = True,
                max_no_rows: int = None,
                max_no_cols: int = None):
    if max_no_cols and len(output_sequence) > max_no_cols:
        output_sequence = output_sequence[:max_no_cols]
        text_scores = text_scores[:, :max_no_cols]
        color_scores = color_scores[:, :max_no_cols]

    if max_no_rows and len(input_sequence) > max_no_rows:
        input_sequence = input_sequence[:max_no_rows]
        text_scores = text_scores[:max_no_rows, :]
        color_scores = color_scores[:max_no_rows, :]

    fig, ax = plt.subplots()
    # Create heatmap from scores.
    color_map = plt.cm.Blues if on_training_flow else plt.cm.Oranges
    ax.imshow(color_scores, cmap=color_map, alpha=0.9)

    # Add input and output sequence as labels.
    ax.set_xticks(np.arange(len(output_sequence)))
    ax.set_xticklabels(output_sequence)
    ax.set_yticks(np.arange(len(input_sequence)))
    ax.set_yticklabels(input_sequence)

    # Rotate and shift output sequence labels.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Add the scores for each cell.
    if show_numbers:
        for i in range(len(input_sequence)):
            for j in range(len(output_sequence)):
                score = "%.2f" % text_scores[i, j]
                ax.text(j, i, score, ha="center", va="center", color="y")

    ax.set_title(title)
    return fig


def save_scores_img_to_tensorboard(summary_writer: SummaryWriter,
                                   title: str,
                                   step: int,
                                   input_sequence: List[str],
                                   output_sequence: List[str],
                                   scores: np.array,
                                   on_training_flow: bool = False):
    image = plot_scores(title, input_sequence, output_sequence, scores, scores, on_training_flow)
    summary_writer.add_figure(title, image, step)


def save_masked_scores_img_to_tensorboard(summary_writer: SummaryWriter,
                                          title: str,
                                          step: int,
                                          input_sequence: List[str],
                                          output_sequence: List[str],
                                          text_scores: np.array,
                                          color_scores: np.array,
                                          on_training_flow: bool = False):
    image = plot_scores(title, input_sequence, output_sequence, text_scores, color_scores, on_training_flow)
    summary_writer.add_figure(title, image, step)
