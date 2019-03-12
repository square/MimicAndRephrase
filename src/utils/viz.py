# Useful visualizations for error analysis
from textwrap import wrap
import re
from typing import List

import numpy as np

try:
    import matplotlib
    # NOTE(arun): setup matplotlib to use agg instead of tkinter because our default Python is built without Tcl.
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_agg as plt_backend_agg
    from matplotlib.figure import Figure
except ModuleNotFoundError:
    print('please install matplotlib')

def print_table(data: List[List[float]], row_labels: List[str], column_labels: List[str],
                number_format: str = "%d") -> str:
    """Pretty print tables.
    Assumes @data is a 2D array and uses @row_labels and @column_labels
    to display table.
    """
    # Convert data to strings
    data = [[number_format % v for v in row] for row in data]
    cell_width = max(
        max(map(lambda x: len(str(x)), row_labels)),
        max(map(lambda x: len(str(x)), column_labels)),
        max(max(map(len, row)) for row in data))

    def c(s):
        """adjust cell output"""
        return str(s) + " " * (cell_width - len(str(s)))

    ret = ""
    ret += "\t".join(map(c, column_labels)) + "\n"

    for l, row in zip(row_labels, data):
        ret += "\t".join(map(c, [l] + row)) + "\n"
    return ret




def figure_to_image(figure: Figure, close=True) -> np.array:
    """Render matplotlib figure to numpy format.

    :param figure matplotlib figure to plot
    :param close flag to close the figure after rendering

    :return image in [CHW] order
    """
    import numpy as np

    canvas = plt_backend_agg.FigureCanvasAgg(figure)
    canvas.draw()
    data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = figure.canvas.get_width_height()
    image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
    image_chw = np.moveaxis(image_hwc, source=2, destination=0)
    if close:
        plt.close(figure)
    return image_chw


def test_figure_to_image():
    fig = plt.Figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([1, 2, 3], [1, 2, 3])
    img = figure_to_image(fig)
    assert img
