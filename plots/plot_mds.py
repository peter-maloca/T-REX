import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import MDS

from definitions import SUBPLOT_NAME_ALL_COMPARTMENTS, SUBPLOT_NAME_VITREOUS, SUBPLOT_NAME_RETINA, SUBPLOT_NAME_CHOROID, \
    SUBPLOT_NAME_SCLERA, PLOTS_OUT_DIR, OUT_IMAGE_FORMAT, OUT_IMAGE_DPI, ALL_COMPARTMENTS_TEST_SET_SUMMARY_CSV, \
    VITREOUS_TEST_SET_SUMMARY_CSV, RETINA_TEST_SET_SUMMARY_CSV, CHOROID_TEST_SET_SUMMARY_CSV, \
    SCLERA_TEST_SET_SUMMARY_CSV

TITLE_SIZE = 16
FONT_SIZE = 12


def plot_mds():

    fig = plt.figure(figsize=(8, 12))
    fig.subplots_adjust(hspace=0.35, left=0.08, right=0.92, top=0.92, bottom=0.08)

    ax1 = plt.subplot2grid(shape=(3, 2), loc=(0, 0))
    ax1.set_title(SUBPLOT_NAME_ALL_COMPARTMENTS, size=TITLE_SIZE)
    _do_mds_and_plot(ax1, ALL_COMPARTMENTS_TEST_SET_SUMMARY_CSV)

    ax2 = plt.subplot2grid(shape=(3, 2), loc=(0, 1))
    ax2.set_title(SUBPLOT_NAME_VITREOUS, size=TITLE_SIZE)
    _do_mds_and_plot(ax2, VITREOUS_TEST_SET_SUMMARY_CSV)

    ax3 = plt.subplot2grid(shape=(3, 2), loc=(1, 0))
    ax3.set_title(SUBPLOT_NAME_RETINA, size=TITLE_SIZE)
    _do_mds_and_plot(ax3, RETINA_TEST_SET_SUMMARY_CSV)

    ax4 = plt.subplot2grid(shape=(3, 2), loc=(1, 1))
    ax4.set_title(SUBPLOT_NAME_CHOROID, size=TITLE_SIZE)
    _do_mds_and_plot(ax4, CHOROID_TEST_SET_SUMMARY_CSV)

    ax5 = plt.subplot2grid(shape=(3, 2), loc=(2, 0))
    ax5.set_title(SUBPLOT_NAME_SCLERA, size=TITLE_SIZE)
    _do_mds_and_plot(ax5, SCLERA_TEST_SET_SUMMARY_CSV)

    plt.savefig(os.path.join(PLOTS_OUT_DIR, "mds." + OUT_IMAGE_FORMAT), dpi=OUT_IMAGE_DPI)
    plt.close()


def _do_mds_and_plot(ax, csv_path: str):
    pos, names = _multidimensional_scaling(csv_path)

    min_x, max_x = _set_xlim_and_ylim(pos)
    label_height_offset = _determine_label_height_offset(min_x, max_x)

    plt.gca().set_aspect('equal', adjustable='box')  # square plots
    plt.grid(linestyle='--')
    ax.set_axisbelow(True)

    ax.scatter(pos[0:3, 0], pos[0:3, 1], c='black', marker="^")
    ax.scatter(pos[3, 0], pos[3, 1], c='red')

    ax.text(pos[0, 0], pos[0, 1] + label_height_offset, names[0], size=FONT_SIZE)
    ax.text(pos[1, 0], pos[1, 1] + label_height_offset, names[1], size=FONT_SIZE)
    ax.text(pos[2, 0], pos[2, 1] + label_height_offset, names[2], size=FONT_SIZE)
    ax.text(pos[3, 0], pos[3, 1] + label_height_offset, names[3], size=FONT_SIZE)


def _set_xlim_and_ylim(pos: np.array) -> Tuple[float, float]:
    min_x, max_x = min(pos[:, 0]), max(pos[:, 0])
    min_y, max_y = min(pos[:, 1]), max(pos[:, 1])
    delta_x = max_x - min_x
    delta_y = max_y - min_y
    delta = max(delta_x, delta_y) * 1.5
    average_x = max_x - delta_x / 2
    average_y = max_y - delta_y / 2
    x_lim = average_x - delta / 2, average_x + delta / 2
    y_lim = average_y - delta / 2, average_y + delta / 2
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    tol = 1e-6
    assert abs((x_lim[1] - x_lim[0]) - (y_lim[1] - y_lim[0])) < tol
    return min_x, max_x


def _multidimensional_scaling(csv_path: str) -> Tuple[np.array, List[str]]:
    df = _clean_df(pd.read_csv(csv_path))
    mds = MDS(2, random_state=93)
    pos = mds.fit_transform(df)
    return pos, list(df.columns)


def _determine_label_height_offset(min_y: float, max_y: float) -> float:
    plt_height = max_y - min_y
    label_height_offset = plt_height / 15
    return label_height_offset


def _clean_df(df) -> pd.DataFrame:
    first_col = df.columns[0]
    df.drop([first_col], axis=1, inplace=True)  # drop unnecessary cols
    df.drop(df.index[4:], inplace=True)  # drop unnecessary rows
    return df
