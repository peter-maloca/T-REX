import os
from typing import Tuple, Iterable

import pandas as pd
from matplotlib import pyplot as plt

from definitions import ALL_COMPARTMENTS_TEST_SET_DETAILS_CSV, VITREOUS_TEST_SET_DETAILS_CSV, RETINA_TEST_SET_DETAILS_CSV, \
    CHOROID_TEST_SET_DETAILS_CSV, SCLERA_TEST_SET_DETAILS_CSV, SUBPLOT_NAME_ALL_COMPARTMENTS, SUBPLOT_NAME_VITREOUS, \
    SUBPLOT_NAME_RETINA, SUBPLOT_NAME_CHOROID, SUBPLOT_NAME_SCLERA, PLOTS_OUT_DIR, OUT_IMAGE_FORMAT, OUT_IMAGE_DPI, \
    COL_G1_G2, COL_G1_G3, COL_G2_G3, COL_G1_CNN, COL_G2_CNN, COL_G3_CNN

TITLE_SIZE = 16
FONT_SIZE = 11
FIG_SIZE = (9, 12)
Y_AXIS_EXTRA_MARGIN = 0.05


def plot_boxplots(scale_y_axis: bool = False):
    """
    Plot Hamming distances as boxplots. One boxplot
      (a) across all compartments
      (b) vitreous vs others
      (c) retina vs others
      (d) choroid vs others
      (e) sclera vs others

    Parameters
    ----------
    scale_y_axis: if true, same y axis scaling for (1) vitreous and retina, (2) all compartments, choroid, and sclera.
                  if false, different y axis scaling for each boxplot.
    """
    df_all = pd.read_csv(ALL_COMPARTMENTS_TEST_SET_DETAILS_CSV)
    df_vitreous = pd.read_csv(VITREOUS_TEST_SET_DETAILS_CSV)
    df_retina = pd.read_csv(RETINA_TEST_SET_DETAILS_CSV)
    df_choroid = pd.read_csv(CHOROID_TEST_SET_DETAILS_CSV)
    df_sclera = pd.read_csv(SCLERA_TEST_SET_DETAILS_CSV)

    fig = plt.figure(figsize=FIG_SIZE)
    fig.subplots_adjust(hspace=0.3, left=0.08, right=0.92, top=0.92, bottom=0.08)

    ax1 = plt.subplot2grid(shape=(3, 2), loc=(0, 0))
    ax1.set_title(SUBPLOT_NAME_ALL_COMPARTMENTS, size=TITLE_SIZE)
    df_all.boxplot(ax=ax1, fontsize=FONT_SIZE, showmeans=True)
    ax1.set_xticklabels([COL_G1_G2.plot, COL_G1_G3.plot, COL_G2_G3.plot, COL_G1_CNN.plot, COL_G2_CNN.plot, COL_G3_CNN.plot])

    ax2 = plt.subplot2grid(shape=(3, 2), loc=(0, 1))
    ax2.set_title(SUBPLOT_NAME_VITREOUS, size=TITLE_SIZE)
    df_vitreous.boxplot(ax=ax2, fontsize=FONT_SIZE, showmeans=True)
    ax2.set_xticklabels([COL_G1_G2.plot, COL_G1_G3.plot, COL_G2_G3.plot, COL_G1_CNN.plot, COL_G2_CNN.plot, COL_G3_CNN.plot])

    ax3 = plt.subplot2grid(shape=(3, 2), loc=(1, 0))
    ax3.set_title(SUBPLOT_NAME_RETINA, size=TITLE_SIZE)
    df_retina.boxplot(ax=ax3, fontsize=FONT_SIZE, showmeans=True)
    ax3.set_xticklabels([COL_G1_G2.plot, COL_G1_G3.plot, COL_G2_G3.plot, COL_G1_CNN.plot, COL_G2_CNN.plot, COL_G3_CNN.plot])

    ax4 = plt.subplot2grid(shape=(3, 2), loc=(1, 1))
    ax4.set_title(SUBPLOT_NAME_CHOROID, size=TITLE_SIZE)
    df_choroid.boxplot(ax=ax4, fontsize=FONT_SIZE, showmeans=True)
    ax4.set_xticklabels([COL_G1_G2.plot, COL_G1_G3.plot, COL_G2_G3.plot, COL_G1_CNN.plot, COL_G2_CNN.plot, COL_G3_CNN.plot])

    ax5 = plt.subplot2grid(shape=(3, 2), loc=(2, 0))
    ax5.set_title(SUBPLOT_NAME_SCLERA, size=TITLE_SIZE)
    df_sclera.boxplot(ax=ax5, fontsize=FONT_SIZE, showmeans=True)
    ax5.set_xticklabels([COL_G1_G2.plot, COL_G1_G3.plot, COL_G2_G3.plot, COL_G1_CNN.plot, COL_G2_CNN.plot, COL_G3_CNN.plot])

    if scale_y_axis:
        y_axis_scaling_overall_choroid_sclera = _get_y_axis_scaling([df_all, df_choroid, df_sclera])
        y_axis_scaling_retina_vitreous = _get_y_axis_scaling([df_retina, df_vitreous])
        ax1.set_ylim(y_axis_scaling_overall_choroid_sclera)
        ax2.set_ylim(y_axis_scaling_retina_vitreous)
        ax3.set_ylim(y_axis_scaling_retina_vitreous)
        ax4.set_ylim(y_axis_scaling_overall_choroid_sclera)
        ax5.set_ylim(y_axis_scaling_overall_choroid_sclera)

    plt.savefig(os.path.join(PLOTS_OUT_DIR, "boxplots." + OUT_IMAGE_FORMAT), dpi=OUT_IMAGE_DPI)
    plt.close()


def _get_y_axis_scaling(dfs: Iterable[pd.DataFrame]) -> Tuple[float, float]:
    """
    Get y axis scaling from a list of pandas data frames.
    """
    maximums = [df[[COL_G1_G2.csv, COL_G1_G3.csv, COL_G2_G3.csv, COL_G1_CNN.csv, COL_G2_CNN.csv, COL_G3_CNN.csv]]
                    .to_numpy().max() for df in dfs]
    minimums = [df[[COL_G1_G2.csv, COL_G1_G3.csv, COL_G2_G3.csv, COL_G1_CNN.csv, COL_G2_CNN.csv, COL_G3_CNN.csv]]
                    .to_numpy().min() for df in dfs]
    maximum = max(maximums)
    minimum = min(minimums)
    delta = maximum - minimum
    return minimum - Y_AXIS_EXTRA_MARGIN * delta, maximum + Y_AXIS_EXTRA_MARGIN * delta
