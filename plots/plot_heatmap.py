import os

import matplotlib.pyplot as plt
import matplotlib.colors as col
import pandas as pd
import seaborn as sns

from definitions import ALL_COMPARTMENTS_TEST_SET_DETAILS_CSV, PLOTS_OUT_DIR, OUT_IMAGE_FORMAT, OUT_IMAGE_DPI, COL_G1_G2, \
    COL_G1_G3, COL_G2_G3, COL_G1_CNN, COL_G2_CNN, COL_G3_CNN


def plot_heatmap():
    """
    Plot Hamming distances for each of the 200 B-Scans of the test set as heat map.
    These are the six Hamming distances: g1,g2 g1,g3 g2,g3 g1,cnn g2,cnn g3,cnn.
    """
    df = _read_csv(ALL_COMPARTMENTS_TEST_SET_DETAILS_CSV)
    cmap = col.LinearSegmentedColormap.from_list("", ["green", "yellow", "red"])

    xticklabels = list(range(1, 201, 25)) + [200]
    xticks = [i - 0.5 for i in xticklabels]

    plot = sns.heatmap(df[[COL_G1_G2.csv, COL_G1_G3.csv, COL_G2_G3.csv, COL_G1_CNN.csv, COL_G2_CNN.csv, COL_G3_CNN.csv]].transpose(),
                       annot=False, cmap=cmap, xticklabels=xticklabels)
    plot.vlines(range(25, 176, 25), *plot.get_xlim(), colors='k', linestyles='solid', linewidth=0.9)
    plot.hlines(3, xmin=0, xmax=200, linestyles='solid', linewidth=0.9, colors='k')

    plot.set_xticks(xticks)
    plot.set_xticklabels(xticklabels, fontsize=9)
    plot.set_yticklabels([COL_G1_G2.plot, COL_G1_G3.plot, COL_G2_G3.plot, COL_G1_CNN.plot, COL_G2_CNN.plot,
                          COL_G3_CNN.plot])

    fig = plt.gcf()
    fig.set_size_inches(11, 2)

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.savefig(os.path.join(PLOTS_OUT_DIR, "heatmap." + OUT_IMAGE_FORMAT), dpi=OUT_IMAGE_DPI)
    plt.close()


def _read_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=',')
    df.sort_values('b_scan_name', inplace=True, key=_to_padded_names)
    return df


def _to_padded_names(names: pd.Series) -> pd.Series:
    return names.apply(_to_padded_name)


def _to_padded_name(name: str) -> str:
    """
    Pad image number in file name string, e.g. Eye1_1.png -> Eye1_01.png
    """
    name_split = name.split('_')
    name_split[-1] = name_split[-1].zfill(2)
    return '_'.join(name_split)
