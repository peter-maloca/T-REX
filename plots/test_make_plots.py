from plots.plot_boxplots import plot_boxplots
from plots.plot_mds import plot_mds
from plots.plot_heatmap import plot_heatmap


def test_generate_boxplots():
    plot_boxplots(scale_y_axis=True)


def test_generate_multidimensional_scaling_plots():
    plot_mds()


def test_generate_heatmap_plot():
    plot_heatmap()
