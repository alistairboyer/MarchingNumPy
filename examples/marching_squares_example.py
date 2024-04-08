import numpy
import matplotlib
import matplotlib.pyplot as plt

import MarchingNumPy


def plot_2d_volume(
    ax,
    volume,
    level=0.5,
    *,
    border=0.5,
    hideticklabels=False,
    colorbar=True,
    value_display_format="",
    scatter_kwargs=None,
    grid_kwargs=None,
) -> None:
    """
    Format a matplotlib Axis and plot a volume on the formatted axis.
    The volume should be scaled between 0.0 and 1.0 and is split at the specified level.

    Args:
        ax (matplotlib.Axis): axis to plot the volume on
        volume (numpy.NDArray): the volume data. Values should be between 0.0 and 1.0
        level (float, optional): the level to cut the data. Defaults to 0.0.
        border (float, optional): width of border. Defaults to 0.5.
        hideticklabels (bool, optional): when False then the data indices are shown as tick labels on the axes. Defaults to False.
        colorbar (bool, optional): show a colorbar. Defaults to True.
        value_display_format, (str, optional): the format to display values e.g. ".2f" for 2 D.P. float. Defaults to "" when values will not be shown.
        scatter_kwargs (_type_, optional): kwargs for ax.scatter() with a basis of dict(edgecolors="k", cmap="seismic", zorder=100). Defaults to None.
        grid_kwargs (_type_, optional): kwargs for ax.grid() with a basis of dict(visible=True, linestyle="--", color="grey"). Defaults to None.

    """

    # get indices of data
    width, height = volume.shape[:2]
    xy = numpy.indices((width, height)).reshape(-1, width * height)

    # set the axis ranges
    ax.set_xlim((-border, width - 1 + border))
    ax.set_ylim((-border, height - 1 + border))

    # set the axis grid and hide labels and ticks
    ax.set_xticks(range(0, width), [""] * width if hideticklabels else None)
    ax.set_yticks(range(0, height), [""] * height if hideticklabels else None)
    ax.tick_params(length=0)
    kwargs = dict(visible=True, linestyle="--", color="grey")
    if grid_kwargs:
        kwargs.update(grid_kwargs)
    ax.grid(**kwargs)

    # turn off border
    for spine in ax.spines.values():
        spine.set_visible(False)

    # make square axes
    ax.set_aspect("equal")

    # custom scaler for levels to have uniform distance from 0.0 to level to 1.0
    def cmap_scaler(a, level):
        if level == 0.5:
            return a
        higher_than_level = (a > level).astype(int)
        # scale numbers upto and including level by a factor that maps level to be 0.5
        scaled = (1.0 - higher_than_level) * a / level
        # scale numbers greater than level similarly
        scaled += higher_than_level * (1 + a / (1.0 - level))
        # 0.5 has been factored out of the above calculations
        return scaled * 0.5

    # plot the dots
    kwargs = dict(edgecolors="k", cmap="seismic", zorder=100)
    if scatter_kwargs:
        kwargs.update(scatter_kwargs)
    # draw the dots
    dots = ax.scatter(
        xy[0],
        xy[1],
        c=cmap_scaler(
            volume, level
        ),  # scale the colorbar so that level is in the center
        vmax=1.0,
        vmin=0.0,
        **kwargs,
    )

    # add the numeric values to the ax
    if value_display_format:
        for x, y in xy.T:
            ax.text(x, y, f"{{:{value_display_format}}}".format(volume[x, y]))

    # add a colorbar
    if colorbar:
        cbar = plt.colorbar(dots, ax=ax)
        cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels([0.0, level / 2, level, 1 - level / 2, 1.0])


def plot_2d_geometry(ax, verts, lines, *, arrow=True, **plot_or_arrow_kwargs):
    """
    Plot 2d geometry on an axis.

    Args:
        ax (matplotlib.Axis): axis to plot on
        verts (_type_): sequence of x, y coordinates, e.g. numpy.array with shape == (x, 2)
        lines (_type_): collection of indexes to the verts at the start anbd end of each line, e.g. numpy.array with shape == (x, 2)
        arrow (bool, optional): whe true, draw an arrow of the data otherwise draw a line. Defaults to True.
        **kwargs (Any): Passed to either ax.arrow() or ax.plot() depending on the value of the arrow parameter.
    """

    # lines are numeric indices to the verts
    # so the coordinates can be extraced like this:
    for [x1, y1], [x2, y2] in verts[lines]:

        # draw an arrow - includes direction information
        if arrow:
            kwargs = dict(
                head_width=0.1, width=0.05, linewidth=0, length_includes_head=True
            )
            kwargs.update(plot_or_arrow_kwargs)
            ax.arrow(x1, y1, x2 - x1, y2 - y1, **kwargs)

        # draw an line
        else:
            kwargs = dict()
            kwargs.update(plot_or_arrow_kwargs)
            ax.plot([x1, x2], [y1, y2], **kwargs)


def example_arrows(width: int = 20, height: int = 10, level: float = 0.5) -> None:
    """
    Show an example of how MarchingNumPy.marching_squares() is used and can be plot using matplotlib.

    A random array of shape width, height is generated with values between 0.0 and 1.0.
    The contours are drawn along the level specified as arrows.

    Args:
        width (int, optional): Defaults to 20.
        height (int, optional): Defaults to 10.
        level (float, optional): Level at which to draw the isolines. Defaults to 0.5.
    """

    # generate random data
    array2d = numpy.random.rand(width, height)

    # initialise figure
    fig, ax = plt.subplots(figsize=(width, height))

    # set up plot by plotting dots representing the values in the volume
    plot_2d_volume(ax, array2d, level=level)

    # MarchingNumPy.marching_squares
    verts, lines = MarchingNumPy.marching_squares(
        array2d, level, interpolation="LINEAR", resolve_ambiguous=True
    )

    # plot the ouptut from marching squares
    plot_2d_geometry(ax, verts, lines, arrow=True, color="green")

    ax.set_title(f"Marching NumPy: Marching Squares with Contour at {level}")

    plt.show()


def example_contours(
    width: int = 20, height: int = 10, n_levels: int = 30, cmap_name: str = "cool"
) -> None:
    """
    Show an example of how MarchingNumPy.marching_squares() is used and can be plot using matplotlib.

    A random array of shape width, height is generated with values between 0.0 and 1.0.
    The contours are drawn along the level specified.

    Args:
        width (int, optional): _description_. Defaults to 20.
        height (int, optional): _description_. Defaults to 10.
        n_levels (int, optional): _description_. Defaults to 30.
        cmap_name (str, optional): _description_. Defaults to 'cool'.
    """
    # get the colormap
    cmap = matplotlib.colormaps.get_cmap(cmap_name)

    # generate random data
    array2d = numpy.random.rand(width, height)

    # initialise figure
    fig, ax = plt.subplots(figsize=(width, height))

    # plot dots representing the volume
    plot_2d_volume(ax, array2d, scatter_kwargs=dict(cmap=cmap))

    # calculate lines for each level as contours
    for level in numpy.linspace(0.0, 1.0, n_levels + 2)[1:-1]:
        verts, lines = MarchingNumPy.marching_squares(
            array2d, level, interpolation="LINEAR", resolve_ambiguous=True
        )

        plot_2d_geometry(ax, verts, lines, arrow=False, color=cmap(level))

    ax.set_title(f"Marching NumPy: Marching Squares with {n_levels} Contours")

    plt.show()


if __name__ == "__main__":
    example_arrows()
    example_contours()
