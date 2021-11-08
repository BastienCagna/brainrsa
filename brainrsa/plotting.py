import numpy as np
# from scipy.spatial.distance import squareform, euclidean

from brainrsa.rdm import check_rdm

import matplotlib.pyplot as plt
from matplotlib.ticker import IndexFormatter, FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation
import seaborn as sns
sns.set(style="ticks")


def plot_rdm(rdm, title="", labels=None, contours=None, cnames=None,
             save=None, discret=False, ax=None, fig=None, colorbar=True,
             ticks_formatter=None, cmap=None, cbar_loc=None, legend_loc=None,
             ccolor="dimgrey", clinewidth=2, clinestyle="solid", cblabel=None,
             triangle="both", norm=None, vmin=None, vmax=None, fontsize=12,
             **kwargs):
    # If vmin and vmax are given, upper triangle use the common colorbar.
    # Lower triangle use maximal color range of the colorbar.
    # Force the RDM to vary between 0 and 1 and show real values in the colorbar
    rdm_min, rdm_max = np.min(rdm), np.max(rdm)

    rdm = check_rdm(rdm, force="matrix", triangle=triangle, vmin=vmin, vmax=vmax,
                    norm=norm, **kwargs)

    if ax is None:
        fig, ax = plt.subplots()
    if fig is None:
        fig = plt.gcf()

    if cmap is None:
        if discret:
            n_val = len(np.unique(rdm[(np.isfinite(rdm)) * (rdm != 0)])) + 1
            cmap = plt.cm.get_cmap('Set1', n_val)
        else:
            cmap = "viridis"

    img = ax.imshow(rdm, interpolation="nearest", aspect="equal", cmap=cmap)
    ax.set_title(title, fontsize=fontsize)

    # Change ticks if the label of each object is given
    if labels is not None:
        ax.xaxis.set_major_formatter(IndexFormatter(labels))
        ax.yaxis.set_major_formatter(IndexFormatter(labels))
        for tick in ax.get_xticklabels():
            tick.set_rotation(60)

    # Add contours
    if contours is not None:
        # Save image dimensions
        xlim, ylim = plt.xlim(), plt.ylim()

        # Add lines
        for ic, c in enumerate(contours):
            clr = ccolor[ic] if isinstance(ccolor, list) else ccolor
            ls = clinestyle[ic] if isinstance(clinestyle, list) else clinestyle
            lw = clinewidth[ic] if isinstance(clinewidth, list) else clinewidth

            if triangle == "upper":
                xxa, yya, xxb, yyb = [c, c], [ylim[1], c], [c, xlim[1]], [c, c]
            elif triangle == "lower":
                xxa, yya, xxb, yyb = [c, c], [c, ylim[0]], [c, ylim[1]], [c, c]
            else:
                xxa, yya, xxb, yyb = [c, c], ylim, xlim, [c, c]

            line = ax.plot(xxa, yya, color=clr, linestyle=ls, linewidth=lw)
            if isinstance(cnames, list):
                line[0].set_label(cnames[ic])
            ax.plot(xxb, yyb, color=clr, linestyle=ls, linewidth=lw)

        # Keep the original dimensions
        plt.xlim(xlim)
        plt.ylim(ylim)

        if cnames is not None:
            if legend_loc is None:
                lgd_loc = "lower left" if triangle == "upper" else None
            else:
                lgd_loc = legend_loc
            if isinstance(cnames, list):
                ax.legend(fontsize=9, loc=lgd_loc)
            elif isinstance(cnames, str):
                ax.legend((line), ([cnames]), fontsize=9, loc=lgd_loc)

    cbar_loc = "right" if cbar_loc is None else cbar_loc
    if triangle == "upper":
        ax.yaxis.tick_right()
        ax.xaxis.tick_top()
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        cbar_loc = "left" if cbar_loc is None else cbar_loc
    elif triangle == "lower":
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    divider = make_axes_locatable(ax)
    cb_ax = divider.append_axes(cbar_loc, size="05%", pad=0.05)
    if norm:
        def norm_formatter(val, pos):
            norm_v = val * (vmax - vmin) + vmin
            rdm_v = val * (rdm_max - rdm_min) + rdm_min
            return "{}\n{}".format(norm_v, rdm_v)
        ticks_formatter = FuncFormatter(norm_formatter)
    cb = fig.colorbar(img, cax=cb_ax, format=ticks_formatter)
    cb_ax.yaxis.set_ticks_position(cbar_loc)
    if cblabel is not None:
        cb.set_label(cblabel, fontsize=fontsize)

    if save:
        fig.savefig(save)

    return fig, ax


# def animate(frame, images, subjects, ages, ax):
#    images[images==0] = np.nan
#    print("[animate frame: {}/{}]".format(frame, len(images)), end="\r")
#    img = ax.imshow(images[frame], cmap="viridis")
#    ax.text(5, 35, "age: {:02}".format(ages[frame]))
#    ax.set_title(subjects[frame])
#    return img


# def rdms_animation(rdms, subjects, ages, fps=25):
#    n_rdms = len(rdms)
#    rdms = np.array(list(squareform(rdms[i]) for i in range(n_rdms)))
#    fig, ax = plt.subplots(figsize=(8, 8))
#    ax.imshow(rdms[0], vmin=0, vmax=1)
#    #plt.colorbar()

#    animation = FuncAnimation(
#        # Your Matplotlib Figure object
#        fig,
#        # The function that does the updating of the Figure
#        animate,
#        # Frame information (here just frame number)
#        np.arange(n_rdms),
#        # Extra arguments to the animate function
#        fargs=[rdms, subjects, ages, ax],
#        # Frame-time in ms; i.e. for a given frame-rate x, 1000/x
#        interval= 1000 / fps
#    )
#    return animation


def plot_position(pos, names=None, labels=None, title="MDS 2D space", ax=None,
                  colors=None, fig=None):
    """
        Arguments
        =========
        pos: 2D array (n_samples, n_components)

        names: list of str
            Name of each object

        labels: list of str
            Label of each object

        title: str or None

        ax: matplotlib.pyplot.axes or None

        fig: matplotlib.pyplot.figure or None
    """
    if ax is None:
        fig, ax = plt.subplots()
    elif fig is None:
        fig = plt.gcf()

    vmax = [np.max(pos[:, 0]), np.max(pos[:, 1])]
    vmin = [np.min(pos[:, 0]), np.min(pos[:, 1])]
    s = 300  # euclidean(vmax, vmin)**2
    if labels is not None:
        ulabels = np.unique(labels)

        if colors is None:
            cmap = plt.cm.get_cmap('Set1', len(ulabels))
            colors = list(cmap(i) for i in range(len(ulabels)))

        for il, label in enumerate(ulabels):
            sel = labels == label
            ax.scatter(pos[sel, 0], pos[sel, 1],
                       color=colors[il], label=label, s=s)
        # ax.legend()
    else:
        ax.scatter(pos[:, 0], pos[:, 1], color="turquoise", s=s)

    if names is not None:
        for i, (x, y) in enumerate(pos):
            ax.text(pos[i, 0], pos[i, 1], "\n" + names[i])

    ax.grid()
    ax.set_title(title)
    return fig, ax


def plot_dist(scores, true_score, p, title=None, save=None, ax=None, fig=None,
              two_sides=False, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    elif fig is None:
        fig = plt.gcf()

    # Plot distrib
    sns.distplot(scores, ax=ax, *kwargs)
    ax.set_title(title)
    ax.grid()

    # Mark true score value
    ax.plot([true_score, true_score], ax.get_ylim(), 'r--')
    if two_sides:
        ax.plot([-true_score, -true_score], ax.get_ylim(), 'r--')

    # Add text
    xmax, ymax = ax.get_xlim()[1], ax.get_ylim()[1]
    xtext = 1.1*true_score if true_score < 0.7*xmax else 0.5*true_score
    ax.text(
        xtext, ymax*0.8,
        "True correlation: {:0.06f}\np = {:.02f}%".format(true_score, p*100),
        fontsize=10
    )

    if save is not None:
        fig.savefig(save)
    return ax
