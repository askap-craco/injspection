import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

# Change some defaults plot parameters (sneaky).
TEXTWIDTH = 7.0282
COLUMNWIDTH = 3.37574803

plt.rc('figure', figsize=(COLUMNWIDTH, COLUMNWIDTH*3/4))

SMALL_SIZE = 9
MEDIUM_SIZE = 9
BIGGER_SIZE = 9

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)


def plot_injection_param(injected, detected, parameter_name):
    """For one parameter plot the injected against the detected signal.

    Return a figure with two axis, the second holding the difference between injected and detected parameters.
    """
    height_ratios = (2, 1)
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.04, 'height_ratios' : (2, 1)},
                                  figsize=(COLUMNWIDTH, COLUMNWIDTH*4/3))
    ax.plot(injected, detected, '.')
    ax2.plot(injected, detected-injected, '.')

    # Change to physical names.
    if parameter_name == 'dm_pccm3':
        parameter_name = "DM (pc/cm$^3$)"

    # Label.
    ax2.set_xlabel(f"Injected {parameter_name}")
    ax.set_ylabel(f"Detected {parameter_name}")
    ax2.set_ylabel("Injected $-$ detected")

    # Set limits.
    max_data = np.max([detected, injected])
    min_data = np.min([detected, injected])
    excess = (max_data-min_data)/20
    lim = [min_data-excess, max_data+excess]
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax2.set_xlim(lim)
    ax.set_aspect('equal', 'box')

    # Make the second axis have the same width.
    x1,x2 = ax2.get_xlim()
    y1,y2 = ax2.get_ylim()
    xrange = x2-x1
    yrange = y2-y1
    ax2.set_aspect(height_ratios[1]/height_ratios[0]*xrange/yrange)

    fig.tight_layout()

    return fig, (ax, ax2)


def plot_var_in_all_beams(selected_data, x_var):
    """Plot efficiency against one variable for all beams individually.

    Args:
        selected_data (pandas.DataFrame): Injection results.
        x_var (str): Variable to plot on the x-axis, e.g., one of "SNR_inj", "width_samps_inj", "lpix", "dm_pccm3_inj".
        fig_path (str): Path to save the figure.
    """
    plot_mean = False
    fig, axs = plt.subplots(6, 6, sharex=True, sharey=True, gridspec_kw={'hspace': 0.0, 'wspace':0.0}, figsize=(TEXTWIDTH, TEXTWIDTH))  # IEEE two columns are 7.16.
    # fig.suptitle("SNR in [10, 50], width=2, DM=200, (lpix, mpix)=(100, 100)", y=0.93)
    # fig.suptitle("SNR=16, width in (0.25, 10), DM=200, (lpix, mpix)=(100, 100)", y=0.93)
    # fig.suptitle("SNR=16, width=2, DM=200, mpix=64, lpix in linspace(0, 256, 100)", y=0.93)
    # fig.suptitle("SNR=16, width=2, DM=0, mpix=127, lpix in linspace(0, 270, 271)", y=0.93)
    snr_low_limit = 9  # SNR limit used in the search.

    beam_fit = []

    for beam in range(36):
        # Plot data for a single beam.
        row, col = np.divmod(beam, 6)
        data = selected_data[selected_data['beam'] == beam]

        # Make a few masks for different happenings.
        found = ~data['missed'] & ~data['known_source']
        too_low_snr = (~data['missed']) & (data['SNR'] < snr_low_limit)
        misclassified = data['missed'] & (~data['SNR'].isna())
        not_seen = data['missed'] & data['SNR'].isna()
        misattributed = data['known_source']
        uniq_mc = (data.loc[misclassified, 'classification'] == 'uniq') & ~data.loc[misclassified, 'also_in_rfi']
        side_of_rfi = (data.loc[misclassified, 'classification'] == 'uniq') & data.loc[misclassified, 'also_in_rfi']

        ax = axs[row, col]
        ax.plot(data.loc[found, x_var], data.loc[found, 'SNR/SNR_inj'], '.', label=f"{len(data)} total")
        if too_low_snr.any():
            ax.plot(data.loc[too_low_snr, x_var], data.loc[too_low_snr, 'SNR/SNR_inj'], '.', color='purple',
                    label=f"{np.sum(too_low_snr)} SNR < {snr_low_limit}")
        if misclassified.any():
            if (~uniq_mc).any():
                ax.plot(data[misclassified].loc[~uniq_mc, x_var], data[misclassified].loc[~uniq_mc, 'SNR/SNR_inj'],
                        '.', color='red', label=f"{np.sum(~uniq_mc)} RFI")
            if uniq_mc.any():
                ax.plot(data[misclassified].loc[uniq_mc, x_var], data[misclassified].loc[uniq_mc, 'SNR/SNR_inj'],
                        '.', color='orange', label=f"{np.sum(uniq_mc)} in uniq")
            if side_of_rfi.any():
                ax.plot(data[misclassified].loc[side_of_rfi, x_var],
                        data[misclassified].loc[side_of_rfi, 'SNR/SNR_inj'],
                        '.', color='darkorange', label=f"{np.sum(side_of_rfi)} sidecluster")
        if misattributed.any():
            ax.plot(data.loc[misattributed, x_var], data.loc[misattributed, 'SNR/SNR_inj'], '.', color='magenta',
                    label=f"{np.sum(misattributed)} psr etc.")
        if not_seen.any():
            ax.plot(data.loc[not_seen, x_var], 6/data.loc[not_seen, 'SNR_inj'], 'v', color='r',
                    label=f"{np.sum(not_seen)} missed")
        if plot_mean:
            ax.plot(data.groupby(x_var)[x_var].first(), data.groupby(x_var)['SNR/SNR_inj'].mean(), color='black')
        if x_var == "lpix":
            func = lambda x, A, w, phi=127: A*np.cos(np.pi*w*(x-phi)/256)
            x_data = data.loc[~not_seen, x_var].dropna().astype(float)
            y_data = data.loc[~not_seen, 'SNR/SNR_inj'].dropna().astype(float)
            fit = curve_fit(func, x_data, y_data, p0=[1, 1, 127])
            ax.plot(x_data, func(x_data, *fit[0]), c='black')
            beam_fit.append(fit)

        ax.legend(loc='upper right', bbox_to_anchor=(1., 1.1), fontsize='xx-small')
        ax.set_title(f"Bm {beam}", loc='left', y=0.8, fontsize='small')

    if x_var == 'SNR_inj':
        fig.supxlabel("Injected SNR")
    else:
        fig.supxlabel(x_var)

    fig.supylabel("Detected / Injected SNR")

    plt.tight_layout()
    return fig, axs


def plot_efficiency(data, snr='SNR_inj', snr_low_limit=9, ms=3):
    fig, axs = plt.subplots(3, 3, tight_layout=True, figsize=(TEXTWIDTH, TEXTWIDTH), sharey=True,
                            gridspec_kw={'wspace': 0.0, 'hspace':0.3})
    #sharex=True ,, 'wspace':0.01

    # Make a few masks for different happenings.
    found = ~data['missed'] & ~data['known_source']
    too_low_snr = (~data['missed']) & (data['SNR'] < snr_low_limit)
    misclassified = data['missed'] & (~data['SNR'].isna())
    not_seen = data['missed'] & data['SNR'].isna()
    misattributed = data['known_source']
    uniq_mc = data.loc[misclassified, 'classification'] == 'uniq'

    x_vars = [snr, 'width_samps_inj', 'subsample_phase_inj', 'lpix', 'mpix', 'dm_pccm3_inj', 'beam',
              'total_sample_inj']
    xlabels = [snr, r"Width (samples)", "Subsample phase",
               "lpix", "mpix", r"DM (pc cm$^3$)",
               "Beam", "Time (samples)"]

    # Take the mean of the efficiency over different beams.
    agg_dict = {x_var : 'first' for x_var in x_vars}
    agg_dict.update({'SNR/'+snr : 'mean'})
    eff_mean = data.groupby('INJ_name').agg(agg_dict)

    for ax, x_var, xlabel in zip(axs.flatten(), x_vars, xlabels):
        if found.any():
            ax.plot(data.loc[found, x_var], data.loc[found, 'SNR/'+snr], '.', ms=ms, alpha=0.3,
                    label=f"{len(data)} total")  #np.sum(found)
        if too_low_snr.any():
            ax.plot(data.loc[too_low_snr, x_var], data.loc[too_low_snr, 'SNR/'+snr], '.', color='purple', ms=ms,
                    alpha=0.3, label=f"{np.sum(too_low_snr)} SNR < {snr_low_limit}")
        if not x_var == 'beam':
            x_var_sorted = eff_mean[x_var].sort_values()
            ax.plot(x_var_sorted, eff_mean.loc[x_var_sorted.index, 'SNR/'+snr], color='k', lw=1, label="Beam average")
        if misclassified.any():
            ax.plot(data[misclassified].loc[uniq_mc, x_var], data[misclassified].loc[uniq_mc, 'SNR/'+snr], '.',
                    color='orange', ms=ms, alpha=0.3, label=f"{np.sum(uniq_mc)} in uniq")
            ax.plot(data[misclassified].loc[~uniq_mc, x_var], data[misclassified].loc[~uniq_mc, 'SNR/'+snr], '.',
                    color='red', ms=ms, label=f"{np.sum(~uniq_mc)} RFI")
        if misattributed.any():
            ax.plot(data.loc[misattributed, x_var], data.loc[misattributed, 'SNR/'+snr], '.', color='magenta', ms=ms,
                    label=f"{np.sum(misattributed)} psr etc.")
        if not_seen.any():
            ax.plot(data.loc[not_seen, x_var], 6/data.loc[not_seen, snr], 'v', color='r', ms=0.5*ms,
                    label=f"{np.sum(not_seen)} missed")

        ax.set_xlabel(xlabel)

    ax.legend(loc='upper left', bbox_to_anchor=(1.1, .9), fontsize='small')
    fig.supylabel("Detected / Expected SNR")

    # Delete unused axes.
    for ax in axs.flatten():
        if not ax.has_data():
            ax.axis('off')

    return fig, axs


def plot_beamfitting(bestfit, uncert):
    if uncert:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, COLUMNWIDTH))
        plt.rc('axes', labelsize=12)
        # 0, 1, 2 are Amplitude, freqency/narrowness, phase
        x, y = 1, 0
        ax1.errorbar(bestfit[:,x], bestfit[:,y], xerr=np.sqrt(uncert[:,x,x]), yerr=np.sqrt(uncert[:,y,y]), fmt='.')
        ax1.set_ylabel("Amplitude")
        ax1.set_xlabel("Covered Beamfraction")

        if bestfit.shape[1] == 4:
            x, y = 2, 3
            ax2.errorbar(bestfit[:,x], bestfit[:,y], xerr=np.sqrt(uncert[:,x,x]), yerr=np.sqrt(uncert[:,y,y]), fmt='.')
            ax2.xlabel("Beam centre l")
            ax2.ylabel("Beam centre m")
        else:
            y = 2
            ax2.errorbar(range(36), bestfit[:,y], yerr=np.sqrt(uncert[:,y,y]), fmt='.')
            ax2.xlabel("Beam")
            ax2.ylabel("Beam centre m")

        plt.title("Fit results when fitting beams individually")

    return fig, (ax1, ax2)

def make_all_pretty_plots(collated_data, pixelfit=None, pixelfit_unc=None, fig_path="", sbid="", run=""):
    """Make the various plots from this module to analyse the injections."""
    x_var = "SNR_inj"
    fig, axs = plot_var_in_all_beams(collated_data, x_var=x_var)
    if fig_path:
        fig.suptitle(f"{sbid}, {run}")
        fig.savefig(os.path.join(fig_path, f"{x_var}_efficiency_per_beam.pdf"))

    fig, axs = plot_efficiency(collated_data)
    if fig_path:
        fig.suptitle(f"{sbid}, {run}")
        fig.savefig(os.path.join(fig_path, "efficiency_pre_fit.pdf"))
    fig, axs = plot_efficiency(collated_data, snr='SNR_expected')
    if fig_path:
        fig.suptitle(f"{sbid}, {run}")
        fig.savefig(os.path.join(fig_path, "efficiency.pdf"))

    if pixelfit is None:
        fig, axs = plot_beamfitting(pixelfit, pixelfit_unc)
        if fig_path:
            fig.suptitle(f"{sbid}, {run}")
            fig.savefig(os.path.join(fig_path, "pixelization_shape.pdf"))

    inj_snr = collated_data.groupby('INJ_name')['SNR']
    snr_mean, snr_std = inj_snr.mean(), inj_snr.std()
    fig, ax = plt.subplots()
    ax.plot(snr_mean, snr_std, '.')
    ax.set(title=f"{sbid}, {run}", xlabel="SNR", ylabel="Standarddeviation in SNR for each injection")
    if fig_path:
        fig.savefig(os.path.join(fig_path, "stddev.pdf"))

    # Plot parameters against their injected values.
    params = ['SNR', 'dm_pccm3', 'lpix', 'mpix', 'total_sample', 'width_samps']
    for param in params:
        injected = collated_data[param+'_inj']
        if param == 'SNR':
            detected = collated_data[param].fillna(5)
        elif param == 'width_samps':
            detected = collated_data['boxc_width']
        else:
            detected = collated_data[param]

        fig, axs = plot_injection_param(injected, detected, param)
        if fig_path:
            fig.savefig(os.path.join(fig_path, f"injection_{param}_{sbid}.pdf"))
