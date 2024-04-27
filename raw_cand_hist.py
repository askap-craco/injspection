"Make a 2D histogram of raw candidates in an SB range."
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from glob import glob

from craft import sigproc
from craco.datadirs import format_sbid

from check_injections import InjectionResults


def get_raws_from_sbrange(sb_start, sb_end=None, exclude_single_events=False, get_raws=True):
    """Get the raw SNR and DM from a range of SBs."""
    if not sb_end:
        sb_end = sb_start + 1  # Search only one SB.

    sb_start = int(format_sbid(sb_start)[2:])
    sb_end = int(format_sbid(sb_end)[2:])

    if not os.path.exists('/CRACO/DATA_00/craco/'+format_sbid(sb_end)) and sb_end > sb_start:
        # If nonexistent, get newest SB id.
        sb_end = sorted(glob('/CRACO/DATA_00/craco/SB0?????'))[-1]
        sb_end = os.path.basename(sb_end)
        sb_end = int(format_sbid(sb_end)[2:]) + 1

    raw_SNR_dm = []
    uniq_SNR_dm = []
    obs_durations = []
    log_dics = []
    # sbid = sb_start
    for sbid in range(sb_start, sb_end):
        sbid = format_sbid(sbid)
        run = 'results'

        # Define Data locations to be searched.
        inj_pattern = os.path.join('/CRACO/DATA_??/craco/', sbid, 'scans/??/*/', run)

       # For every found log file try to get the candidates.
        log_files = sorted(glob(os.path.join(inj_pattern, 'search_pipeline_b??.log')))
        for file in log_files:
            scan = file[file.find('scans/')+6 : file.find('scans/') + 8]
            scantime = file[file.find('scans/')+9 : file.find('scans/') + 23]
            beam = file[file.find('pipeline_b')+10 : file.find('pipeline_b') + 12]

            obsi = InjectionResults(sbid, beam, scan=scan, run=run, scantime=scantime)
            if get_raws and obsi.raw_cand_file:
                raw_cands = pd.read_csv(obsi.raw_cand_file)
                if exclude_single_events:
                    cluster_members = raw_cands.groupby('cluster_id').count()['SNR']
                    raw_cands = raw_cands[raw_cands['cluster_id'].isin(cluster_members[cluster_members > 1].index)]

                raw_cands['sbid'] = sbid
                raw_cands['beam'] = beam
                raw_SNR_dm.append(raw_cands[['SNR', 'dm']])

                # Get raws that were later classified as unique.
                if obsi.uniq_path:
                    uniq_cands = pd.read_csv(obsi.uniq_path, index_col=0)
                    known_source = ~uniq_cands.loc[:, 'PSR_name':'ALIAS_sep'].isna().all(axis=1)
                    uniq_ids = uniq_cands.loc[known_source, 'cluster_id'].astype(int).values
                    uniq_raws = raw_cands[~raw_cands['cluster_id'].isin(uniq_ids)]

                    # no_known_source = uniq_cands.loc[:, 'PSR_name':'ALIAS_sep'].isna().all(axis=1)
                    # uniq_ids = uniq_cands.loc[no_known_source, 'cluster_id'].astype(int).values
                    # uniq_raws = raw_cands[raw_cands['cluster_id'].isin(uniq_ids)]
                    # uniq_cands = uniq_cands.loc[no_known_source]

                    uniq_SNR_dm.append(uniq_raws[['sbid', 'beam', 'SNR', 'dm']])

            if obsi.pcb_path:
                # Get the observation duration
                f = sigproc.SigprocFile(obsi.pcb_path)
                obs_durations.append(f.observation_duration)
            log_dics.append({'sbid':sbid, 'scan':scan, 'scantime':scantime, 'beam':beam,
                             'raws': bool(obsi.raw_cand_file), 'uniqs':bool(obsi.uniq_path),
                             'time':bool(obsi.pcb_path)})

    if raw_SNR_dm:
        raw_SNR_dm = pd.concat(raw_SNR_dm)
    if uniq_SNR_dm:
        uniq_SNR_dm = pd.concat(uniq_SNR_dm)
    if obs_durations:
        obs_durations = np.array(obs_durations)
    if log_dics:
        log_dics = pd.DataFrame(log_dics)
    return raw_SNR_dm, uniq_SNR_dm, obs_durations, log_dics, sb_end


def calculate_level_positions(snr, time, levels=np.array([10, 100, 1000])):
    """Calculate the indeces at which a given level of events per time is reached."""
    if not isinstance(levels, np.ndarray):
        levels = np.array(levels)

    bin_center = np.arange(snr.min(), snr.max()+0.1, 0.1)
    hist = np.histogram(snr, bins=bin_center-0.05)[0]

    # Do the cumsum of decresing SNR but keep the axis order
    cum_uniqs = np.cumsum(hist[::-1])[::-1]

    fp_per_time = cum_uniqs/time
    level_index = np.abs(levels - fp_per_time[:, None]).argmin(axis=0)
    return bin_center[level_index], level_index, fp_per_time


def fig_with_2d_hist(dm_data, snr_data, level_snrs=[], levels=[], **plot_kw):
    """Make a nice figure with all the cosmetics, while plotting is done in plot_hist2d."""
    fig, axs = plt.subplots(2, 2, sharex='col', sharey='row', tight_layout=True, gridspec_kw={
        'height_ratios':[1, 1.618], 'width_ratios':[1.618, 1], 'hspace': 0.0, 'wspace':0.0})
    ax_dm, ax_nothing, ax_2d, ax_snr = axs.flatten()
    plot_hist2d(dm_data, snr_data, xstep=1, ystep=0.1, axs=[ax_dm, ax_2d, ax_snr], **plot_kw)
    ax_2d.set(xlabel="DM (samples)", ylabel="SNR")
    ax_2d.axhline(level_snrs[0], label=f"{levels[0]} per minute (7$\sigma$)", color='green')
    ax_2d.axhline(level_snrs[1], label=f"{levels[1]} per minute (6$\sigma$)", color='green')
    for snr, level in zip(level_snrs[2:], levels[2:]):
        ax_2d.axhline(snr, label=f"{level} per minute")
    ax_2d.legend(loc='lower left', bbox_to_anchor=(1,1))

    ax_dm.xaxis.set_ticks_position('none')
    ax_snr.yaxis.set_ticks_position('none')
    ax_nothing.axis('off')

    return fig, axs


def plot_hist2d(x, y, xstep=1, ystep=1, x_min=None, x_max=None, y_min=None, y_max=None, axs=None, log=True):
    """Make a 2D histogram of the data. Also 1D in the first and last given axes."""
    x_min = x_min if x_min else x.min()
    x_max = x_max if x_max else x.max()
    y_min = y_min if y_min else y.min()
    y_max = y_max if y_max else y.max()

    x_bins = np.arange(x_min, x_max+xstep, xstep) - xstep/2
    y_bins = np.arange(y_min, y_max+ystep, ystep) - ystep/2

    if not axs:
        axs = plt.gcf().get_axes().flatten()
    if len(axs) == 3:
        ax_dm, ax_2d, ax_snr = axs
    else:
        ax_dm, ax_2d, ax_snr = axs[0], axs[2], axs[3]

    if log:
        counts, xedges, yedges, image = ax_2d.hist2d(x, y, bins=[x_bins, y_bins],  norm=mpl.colors.LogNorm())
    else:
        counts, xedges, yedges, image = ax_2d.hist2d(x, y, bins=[x_bins, y_bins])

    ax_dm.stairs(counts.sum(axis=1), x_bins, color='k')
    ax_snr.stairs(counts.sum(axis=0), y_bins, orientation='horizontal', color='k')

    if log:
        ax_dm.set_yscale('log')
        ax_snr.set_xscale('log')

    plt.colorbar(image, ax=ax_snr)

    return ax_dm, ax_2d, ax_snr


def make_hist_of_sb_range(sb_start, sb_end=None, levels=[1000, 10000], fig_path=None, all_raw=False, **plot_kw):
    """Main function calling the rest"""
    raw_SNR_dm, uniq_SNR_dm, obs_durations, log_dics, sb_end = get_raws_from_sbrange(sb_start, sb_end,
                                                                                     exclude_single_events=False)

    # Do a quick test.
    compatible_file_presences = (not np.any(log_dics['uniqs'] & ~log_dics['time'])
                                 and not np.any(log_dics['raws'] & ~log_dics['time']))
    if not compatible_file_presences:
        print("Problem Sir")

    # Calculate events per minute for cumulative SNR.
    total_time = obs_durations.sum() / 36 / 60  # /n_beams and s to minutes
    levels = [3, 94] + levels
    level_snrs_uniq, level_index_uniq, fp_per_minute_uniq = calculate_level_positions(uniq_SNR_dm['SNR'], total_time,
                                                                                      levels=levels)

    if fig_path == '':
        fig_path = os.getcwd()

    if all_raw:
        level_snrs, level_index, fp_per_minute = calculate_level_positions(raw_SNR_dm['SNR'], total_time, levels=levels)
        fig, axs = fig_with_2d_hist(raw_SNR_dm['dm'], raw_SNR_dm['SNR'], level_snrs, levels, **plot_kw)
        fig.suptitle(f"All raw candidates in the SB range [{sb_start}, {sb_end})")
        if sb_end:
            fig.savefig(os.path.join(fig_path, f"all_raw_cands_in_{sb_start}_{sb_end}.png"))

    fig, axs = fig_with_2d_hist(uniq_SNR_dm['dm'], uniq_SNR_dm['SNR'], level_snrs_uniq, levels, **plot_kw)
    fig.suptitle(f"Unknown raw candidates in the SB range [{sb_start}, {sb_end})")
    if fig_path:
        fig.savefig(os.path.join(fig_path, f"unknown_raw_cands_in_{sb_start}_{sb_end}.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(dest='sb_start', type=int, help="The start SB, e.g., 58306")
    parser.add_argument('-e', '--sb_end', type=int, help="The end SB, e.g., 58316. If None, only sb_start will be "
                        "plotted. If higher than existing, all SBs until the last will be used.")
    parser.add_argument('-l', '--levels', type=list, default=[1000, 10000],
                        help="Levels to be shown in candidates per minute.")
    parser.add_argument('-f', '--fig_path', type=str, help="Location to store figures. Default is the current working "
                        "directory.")
    parser.add_argument('-r', '--all_raw', action='store_true',
                        help="Whether to make a plot for all raw candidates including known sources.")
    parser.add_argument('-u', '--log', action='store_false',
                        help="If you don't want the histogram to be in log.")
    parser.add_argument('-y', '--y_max', type=int, default=20, help="Maximum SNR to be plotted.")

    args = parser.parse_args()
    print(args)

    make_hist_of_sb_range(**vars(args))