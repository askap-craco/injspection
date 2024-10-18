import os
import warnings
import numpy as np
import pandas as pd

from craft import sigproc
from .create_injection_params_file import total_burst_length


def sbid_str(sbid):
    """Convert the five digit SB number to SB0XXXXX."""
    if len(str(sbid)) == 5:
        sbid = 'SB0' + str(sbid)

    return sbid


def path_if_exists(path):
    if not os.path.exists(path):
        path = None
    return path


def get_dm_pccm3(freqs, dm_samps, tsamp):
    '''Stolen from
    freqs in Hz
    tsamp in s
    '''
    delay_s = dm_samps * tsamp
    fmax = np.max(freqs) * 1e-9
    fmin = np.min(freqs) * 1e-9
    dm_pccc = delay_s / 4.15 / 1e-3 / (1 / fmin**2 - 1 / fmax**2)
    return dm_pccc


def find_close_cands(raw_cands, inj, space_check=True):
    if isinstance(inj, pd.DataFrame):
        inj = inj.squeeze()  # Avoid error.

    before, after = total_burst_length(inj['dm_pccm3_inj'], width=0, bonus=0)

    close_in_time = ((raw_cands['total_sample'] > (inj['total_sample_inj'] - before))
                     & (raw_cands['total_sample'] < (inj['total_sample_inj'] + before)))
    raw_cands = raw_cands[close_in_time]

    # close_in_dm = ((raw_cands['dm_pccm3'] > (inj['dm_pccm3_inj'] - dm_dist))
    #                & (raw_cands['dm_pccm3'] < (inj['dm_pccm3_inj'] + dm_dist)))
    if space_check:
        close_in_space = np.sqrt((raw_cands['lpix']-inj['lpix_inj'])**2 + (raw_cands['mpix']-inj['mpix_inj'])**2) < 5
        raw_cands = raw_cands[close_in_space]

    return raw_cands


def count_found_clusters(found_injs, raw_cand_file):
    """Count the clusters of found injections.

    This has been outsourced cause it is time consuming (i.e. slow af).
    """
    raw_cands = pd.read_csv(raw_cand_file, index_col=0)
    found_injs = found_injs.copy()  # Avoid setting on copy warnings.
    for i in found_injs.index:
        close_raws = find_close_cands(raw_cands, found_injs.loc[i])
        if isinstance(close_raws['cluster_id'], np.number):
            found_injs.loc[i, 'n_clusters'] = 1
        else:
            found_injs.loc[i, 'n_clusters'] = len(close_raws['cluster_id'].unique())

    return found_injs


def check_masked_channels(injs, filpath):
    """Check number of channels and data drops at burst locations."""
    v, _, _ = load_filterbank(filpath)
    fil_length = v.shape[0]
    in_obsi = injs['total_sample_inj'] < fil_length
    n_used_chans = np.sum(np.isfinite(v[injs.loc[in_obsi, 'total_sample_inj'].astype(int)]), axis=-1)
    injs.loc[in_obsi, 'masked'] = n_used_chans / v.shape[-1]
    injs.loc[~in_obsi, 'masked'] = 0.

    return injs


def add_missed_cols_etc(close_cands, missed_inj, found_in='unspec'):
    """Add columns from missed_inj to the highest close cand"""
    # max_snr = close_cands['snr'].idxmax()
    # close_cands = close_cands.loc[max_snr:max_snr+1].copy()
    close_cands = close_cands.copy()
    close_cands['classification'] = found_in
    cols_in_missed = missed_inj[~pd.isna(missed_inj)].index  #[col for col in missed_inj.index if not pd.isna(col)]  # if col not in close_cands.columns
    # Don't overwrite.
    not_in_close_cands = ~cols_in_missed.isin(close_cands.columns)  #.dropna(axis='columns', how='all')
    close_cands = close_cands.assign(**{col: missed_inj[col] for col in cols_in_missed[not_in_close_cands]})
    return close_cands


def load_filterbank(filpath, tstart=0, ntimes=None):
    if tstart < 0:
        tstart = 0

    # load files
    f = sigproc.SigprocFile(filpath)
    if ntimes:
        nelements = ntimes*f.nifs*f.nchans
    else:
        nelements = -1  # loads the whole data set

    f.seek_data(f.bytes_per_element*tstart)

    if (f.nbits == 8): dtype = np.uint8
    elif (f.nbits == 32): dtype = np.float32

    v = np.fromfile(f.fin, dtype=dtype, count=nelements )
    v = v.reshape(-1, f.nifs, f.nchans)

    tend = tstart + v.shape[0]

    ### give a list of time
    taxis = np.linspace(tstart, tend, v.shape[0], endpoint=False)
    faxis = np.arange(f.nchans) * f.foff + f.fch1

    ### change 0 value to nan
    v[v == 0.] = np.nan

    return v, taxis, faxis


def list_to_str(int_list):
    """Convert list into a string with comma separated entries for printing."""
    return ''.join([str(beam)+', ' for beam in sorted(int_list)])[:-2]


def test_closest_inj(injs):
    if 'INJ_closest' in injs.columns and not np.all((injs['INJ_closest'] == injs['INJ_name']) | injs['INJ_closest'].isna()):
        warnings.warn("Found injection is not the closest in time:"
            f"{injs[(injs['INJ_closest'] != injs['INJ_name']) & ~injs['INJ_closest'].isna()]}")
        # only na when found in uniq or rfi


def test_id_presence(obsi):
    """Assert that every cluster is once and only once in the catalogs."""
    # Only use candidate files that exist.
    cand_files = [file for file in [obsi.rfi_path, obsi.found_inj_path, obsi.uniq_path] if file]
    raws = pd.read_csv(obsi.raw_cand_file, index_col=0)[['cluster_id', 'spatial_id']]
    cluster_ids = raws['cluster_id'].unique()
    in_file = np.zeros((len(cand_files), len(cluster_ids)), dtype=int)
    for i, file in enumerate(cand_files):
        file_ids = pd.read_csv(file, index_col=0)['cluster_id'].astype(int).values
        in_file[i] = np.sum(cluster_ids[:, None] == file_ids, axis=1)

    presences = in_file.sum(axis=0)

    # If a cluster is present in several files it must have different spatial IDs.
    lone_cluster_ids = np.nonzero(presences > 1)[0]
    lone = raws.loc[raws['cluster_id'].isin(lone_cluster_ids)]
    n_spatial_clusters = lone.groupby('cluster_id')['spatial_id'].apply(lambda gdf: len(gdf.unique()))
    problematic = n_spatial_clusters != presences[presences > 1]
    if np.any(problematic):
        print("Non-unique cluster ID detected. Number in Raw:")
        print(n_spatial_clusters.loc[problematic])
        print("Numbers in rfi, inj, uniq catalogs:")
        print(in_file[:, presences > 1][:, problematic])
    # assert not np.any(problematic)
    # return np.all(problematic)


def classification_to_bools(data):
    """Return some handy classifications for reporting and ploting."""
    uniq = (data['classification'] == 'uniq')
    found = (data['classification'] == 'inj') | uniq
    missed = data['classification'] == 'missed'
    rfi = data['classification'] == 'rfi'
    known = data['classification'] == 'known_source'
    raw = data['classification'] == 'raw'
    side = data['detected_in_side']
    would_missed = (missed | rfi | known | raw ) & ~side

    return uniq, found, missed, rfi, known, raw, side, would_missed



# def add_yaml_parameters(injs, yaml_path):
#     """Add some injected parameters that have not been saved by the classifier."""
#     yaml_path = yaml_path
#     blueprint = yaml.safe_load(open(yaml_path, 'r'))

#     bp_times = blueprint['injection_tsamps']
#     bp_sort_key = np.argsort(bp_times)
#     widths = np.array([blueprint['furby_props'][n]['width_samps'] for n in range(len(bp_times))])

#     injs = injs.sort_values('total_sample_inj')
#     widths_sorted = widths[bp_sort_key]
#     if len(widths) == len(injs):
#         injs['width_inj'] = widths_sorted
#     else:
#         for i, inj_name in enumerate(injs['total_sample_inj'].unique()):  # ugly but shouldn't happen often.
#             injs.loc[injs['total_sample_inj'] == inj_name, 'width_inj'] = widths_sorted[i]

#     return injs


# def get_raw_cand_file(clustering_file):
#     """Get the file with raw candidates for a file from the clustering algorithms."""
#     path, file = os.path.split(clustering_file)

#     return os.path.join(path, file[:18] + '.rawcat.csv')

# def analyze_filterbank(filpath, block_length=256):
#     """Return the mean fraction of masked channels"""
#     v, taxis, _ = load_filterbank(filpath)
#     masked_blocks = np.isnan(v).reshape(v.shape[0]//block_length, block_length, v.shape[-2], v.shape[-1])
#     masked_blocks = masked_blocks.all(1)

#     masked_fraction = masked_blocks.sum(-1) / masked_blocks.shape[-1]

#     return masked_fraction.mean, taxis.shape[0]

# Used highest SNR candidate now
# def select_candidate(group, snr_discard=9):
#     """For Every group select one candidate to keep.

#     If there is non-RFI candidates take the one with maximum SNR of those,
#     otherwise just take the max SNR one.
#     """
#     if len(group) == 1:
#         primary_cand = True
#     else:
#         primary_cand = (group['snr'] > snr_discard) & (group['classification'] != 'rfi')
#         if primary_cand.any():
#             primary_cand &= group['snr'] == group.loc[primary_cand, 'snr'].max()
#         else:
#             primary_cand = group['snr'] == group['snr'].max()

#     return primary_cand


# def old_empirical_beam_model(lmpix, eta=0.9, phi=0.6, lpix0=127.5, mpix0=127.5):
#     """Loss in SNR due to efficiency and pixelization.

#     lpix, mpix (int, float, or array): Pixel position
#     eta (float): Efficiency at beam center.
#     phi (float): Fraction of the total primary beam covered by the pixel grid.
#     lpix0, mpix0 (int or float): Beam center.
#     """
#     lpix, mpix = lmpix[:, 0], lmpix[:, 1]
#     theta_l = np.pi*phi*(lpix-lpix0)/256
#     theta_m = np.pi*phi*(mpix-mpix0)/256
#     return eta*np.cos(np.sqrt(theta_l**2 + theta_m**2))