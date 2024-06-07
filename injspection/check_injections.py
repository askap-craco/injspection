import os
import warnings
import numpy as np
import pandas as pd
import yaml
import pickle

from glob import glob
from scipy.optimize import curve_fit
from importlib.metadata import version

from craft import sigproc
from craft import uvfits

# import injspection

from .create_injection_params_file import total_burst_length
from injspection.plot_results import make_all_pretty_plots


class InjectionResults:
    """Make all the path handling easier by saving them in a class"""

    def __init__(self, sbid, beam='00', run='inj', scan='00', scantime='??????????????',
                 clustering_dir='clustering_output', tsamp=0.013824, log_file=None):
        """Define all the paths

        Parameters:
            sbid (str): Observation ID, e.g. 'SB058479'.
            beam (str or int): Beam number
            run (str): The name of the run directory, e.g. 'results'.
            scan (str): Name of scan directory.
            scan_pattern (str): The pattern to search for the run directory.
            clustering_dir (str): Directory with the results of the clustering
            tsamp (float): Sampling time (s). This is not in the logs.
        """
        sbid = sbid_str(sbid)

        # Make beam possible as int and str.
        if isinstance(beam, np.integer):
            self.beam_int = int(beam)
            beam = str(beam)
        else:
            self.beam_int = int(beam)
        if len(beam) == 1:
            beam = '0' + beam

        # Define Data locations to be searched.
        obs_path_pattern = '/CRACO/DATA_??/craco/'
        inj_pattern = os.path.join(obs_path_pattern, sbid, 'scans/', scan, scantime, run, f'search_pipeline_b{beam}.log')
        log_paths = glob(inj_pattern)
        if len(log_paths) == 0:
            raise ValueError(f"No logfile found at {inj_pattern}.")
        elif len(log_paths) > 1:
            warnings.warn(f"{len(log_paths)} directories matching the pattern have been found. Continuing with {log_paths[0]}")
        inj_path = os.path.dirname(log_paths[0])

        # Get all the useful paths
        self.sbid = sbid
        self.beam = beam
        self.tsamp = tsamp
        self.run_path = inj_path
        self.log_path = log_paths[0]
        self.log_file = log_file
        clustering_path = os.path.join(inj_path, clustering_dir)
        self.clustering_path = clustering_path
        self.orig_inj_path = path_if_exists(os.path.join(clustering_path, f'candidates.b{beam}.txt.inject.orig.csv'))
        self.found_inj_path = path_if_exists(os.path.join(clustering_path, f'candidates.b{beam}.txt.inject.cand.csv'))
        self.raw_cand_file = path_if_exists(os.path.join(clustering_path, f'candidates.b{beam}.txt.rawcat.csv'))
        self.uniq_path = path_if_exists(os.path.join(clustering_path, f'candidates.b{beam}.txt.uniq.csv'))
        self.rfi_path = path_if_exists(os.path.join(clustering_path, f'candidates.b{beam}.txt.rfi.csv'))
        self.crossmatch_path = path_if_exists(os.path.join(clustering_path, f'candidates.b{beam}.txt.catalog_cross_match.i1.csv'))
        self.alias_path = path_if_exists(os.path.join(clustering_path, f'candidates.b{beam}.txt.alias_filter.i2.csv'))
        self.pcb_path = path_if_exists(os.path.join(inj_path, f'pcb{beam}.fil'))
        self.uvfits_path = path_if_exists(os.path.join(os.path.dirname(inj_path), f'b{beam}.uvfits'))

    def calculate_dm_pccm3(self):
        """Calculate maximum searched DM."""
        fmin = self.fmin
        fmax = fmin + self.foff*self.nchan
        self.dm_pccm3 = get_dm_pccm3([fmin, fmax], self.ndm, self.tsamp)

    def compare_uvfits_fil_lengths(self):
        """Check duration length in uvfits file and compare agains the filterbank file."""
        fits_length = self.get_planned_obs_length()
        fil_length = self.get_fil_length()
        length_ratio = fil_length / fits_length
        if length_ratio < 0.9:
            warnings.warn(f"The pcb file is only {length_ratio} of the uvfits file. Better check the log mate.")

    def get_yaml_path(self, log_path=None):
        """Get the yaml file that has been used for injections from the log and some more log things."""
        if not log_path:
            log_path = self.log_path

        yaml_path = ''  # incase
        with open(log_path, 'r') as f:
            for line in f:
                if line.startswith('INFO:craco.search_pipeline:Injecting data described by'):
                    path_start = line.find('by') + 3
                    yaml_path = line[path_start:].rstrip('\n')
                if line.startswith('INFO:craft.craco_plan:Nbl='):
                    self.fmin, self.foff, self.nchan = [float(par.split('=')[1]) for par in line.split()
                                                        if par[:4] in ['Fch1', 'foff', 'ncha']]
                if line.startswith('INFO:craft.craco_plan:making Plan'):
                    self.ndm = [int(par.split('=')[1]) for par in line.split(sep=', ')
                                                        if par[:4] == 'ndm='][0]

        return yaml_path

    def count_log_injections(self, log_path=None):
        """Count the number of times a successful injection is logged in the logfile."""
        if not log_path:
            log_path = self.log_path
        count = 0
        with open(log_path, 'r') as f:
            for line in f:
                if "INFO:Visbility_injector:Simulating" in line:
                    count += 1
        return count

    def get_planned_obs_length(self):
        """Duration of the uvfits file in samples."""
        uvsource = uvfits.open(self.uvfits_path)
        nblocks = uvsource.nblocks  # The definition in craft is weird, this seems to be samples.
        return nblocks

    def get_fil_length(self):
        """Duration of filterbank file in samples."""
        f = sigproc.SigprocFile(self.pcb_path)
        return f.nsamples

    def search_classification(self, cluster_id):
        """Search the different classification files if they contain the cluster ID"""
        cluster_id = int(cluster_id)
        if cluster_id in pd.read_csv(self.uniq_path, index_col=0)['cluster_id'].astype(int).values:
            classification = 'uniq'
        elif cluster_id in pd.read_csv(self.rfi_path, index_col=0)['cluster_id'].astype(int).values:
            classification = 'rfi'
        # elif cluster_id in pd.read_csv(self.crossmatch_path, index_col=0)['cluster_id'].astype(int).values:
        #     classification = 'catalog'
        # elif cluster_id in pd.read_csv(self.alias_path, index_col=0)['cluster_id'].astype(int).values:
        #     classification = 'alias'
        elif cluster_id in pd.read_csv(self.found_inj_path, index_col=0)['cluster_id'].astype(int).values:
            classification = 'inject'
        else:
            classification = 'unclear'
        return classification

    def do_all_checks(self):
        """Execute all functions."""
        # self.compare_uvfits_fil_lengths()
        # test_id_presence(self)
        self.yaml_path = path_if_exists(self.get_yaml_path())
        log_injs = self.count_log_injections()
        injs = check_candpipe_files(self.orig_inj_path, self.found_inj_path, self, num_injs=log_injs,
                                    log_file=self.log_file)
        test_closest_inj(injs)
        # if not 'n_clusters' in injs.columns:
        #     # Initialize column, else it will be discarded.
        #     injs['n_clusters'] = pd.Series(dtype='int')
        # injs[~injs['missed']] = count_found_clusters(found_injs=injs[~injs['missed']], raw_cand_file=self.raw_cand_file)  #slow af
        # injs = add_yaml_parameters(injs, self.yaml_path)  #"/CRACO/DATA_00/craco/injection/inj_params7_pc_width.yml") #
        injs = check_masked_channels(injs, filpath=self.pcb_path)

        return injs


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


def check_candpipe_files(orig_file, found_file, obsi, print_missed=False, num_injs=None, log_file=None):
    """Compare the .inject.orig.csv with the .inject.cand.csv file.

    Search the raw candidates for missing candidates and collect all in a DataFrame.
    """
    orig_injs = pd.read_csv(orig_file)

    # Exclude candidates beyond the filelength.
    v, _, _ = load_filterbank(obsi.pcb_path)
    fil_length = v.shape[0]
    in_obsi = orig_injs['total_sample_inj'] < fil_length
    orig_injs = orig_injs[in_obsi]

    if num_injs and len(orig_injs) > num_injs:
        print("Less bursts have been injected according to the log than planned!", file=log_file)
        # orig_injs = orig_injs.loc[:num_injs]  # Feel like this is unsave (sometimes an injection gets picked up several times).

    found_injs = pd.read_csv(found_file)
    found_injs = found_injs[~found_injs['snr'].isna()]  # Not needed depending on the day.
    # Yuanming seems to be changing the pipeline output on a weekly basis.
    # found_injs = found_injs[found_injs['INJ_name'].isin(orig_injs['name'])]  Was workaround with line above

    # Crosscheck the time to make sure which injection was found. (Don't trust pipeline)
    closest_inj = np.argmin(np.abs(found_injs['total_sample'].to_numpy()[:, None]
                                   -orig_injs['total_sample_inj'].to_numpy()),
                            axis=1)
    found_injs['INJ_closest'] = orig_injs['INJ_name'].to_numpy()[closest_inj]

    missed = ~ orig_injs['INJ_name'].isin(found_injs['INJ_name'])
    missed_injs = orig_injs[missed].copy()
    # Print a useful message.
    if not missed_injs.empty and print_missed:
        print(f"================ {obsi.sbid}  Beam {obsi.beam_int} ================", file=log_file)
        print("Missed:", file=log_file)
        print(missed_injs, file=log_file)

    # Summarize useful data from all injections for plotting etc.
    # missed_injs = missed_injs.rename(columns={'name':'INJ_name', 'total_sample':'total_sample_inj',
    #                                           'dm_pccm3':'dm_pccm3_inj'})
    missed_injs['missed'] = True
    found_injs['missed'] = False

    # Search uniq, rfi, and raw candidates for missed ones and save their properties. Exclude the raws that were found.
    if not missed_injs.empty:
        uniq_cands = pd.read_csv(obsi.uniq_path, index_col=0)
        rfi_cands = pd.read_csv(obsi.rfi_path, index_col=0)
    raw_loaded = False
    noninj_cands = []

    for mj in missed_injs.index:
        missed_inj = missed_injs.loc[mj]
        close_uniqs = find_close_cands(uniq_cands, missed_inj)
        if not close_uniqs.empty:
            noninj_cands.append(add_missed_cols_etc(close_uniqs, missed_inj, found_in='uniq'))

        close_rfis = find_close_cands(rfi_cands, missed_injs.loc[mj])
        if not close_rfis.empty:
            noninj_cands.append(add_missed_cols_etc(close_rfis, missed_inj, found_in='rfi'))

        if close_uniqs.empty and close_rfis.empty:
            if not raw_loaded:
                raw_cands = pd.read_csv(obsi.raw_cand_file, index_col=0)
                raw_cands = raw_cands[~raw_cands['cluster_id'].isin(found_injs['cluster_id'])]
            close_raws = find_close_cands(raw_cands, missed_injs.loc[mj])
            if not close_raws.empty:
                max_snr = close_raws['snr'].idxmax()
                missed_injs.loc[mj, close_raws.columns] = close_raws.loc[max_snr]
                missed_injs.loc[mj, 'n_clusters'] = len(close_raws['cluster_id'].unique())
                missed_injs.loc[mj, 'found_in'] = 'raw'
                missed_injs.loc[mj, 'classification'] = obsi.search_classification(close_raws.loc[max_snr, 'cluster_id'])
                # Indexing with lists avoids conversion to Series. But doesn't work because index is different.
        else:
            missed_injs = missed_injs.drop(index=mj)

    injs = pd.concat([found_injs, missed_injs, *noninj_cands], ignore_index=True).sort_values('total_sample_inj').reset_index(drop=True)
    injs['beam'] = obsi.beam_int
    injs['missed'] = injs['missed'].astype(bool)

    return injs


def add_missed_cols_etc(close_cands, missed_inj, found_in='unspec'):
    """Add columns from missed_inj to the highest close cand"""
    max_snr = close_cands['snr'].idxmax()
    close_cands = close_cands.loc[max_snr:max_snr+1].copy()
    close_cands['classification'] = found_in
    missing_columns = [col for col in missed_inj.index if not pd.isna(col)]  # if col not in close_cands.columns
    close_cands = close_cands.assign(**{col: missed_inj[col] for col in missing_columns})
    return close_cands


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


def add_yaml_parameters(injs, yaml_path):
    """Add some injected parameters that have not been saved by the classifier."""
    yaml_path = yaml_path
    blueprint = yaml.safe_load(open(yaml_path, 'r'))

    bp_times = blueprint['injection_tsamps']
    bp_sort_key = np.argsort(bp_times)
    widths = np.array([blueprint['furby_props'][n]['width_samps'] for n in range(len(bp_times))])

    injs = injs.sort_values('total_sample_inj')
    widths_sorted = widths[bp_sort_key]
    if len(widths) == len(injs):
        injs['width_inj'] = widths_sorted
    else:
        for i, inj_name in enumerate(injs['total_sample_inj'].unique()):  # ugly but shouldn't happen often.
            injs.loc[injs['total_sample_inj'] == inj_name, 'width_inj'] = widths_sorted[i]

    return injs


def check_masked_channels(injs, filpath):
    """Check number of channels and data drops at burst locations."""
    v, _, _ = load_filterbank(filpath)
    fil_length = v.shape[0]
    in_obsi = injs['total_sample_inj'] < fil_length
    n_used_chans = np.sum(np.isfinite(v[injs.loc[in_obsi, 'total_sample_inj'].astype(int)]), axis=-1)
    injs.loc[in_obsi, 'masked'] = n_used_chans / v.shape[-1]
    injs.loc[~in_obsi, 'masked'] = 0.

    return injs


def get_injection_results(sbid, run='inj', obs_path_pattern='/CRACO/DATA_??/craco/', scan_pattern='scans/??/*/',
                          clustering_dir='clustering_output', log_file=None):
    """For the given injection run get the found and missed candidates."""
    sbid = sbid_str(sbid)
    # Define Data locations to be searched.
    inj_pattern = os.path.join(obs_path_pattern, sbid, scan_pattern, run)

    # Get beam numbers that were used in this observation (usually 00 to 35).
    log_files = sorted(glob(os.path.join(inj_pattern, 'search_pipeline_b??.log')))

    collated_data = []

    for file in log_files:
        scan = file[file.find('scans/')+6 : file.find('scans/') + 8]
        scantime = file[file.find('scans/')+9 : file.find('scans/') + 23]
        beam = file[file.find('pipeline_b')+10 : file.find('pipeline_b') + 12]

        obsi = InjectionResults(sbid, beam, scan=scan, run=run, scantime=scantime, clustering_dir=clustering_dir,
                                log_file=log_file)
        try:
            collated_data.append(obsi.do_all_checks())
        except ValueError:
            print(f"{sbid}, {beam}, {scan}, {run}, {scantime}, {clustering_dir}, {log_file}")

    collated_data = pd.concat(collated_data)

    return collated_data


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


def get_raw_cand_file(clustering_file):
    """Get the file with raw candidates for a file from the clustering algorithms."""
    path, file = os.path.split(clustering_file)

    return os.path.join(path, file[:18] + '.rawcat.csv')

# def analyze_filterbank(filpath, block_length=256):
#     """Return the mean fraction of masked channels"""
#     v, taxis, _ = load_filterbank(filpath)
#     masked_blocks = np.isnan(v).reshape(v.shape[0]//block_length, block_length, v.shape[-2], v.shape[-1])
#     masked_blocks = masked_blocks.all(1)

#     masked_fraction = masked_blocks.sum(-1) / masked_blocks.shape[-1]

#     return masked_fraction.mean, taxis.shape[0]


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
    if not np.all((injs['INJ_closest'] == injs['INJ_name']) | injs['INJ_closest'].isna()):
        warnings.warn(f"Found injection is not the closest in time:\n{injs[(injs['INJ_closest'] == injs['INJ_name']) | injs['INJ_closest'].isna()]}")


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


def select_candidate(group, snr_discard=9):
    """For Every group select one candidate to keep.

    If there is non-RFI candidates take the one with maximum SNR of those,
    otherwise just take the max SNR one.
    """
    if len(group) == 1:
        primary_cand = True
    else:
        primary_cand = (group['snr'] > snr_discard) & (group['classification'] != 'rfi')
        if primary_cand.any():
            primary_cand &= group['snr'] == group.loc[primary_cand, 'snr'].max()
        else:
            primary_cand = group['snr'] == group['snr'].max()

    return primary_cand


def beam_model(lmpix, eta=0.9, phi=0.6, lpix0=128, mpix0=128):
    """Loss in SNR due to efficiency and pixelization.

    lpix, mpix (int, float, or array): Pixel position
    eta (float): Efficiency at beam center.
    phi (float): Fraction of the total primary beam covered by the pixel grid.
    lpix0, mpix0 (int or float): Beam center.
    """
    lpix, mpix = lmpix[:, 0], lmpix[:, 1]
    theta_l = np.pi*phi*(lpix-lpix0)/256
    theta_m = np.pi*phi*(mpix-mpix0)/256
    return eta*np.cos(np.sqrt(theta_l**2 + theta_m**2))


def fit_beams(data, fix_mpix0=False, fix_lpix0=False):
    """"Fit a beam model to the injections."""
    if fix_mpix0 and fix_lpix0:
        raise ValueError("One of lpix and mpix must be fitted. Otherwise what's the point?")

    if fix_mpix0 or fix_lpix0:
        p0 = [1, .5, 128]
    else:
        p0 = [1, .5, 128, 128]
    beams = data['beam'].unique()
    beam_fit = []
    for beam in beams:
        beam_data = data[data['beam'] == beam]
        pixels = beam_data[['lpix_inj', 'mpix_inj']].dropna().astype(np.float64).to_numpy()
        if fix_lpix0:
            # Swap columns such that mpix will be fitted
            pixels = pixels[:,::-1]
        recovery_fraction = beam_data['SNR/SNR_inj'].dropna().astype(np.float64).to_numpy()
        fit = curve_fit(beam_model, pixels, recovery_fraction, p0=p0, maxfev=1000)
        beam_fit.append(fit)

    bestfit, uncert = [fit[0] for fit in beam_fit], [fit[1] for fit in beam_fit]
    bestfit, uncert = np.stack(bestfit), np.stack(uncert)

    return beams, bestfit, uncert


def collate_observation_data(sbid, run='inj', clustering_dir='clustering_output', log_file=None):
    """Get the data for one sbid, clean it up a bit, and give out some numbers."""
    collated_data = get_injection_results(sbid, run, clustering_dir=clustering_dir, log_file=log_file)

    # Drop some unused columns.
    collated_data = collated_data.drop(columns=['time', 'iblk', 'rawsn', 'obstime_sec', 'mjd', 'total_sample_middle', 'mSNR', 'mSlope',
        'mSlopeSmooth', 'lpix_rms', 'mpix_rms', 'num_samps', 'centl', 'centm'])

    # Get one of the beams.
    obsi = InjectionResults(sbid, collated_data['beam'].iloc[0], run, clustering_dir=clustering_dir, log_file=log_file)

    # Exclude injections at times where all channels are flagged.
    no_data = collated_data['masked'] == 0.
    print(f"{no_data.sum()} injections were outside the file or in a dropout.", file=log_file)
    collated_data = collated_data[~no_data]

    # Exclude too high DMs
    obsi.yaml_path = path_if_exists(obsi.get_yaml_path())
    obsi.calculate_dm_pccm3()
    too_high_dm = collated_data['dm_pccm3_inj'] > obsi.dm_pccm3
    print(f"{too_high_dm.sum()} injections had a DM higher than the searched DM.", file=log_file)
    collated_data = collated_data[~too_high_dm]

    # Pick only the highest SNR candidates from every injection, but prefer non-RFI ones.
    snr_discard = 9
    highest_snr = collated_data.groupby(['beam', 'INJ_name'])['snr'].transform('max') == collated_data['snr']
    selected_candidates = collated_data.groupby(['beam', 'INJ_name'], sort=False).apply(lambda g: select_candidate(g, snr_discard))
    selected_candidates = selected_candidates.explode().astype(bool).to_numpy()
    print(f"{(~selected_candidates).sum()} injections were side clusters.", file=log_file)
    n_groups_eq_sel_cands = collated_data.groupby(['beam', 'INJ_name'], sort=False).ngroups == selected_candidates.sum()
    multiple_cands_per_group = np.any(collated_data[selected_candidates].groupby(['beam', 'INJ_name'])['snr'].count() != 1)
    if not n_groups_eq_sel_cands or multiple_cands_per_group:
        print("Problem with the candidate selection.", file=log_file)
    collated_data = collated_data[selected_candidates]
    collated_data["also_in_rfi"] = (selected_candidates & ~highest_snr)[selected_candidates]

    # See if it is marked as a known source.
    collated_data['known_source'] = ~collated_data[['PSR_name', 'PSR_sep', 'RACS_name', 'RACS_sep', 'NEW_name', 'NEW_sep', 'ALIAS_name', 'ALIAS_sep']].isna().all(axis=1)

    collated_data = collated_data.sort_values(['beam', 'total_sample_inj']).reset_index()

    collated_data['SNR/SNR_inj'] =  collated_data['snr'] / collated_data['SNR_inj'] / np.sqrt(collated_data['masked'])

    # Fit a beam model to the lpix, mpix data.
    not_seen = collated_data['missed'] & collated_data['snr'].isna()
    data = collated_data[~not_seen]
    multiple_lpix = not np.all(collated_data['lpix_inj'] == collated_data.loc[0, 'lpix_inj'])
    multiple_mpix = not np.all(collated_data['mpix_inj'] == collated_data.loc[0, 'mpix_inj'])
    if multiple_lpix and multiple_mpix:
        beams, bestfit, uncert = fit_beams(data)
    elif not multiple_lpix and not multiple_mpix:
        print("No variation in lpix nor mpix. Not fitting the beam. Resorting to default beam shape.", file=log_file)
        beams = collated_data['beam'].unique()
        bestfit = np.array(len(beams)*[[0.9, 0.6, 128, 128]])
        uncert = None
    elif multiple_mpix:
        beams, bestfit, uncert = fit_beams(data, fix_lpix0=True)
    elif multiple_lpix:
        beams, bestfit, uncert = fit_beams(data, fix_mpix0=True)

    for i, beam in enumerate(beams):
        lmpix = collated_data.loc[collated_data['beam'] == beam, ['lpix', 'mpix']].to_numpy()
        if multiple_mpix and not multiple_lpix:
            # Swap columns such that mpix fit will be use for mpix
            lmpix = lmpix[:,::-1]
        collated_data.loc[collated_data['beam'] == beam, 'recovery'] = beam_model(lmpix, *bestfit[i])

    collated_data['SNR_expected'] = collated_data['SNR_inj'] * collated_data['recovery'] * np.sqrt(collated_data['masked'])
    collated_data['SNR/SNR_expected'] = collated_data['snr'] / collated_data['SNR_expected']

    return collated_data, bestfit, uncert

def report_outcomes(collated_data, log_file=None):

    beam_numbers = np.sort(collated_data['beam'].unique())
    all_found = beam_numbers[~collated_data.groupby('beam')['missed'].any()]

    # Print which beams missed detections.
    print(f"{len(all_found)} beams had no missed injections. These are beams {list_to_str(all_found)}.", file=log_file)

    beams_missed_injs = collated_data.loc[collated_data['missed'], 'beam'].to_list()
    print(f"{len(beams_missed_injs)} injections have been missed. These are in beams "
          f"{list_to_str(beams_missed_injs)}.", file=log_file)
    print(f"{collated_data['known_source'].sum()} injections have been marked as known sources.", file=log_file)

    # Check misclassified classification.
    misclassified = collated_data['missed'] & (~collated_data['snr'].isna())

    # Missed statistics based on SNR.
    rfi = (misclassified & (collated_data['classification'] == 'rfi'))
    print("Classified as RFI:\n"
        fr"total: {rfi.sum()}/{collated_data.shape[0]}", file=log_file)
    if rfi.any():
        less7 = np.sum(rfi & (collated_data['snr'] <= 7))
        n_less7 = np.sum(collated_data['snr'] <= 7)
        less9 = np.sum(rfi & (collated_data['snr'] > 7) & (collated_data['snr'] <= 9))
        n_less9 = np.sum((collated_data['snr'] > 7) & (collated_data['snr'] <= 9))
        great9 = np.sum(rfi & (collated_data['snr'] > 9))
        n_great9 = np.sum(collated_data['snr'] > 9)
        print(f"SNR<=7: {less7}/{n_less7}\n"
            f"7<SNR<=9: {less9}/{n_less9}\n"
            f"SNR>9: {great9}/{n_great9}\n",
            file=log_file)

    # Known sources by SNR.
    known = collated_data['known_source']
    print(fr"Classified as a known source:", file=log_file)
    print(fr"total: {known.sum()}/{collated_data.shape[0]}", file=log_file)
    if known.any():
        less7 = np.sum(known & (collated_data['snr'] <= 7))
        n_less7 = np.sum(collated_data['snr'] <= 7)
        less9 = np.sum(known & (collated_data['snr'] > 7) & (collated_data['snr'] <= 9))
        n_less9 = np.sum((collated_data['snr'] > 7) & (collated_data['snr'] <= 9))
        great9 = np.sum(known & (collated_data['snr'] > 9))
        n_great9 = np.sum(collated_data['snr'] > 9)
        print(f"SNR<=7:{less7}/{n_less7}\n"
            f"7<SNR<=9: {less9}/{n_less9}\n"
            f"SNR>9: {great9}/{n_great9}",
            file=log_file)

    # Percent classified missed
    missed = collated_data['snr'].isna() | rfi | collated_data['known_source']
    print(f"Total missed: {np.sum(missed)}/{collated_data.shape[0]} = "
          f"{np.sum(missed)/collated_data.shape[0] * 100:.2f} %",
          file=log_file)

    # Save some numbers to disk.
    to_save = {'total' : len(collated_data),
               'missed' : missed.sum(),
               'rfi' : rfi.sum(),
               'known' : known.sum(),
               }

    return to_save


def injspect(sbid, run='inj', clustering_dir='clustering_output', fig_path=None):
    sbid = sbid_str(sbid)
    if not fig_path:
        fig_path = f"/data/craco/craco/jah011/{sbid}/{run}_{clustering_dir}"
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    with open(os.path.join(fig_path, "inj_report.txt"), 'w+') as log_file:
        print(f"injspection version: {version('injspection')}", file=log_file)
        collated_data, pixelfit, pixelfit_unc = collate_observation_data(sbid, run=run,
            clustering_dir=clustering_dir, log_file=log_file)
        to_save = report_outcomes(collated_data, log_file)
    pickle.dump([to_save, pixelfit, pixelfit_unc], open(os.path.join(fig_path, 'some_variables.pkl'), 'wb'))
    make_all_pretty_plots(collated_data, pixelfit, pixelfit_unc, fig_path=fig_path, sbid=sbid, run=run)
