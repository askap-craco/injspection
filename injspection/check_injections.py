import os
import warnings
import numpy as np
import pandas as pd
import pickle

from glob import glob
from scipy.optimize import curve_fit
from importlib.metadata import version

from craft import sigproc
from craft import uvfits

from .utils import sbid_str, path_if_exists, get_dm_pccm3, test_closest_inj, check_masked_channels, find_close_cands
from .utils import add_missed_cols_etc, classification_to_bools, list_to_str
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
        inj_pattern = os.path.join(obs_path_pattern, sbid, 'scans/', scan, scantime, run, clustering_dir, f'candidates.b{beam}.*rawcat.csv')
        rawcat_paths = glob(inj_pattern)
        if len(rawcat_paths) == 0:
            raise ValueError(f"No raw catalog found at {inj_pattern}.")
        elif len(rawcat_paths) > 1:
            warnings.warn(f"{len(rawcat_paths)} directories matching the pattern have been found. Continuing with {rawcat_paths[0]}")
        self.raw_cand_file = rawcat_paths[0]
        clustering_path = os.path.dirname(rawcat_paths[0])
        inj_path = os.path.dirname(clustering_path)

        # Get all the useful paths
        self.sbid = sbid
        self.beam = beam
        self.tsamp = tsamp
        self.run_path = inj_path
        self.log_path = os.path.join(inj_path, f'search_pipeline_b{beam}.log')
        self.log_file = log_file
        self.clustering_path = clustering_path
        self.orig_inj_path = path_if_exists(os.path.join(clustering_path, f'candidates.b{beam}.txt.inject.orig.csv'))
        self.found_inj_path = path_if_exists(os.path.join(clustering_path, f'candidates.b{beam}.txt.inject.cand.csv'))
        # self.raw_cand_file = path_if_exists(os.path.join(clustering_path, f'candidates.b{beam}.rawcat.csv'))
        self.uniq_path = path_if_exists(os.path.join(clustering_path, f'candidates.b{beam}.uniq.csv'))
        self.rfi_path = path_if_exists(os.path.join(clustering_path, f'candidates.b{beam}.rfi.csv'))
        # self.crossmatch_path = path_if_exists(os.path.join(clustering_path, f'candidates.b{beam}.txt.catalog_cross_match.i1.csv'))
        # self.alias_path = path_if_exists(os.path.join(clustering_path, f'candidates.b{beam}.txt.alias_filter.i2.csv'))
        self.pcb_path = path_if_exists(os.path.join(inj_path, f'pcb{beam}.fil'))
        self.uvfits_path = path_if_exists(os.path.join(os.path.dirname(inj_path), f'b{beam}.uvfits'))

        # Check path with ".txt" of older pipeline versions for backwards compatibility.
        # if self.raw_cand_file is None:
        #     self.raw_cand_file = path_if_exists(os.path.join(clustering_path, f'candidates.b{beam}.txt.rawcat.csv'))
        if self.uniq_path is None:
            self.uniq_path = path_if_exists(os.path.join(clustering_path, f'candidates.b{beam}.txt.uniq.csv'))
        if self.rfi_path is None:
            self.rfi_path = path_if_exists(os.path.join(clustering_path, f'candidates.b{beam}.txt.rfi.csv'))

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
            print(f"Warning: The pcb file of beam {self.beam} is only {length_ratio} of the uvfits file. pcb: {fil_length}, "
                  f"uvfits: {fits_length}. Better check the log mate.", file=self.log_path)

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
        self.compare_uvfits_fil_lengths()
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


def check_candpipe_files(orig_file, found_file, obsi, print_missed=False, num_injs=None, log_file=None):
    """Compare the .inject.orig.csv with the .inject.cand.csv file.

    Search the raw candidates for missing candidates and collect all in a DataFrame.
    """
    orig_injs = pd.read_csv(orig_file)

    # Exclude candidates beyond the filelength.
    fil_length = obsi.get_fil_length()
    in_obsi = orig_injs['total_sample_inj'] < fil_length
    orig_injs = orig_injs[in_obsi]
    print(f"Beam {obsi.beam}: PCB length {fil_length}, injections in length: {in_obsi.sum()}, outside:  {(~in_obsi).sum()}", file=log_file)

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
    missed_injs['classification'] = 'missed'
    found_injs['classification'] = 'inj'

    # See if found sources are marked as a known sources.
    if 'MATCH_name' in found_injs.columns:
        known_source = ~found_injs['MATCH_name'].isna()
    else:  # old source logic
        known_source = ~found_injs[['PSR_name', 'PSR_sep', 'RACS_name', 'RACS_sep', 'NEW_name', 'NEW_sep', 'ALIAS_name', 'ALIAS_sep']].isna().all(axis=1)
    found_injs.loc[known_source, 'classification'] = 'known_source'

    # Load catalogs.
    # if not missed_injs.empty:
    uniq_cands = pd.read_csv(obsi.uniq_path, index_col=0)
    rfi_cands = pd.read_csv(obsi.rfi_path, index_col=0)
    raw_loaded = False
    uniq_and_rfi_cands = []

    # Search for side clusters of known sources.
    for mj in found_injs[known_source].index:
        missed_inj = found_injs.loc[mj]
        close_uniqs = find_close_cands(uniq_cands, missed_inj)
        if not close_uniqs.empty:
            uniq_and_rfi_cands.append(add_missed_cols_etc(close_uniqs, missed_inj, found_in='uniq'))
        close_rfis = find_close_cands(rfi_cands, missed_inj)
        if not close_rfis.empty:
            uniq_and_rfi_cands.append(add_missed_cols_etc(close_rfis, missed_inj, found_in='rfi'))

    # Search uniq, rfi, and raw candidates for missed ones and save their properties.
    for mj in missed_injs.index:
        missed_inj = missed_injs.loc[mj]
        close_uniqs = find_close_cands(uniq_cands, missed_inj)
        if not close_uniqs.empty:
            uniq_and_rfi_cands.append(add_missed_cols_etc(close_uniqs, missed_inj, found_in='uniq'))

        close_rfis = find_close_cands(rfi_cands, missed_inj)
        if not close_rfis.empty:
            uniq_and_rfi_cands.append(add_missed_cols_etc(close_rfis, missed_inj, found_in='rfi'))

        if close_uniqs.empty and close_rfis.empty:
            if not raw_loaded:
                raw_cands = pd.read_csv(obsi.raw_cand_file, index_col=0)
                raw_cands = raw_cands[~raw_cands['cluster_id'].isin(found_injs['cluster_id'])]
                raw_loaded = True
            close_raws = find_close_cands(raw_cands, missed_injs.loc[mj])
            if not close_raws.empty:
                max_snr = close_raws['snr'].idxmax()
                missed_injs.loc[mj, close_raws.columns] = close_raws.loc[max_snr]
                missed_injs.loc[mj, 'n_clusters'] = len(close_raws['cluster_id'].unique())
                missed_injs.loc[mj, 'classification'] = 'raw'
                missed_injs.loc[mj, 'raw_and'] = obsi.search_classification(close_raws.loc[max_snr, 'cluster_id'])
                # Indexing with lists avoids conversion to Series. But doesn't work because index is different.
        else:
            # Exclude the raws that were found.
            missed_injs = missed_injs.drop(index=mj)

    injs = pd.concat([df for df in [found_injs, missed_injs, *uniq_and_rfi_cands] if not df.empty], ignore_index=True)
    injs = injs.sort_values('total_sample_inj').reset_index(drop=True)
    injs['beam'] = obsi.beam_int
    injs['missed'] = injs['missed'].astype(bool)

    # See again if any injection is marked as a known source.
    if 'MATCH_name' in injs.columns:
        known_source = ~injs['MATCH_name'].isna()
    elif 'PSR_name' in injs.columns:  # old source logic
        known_source = ~injs[['PSR_name', 'PSR_sep', 'RACS_name', 'RACS_sep', 'NEW_name', 'NEW_sep', 'ALIAS_name', 'ALIAS_sep']].isna().all(axis=1)
    # else found_injs was empty
    if np.any(known_source):
        injs.loc[known_source, 'classification'] = 'known_source'

    return injs


def get_injection_results(sbid, run='inj', obs_path_pattern='/CRACO/DATA_??/craco/', scan_pattern='scans/??/*/',
                          clustering_dir='clustering_output', log_file=None):
    """For the given injection run get the found and missed candidates."""
    sbid = sbid_str(sbid)
    # Define Data locations to be searched.
    inj_pattern = os.path.join(obs_path_pattern, sbid, scan_pattern, run)

    # Get beam numbers that were used in this observation (usually 00 to 35).
    clustering_files = sorted(glob(os.path.join(inj_pattern, clustering_dir, 'candidates.b??.*rawcat.csv')))

    collated_data = []

    for file in clustering_files:
        scan = file[file.find('scans/')+6 : file.find('scans/') + 8]
        scantime = file[file.find('scans/')+9 : file.find('scans/') + 23]
        beam = file[file.find('candidates.b')+12 : file.find('candidates.b') + 14]

        obsi = InjectionResults(sbid, beam, scan=scan, run=run, scantime=scantime, clustering_dir=clustering_dir,
                                log_file=log_file)
        try:
            collated_data.append(obsi.do_all_checks())
        except ValueError as err:
            print(err)
            print(f"{sbid}, {beam}, {scan}, {run}, {scantime}, {clustering_dir}, {obsi.found_inj_path}")

    # Pick only the highest SNR candidates from every injection, but prefer non-RFI ones. Moved here for multi scan obsis.
    snr_discard = 9
    side_count = 0
    for i, sfd in enumerate(collated_data):  # "single file data"
        inj_group = sfd.groupby('INJ_name', sort=False)
        highest_snr = ((inj_group['snr'].transform('max') == sfd['snr'])
                        | sfd['snr'].isna())  # the or is for Missed cands without detection, i.e. nans
        selected_candidates = highest_snr
        # # Apply throws an error in rare cases (beam 35 of 64401), maybe because the first group consists of several rows.
        # try:
        #     selected_candidates = inj_group.apply(lambda g: select_candidate(g, snr_discard))
        #     selected_candidates = selected_candidates.explode().astype(bool).to_numpy()
        # except (AttributeError, TypeError):
        #     # Do the same with a loop.
        #     selected_candidates = []
        #     for g in inj_group:
        #         sc = select_candidate(g[1], snr_discard)
        #         if isinstance(sc, pd.Series):
        #             selected_candidates += sc.to_list()
        #         else:
        #             selected_candidates.append(sc)
        #     selected_candidates = np.array(selected_candidates)

        # Only keep highest SNR candidate. Check if it would have been found through a side cluster.
        sfd_sel = sfd[selected_candidates].copy()
        sfd_sel['detected_in_side'] = False
        any_in_inj_uniq = inj_group.apply(lambda g: np.any((g['snr'] > snr_discard)
            & ((g['classification'] == 'inj') | (g['classification'] == 'uniq'))))
        sfd_sel['detected_in_side'] = (any_in_inj_uniq & (sfd_sel['classification'] != 'inj')
                                       & (sfd_sel['classification'] != 'uniq'))
        # sfd_sel['also_in_rfi'] = (selected_candidates & ~highest_snr)[selected_candidates]
        collated_data[i] = sfd_sel

        # Report and test for consistency.
        side_count += (~selected_candidates).sum()
        n_groups_eq_sel_cands = inj_group.ngroups == selected_candidates.sum()
        cand_count = sfd_sel.groupby('INJ_name')['SNR_inj'].count()
        multiple_cands_per_group = np.any(cand_count != 1)
        if not n_groups_eq_sel_cands or multiple_cands_per_group:
            print("Problem with the candidate selection.", file=log_file)
            print(cand_count[cand_count != 1], file=log_file)

    print(f"{side_count} injections were side clusters.", file=log_file)

    collated_data = pd.concat(collated_data)

    return collated_data


def beam_model(lmpix, eta=0.9, del_l=1., lpix0=127.5, mpix0=127.5):
    """Loss in SNR due to efficiency and pixelization.

    lpix, mpix (int, float, or array): Pixel position
    eta (float): Efficiency at beam center.
    phi (float): Fraction of the total primary beam covered by the pixel grid.
    lpix0, mpix0 (int or float): Beam center.
    """
    lpix, mpix = lmpix[:, 0], lmpix[:, 1]
    l = (lpix-lpix0)/256
    m = (mpix-mpix0)/256
    # Note that a factor pi is contained in numpys sinc, i.e. np.sinc(x)=sin(pi*x)/pi*x
    return eta*np.sinc(del_l*l)*np.sinc(del_l*m)


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
        pixels = beam_data[['lpix', 'mpix']].dropna().astype(np.float64).to_numpy()
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
        'mSlopeSmooth', 'num_samps'])  # vanished: 'lpix_rms', 'mpix_rms', , 'centl', 'centm'

    # Exclude injections at times where all channels are flagged.
    no_data = collated_data['masked'] == 0.
    print(f"{no_data.sum()} injections were outside the file or in a dropout.", file=log_file)
    collated_data = collated_data[~no_data]

    # Exclude too high DMs. Get yaml file from one of the beams.
    obsi = InjectionResults(sbid, collated_data['beam'].iloc[0], run, clustering_dir=clustering_dir, log_file=log_file)
    obsi.yaml_path = path_if_exists(obsi.get_yaml_path())
    obsi.calculate_dm_pccm3()
    too_high_dm = collated_data['dm_pccm3_inj'] > obsi.dm_pccm3
    print(f"{too_high_dm.sum()} injections had a DM higher than the searched DM.", file=log_file)
    collated_data = collated_data[~too_high_dm]

    collated_data = collated_data.sort_values(['beam', 'total_sample_inj']).reset_index()

    collated_data['SNR/SNR_inj'] =  collated_data['snr'] / collated_data['SNR_inj'] / np.sqrt(collated_data['masked'])

    # Fit a beam model to the lpix, mpix data.
    seen = collated_data['classification'] != 'missed'
    data = collated_data[seen]
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
        lmpix = collated_data.loc[collated_data['beam'] == beam, ['lpix_inj', 'mpix_inj']].to_numpy()
        if multiple_mpix and not multiple_lpix:
            # Swap columns such that mpix fit will be used for mpix
            lmpix = lmpix[:,::-1]
        collated_data.loc[collated_data['beam'] == beam, 'recovery'] = beam_model(lmpix, *bestfit[i])

    collated_data['SNR_expected'] = collated_data['SNR_inj'] * collated_data['recovery'] * np.sqrt(collated_data['masked'])
    collated_data['SNR/SNR_expected'] = collated_data['snr'] / collated_data['SNR_expected']

    return collated_data, bestfit, uncert


def report_outcomes(collated_data, log_file=None):

    beam_numbers = np.sort(collated_data['beam'].unique())
    uniq, found, missed, rfi, known, raw, side, would_missed = classification_to_bools(collated_data)

    if np.all(found ^ missed ^ rfi ^ known ^ raw ^ side):
        print("Classifications are consistent.", file=log_file)
    else:
        print("Problem with classification.", file=log_file)

    all_found = beam_numbers[collated_data.groupby('beam').apply(lambda g: (g['classification'] == 'uniq')
        | (g['classification'] == 'inj')).all()]

    # Print which beams missed detections.
    print(f"{len(all_found)} beams had no missed injections. These are beams {list_to_str(all_found)}.", file=log_file)

    # Uniques that would have triggered online, not including side clusters.
    print(f"Classified as unique: {np.sum(uniq)}", file=log_file)

    # beams_missed_injs = collated_data.loc[collated_data['missed'], 'beam'].to_list()
    print(f"{np.sum(missed)} injections have been missed.", file=log_file) #len(beams_missed_injs) These are in beams "
        #   f"{list_to_str(beams_missed_injs)}."
    print(f"{np.sum(known)} injections have been marked as known sources.", file=log_file)

    # Missed statistics based on SNR.
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
        collated_data[rfi & collated_data['snr'] > 9]

    # Known sources by SNR.
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

    # Only found in the raw candidates.
    print(f"Only found in the raw candidates: {raw.sum()}", file=log_file)

    # Recovered through side clusters.
    print(f"Recovered through side clusters: {side.sum()}", file=log_file)

    # Percent classified missed.
    print(f"Total missed: {np.sum(would_missed)}/{collated_data.shape[0]} = "
          f"{np.sum(would_missed)/collated_data.shape[0] * 100:.2f} %",
          file=log_file)
    # Would missed that are expected to be detected.
    should_detect = collated_data['SNR_expected'] > 9
    print(f"Total missed with SNR_expected > 9: {np.sum(should_detect & would_missed)}/{should_detect.sum()} = "
          f"{np.sum(should_detect & would_missed)/should_detect.sum() * 100:.2f} %",
          file=log_file)


    # Save some numbers to disk.
    to_save = {'total' : len(collated_data),
               'found' : found.sum(),
               'uniq' : uniq.sum(),
               'missed' : missed.sum(),
               'rfi' : rfi.sum(),
               'known' : known.sum(),
               'raw' : raw.sum(),
               'side' : side.sum(),
               'wouldmissed>9' : np.sum(should_detect & would_missed),
               'should_detect' : should_detect.sum()
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
    collated_data.to_csv(os.path.join(fig_path, 'collated_data.csv'))
    make_all_pretty_plots(collated_data, pixelfit, pixelfit_unc, fig_path=fig_path, sbid=sbid, run=run)
