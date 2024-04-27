import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from glob import glob

from craft import sigproc
from craft import uvfits

from .create_injection_params_file import total_burst_length


class InjectionResults:
    """Make all the path handling easier by saving them in a class"""

    def __init__(self, obs_id, beam='00', run='inj', scan='00', scantime='??????????????',
                 clustering_dir='clustering_output', tsamp=0.013824):
        """Define all the paths

        Parameters:
            obs_id (str): Observation ID, e.g. 'SB058479'.
            beam (str or int): Beam number
            run (str): The name of the run directory, e.g. 'results'.
            scan (str): Name of scan directory.
            scan_pattern (str): The pattern to search for the run directory.
            clustering_dir (str): Directory with the results of the clustering
            tsamp (float): Sampling time (s). This is not in the logs.
        """
        # Allow for the five digit SB number only.
        if len(str(obs_id)) == 5:
            obs_id = 'SB0' + str(obs_id)

        # Make beam possible as int and str.
        if isinstance(beam, int):
            self.beam_int = beam
            beam = str(beam)
        else:
            self.beam_int = int(beam)
        if len(beam) == 1:
            beam = '0' + beam

        # Define Data locations to be searched.
        obs_path_pattern = '/CRACO/DATA_??/craco/'
        inj_pattern = os.path.join(obs_path_pattern, obs_id, 'scans/', scan, scantime, run, f'search_pipeline_b{beam}.log')
        log_paths = glob(inj_pattern)
        if len(log_paths) > 1:
            warnings.warn(f"{len(log_paths)} directories matching the pattern have been found. Continuing with {log_paths[0]}")
        inj_path = os.path.dirname(log_paths[0])

        # Get all the useful paths
        self.beam = beam
        self.run_path = inj_path
        self.log_path = log_paths[0]
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
        self.yaml_path = path_if_exists(self.get_yaml_path())

        # Calculate maximum searched DM.
        fmin = self.fmin
        fmax = fmin + self.foff*self.nchan
        self.dm_pccc = get_dm_pccc([fmin, fmax], self.ndm, tsamp)


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
        self.compare_uvfits_fil_lengths()
        log_injs = self.count_log_injections()
        injs = check_candpipe_files(self.orig_inj_path, self.found_inj_path, self, num_injs=log_injs)
        injs[~injs['missed']] = count_found_clusters(found_injs=injs[~injs['missed']], raw_cand_file=self.raw_cand_file)
        #injs = add_yaml_parameters(injs, "/CRACO/DATA_00/craco/injection/inj_params7_pc_width.yml") #self.yaml_path)
        injs = check_masked_channels(injs, filpath=self.pcb_path)

        return injs


def path_if_exists(path):
    if not os.path.exists(path):
        path = None
    return path


def get_dm_pccc(freqs, dm_samps, tsamp):
    '''Stolen from
    freqs in Hz
    tsamp in s
    '''
    delay_s = dm_samps * tsamp
    fmax = np.max(freqs) * 1e-9
    fmin = np.min(freqs) * 1e-9
    dm_pccc = delay_s / 4.15 / 1e-3 / (1 / fmin**2 - 1 / fmax**2)
    return dm_pccc


def check_candpipe_files(orig_file, found_file, obsi, print_missed=False, num_injs=None):
    """Compare the .inject.orig.csv with the .inject.cand.csv file.

    Search the raw candidates for missing candidates and collect all in a DataFrame.
    """
    orig_injs = pd.read_csv(orig_file)
    if num_injs and len(orig_injs) > num_injs:
        print("Less bursts have been injected according to the log than planned!")
        # orig_injs = orig_injs.loc[:num_injs]  # Feel like this is unsave (sometimes an injection gets picked up several times).

    found_injs = pd.read_csv(found_file)
    found_injs = found_injs[~found_injs['SNR'].isna()]  # Not needed depending on the day. Yuanming seems to be changing the pipeline output on a weekly basis.

    # found_injs = found_injs[found_injs['INJ_name'].isin(orig_injs['name'])]  Was workaround with line above

    # Crosscheck the time to make sure which injection was found. (Don't trust pipeline)
    closest_inj = np.argmin(np.abs(found_injs['total_sample'].to_numpy()[:, None]-orig_injs['total_sample'].to_numpy()), axis=1)
    found_injs['INJ_closest'] = orig_injs['name'].iloc[closest_inj].to_numpy()

    missed = ~ orig_injs['name'].isin(found_injs['INJ_closest'])
    missed_injs = orig_injs[missed]
    # Print a useful message.
    if not missed_injs.empty and print_missed:
        print(f"================ {obs_id}  Beam {obsi.beam_int} ================")
        print("Missed:")
        print(missed_injs)

    # Summarize useful data from all injections for plotting etc.
    missed_injs = missed_injs.rename(columns={'name':'INJ_name', 'total_sample':'total_sample_inj',
                                              'dm_pccm3':'dm_pccm3_inj'})
    missed_injs['missed'] = True
    found_injs['missed'] = False

    # Get the number of clusters in the raw candidates.
    raw_cands = pd.read_csv(obsi.raw_cand_file, index_col=0)
    injs = found_injs  # in case of no missed injections.

    # Search raw candidates for missed ones and save their properties.
    for mj in missed_injs.index:
        missed_inj = missed_injs.loc[mj]

        close_raws = find_close_cands(raw_cands, missed_inj)
        if close_raws.empty:
            if print_missed:
                print("No raw candidades.")
            injs.loc[mj] = missed_inj
        else:
            if print_missed:
                print("Raw candidates:")
                print(close_raws)
            max_snr = close_raws['SNR'].idxmax()
            injs.loc[mj, close_raws.columns] = close_raws.loc[max_snr]  # Indexing with lists avoids conversion to Series. But doesn't work because index is different.
            injs.loc[mj, ['INJ_name', 'total_sample_inj', 'dm_pccm3_inj', 'snr', 'missed']] = missed_inj[['INJ_name', 'total_sample_inj', 'dm_pccm3_inj', 'snr', 'missed']]
            injs.loc[mj, 'n_clusters'] = len(close_raws['cluster_id'].unique())
            injs.loc[mj, 'classification'] = obsi.search_classification(close_raws.loc[max_snr, 'cluster_id'])

    injs['beam'] = obsi.beam_int
    injs['missed'] = injs['missed'].astype(bool)

    return injs

def count_found_clusters(found_injs, raw_cand_file):
    """Count the clusters of found injections.

    This has been outsourced cause it is time consuming.
    """
    raw_cands = pd.read_csv(raw_cand_file, index_col=0)
    found_injs = found_injs.copy()  # Avoid setting on copy warnings.
    for i in found_injs.index:
        close_raws = find_close_cands(raw_cands, found_injs.loc[i])
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
                          clustering_dir='clustering_output'):
    """For the given injection run get the found and missed candidates."""
    # Define Data locations to be searched.
    inj_pattern = os.path.join(obs_path_pattern, sbid, scan_pattern, run)

    # Get beam numbers that were used in this observation (usually 00 to 35).
    log_files = sorted(glob(os.path.join(inj_pattern, 'search_pipeline_b??.log')))

    collated_data = []

    for file in log_files:
        scan = file[file.find('scans/')+6 : file.find('scans/') + 8]
        scantime = file[file.find('scans/')+9 : file.find('scans/') + 23]
        beam = file[file.find('pipeline_b')+10 : file.find('pipeline_b') + 12]

        obsi = InjectionResults(sbid, beam, scan=scan, run=run, scantime=scantime, clustering_dir=clustering_dir)
        collated_data.append(obsi.do_all_checks())

    collated_data = pd.concat(collated_data)

    return collated_data


def find_close_cands(raw_cands, inj, space_check=True):
    if isinstance(inj, pd.DataFrame):
        inj = inj.squeeze()  # Avoid error.

    before, after = total_burst_length(inj['dm_pccm3_inj'], width=0, bonus=64)

    close_in_time = ((raw_cands['total_sample'] > (inj['total_sample_inj'] - before))
                     & (raw_cands['total_sample'] < (inj['total_sample_inj'] + before)))
    raw_cands = raw_cands[close_in_time]

    # close_in_dm = ((raw_cands['dm_pccm3'] > (inj['dm_pccm3_inj'] - dm_dist))
    #                & (raw_cands['dm_pccm3'] < (inj['dm_pccm3_inj'] + dm_dist)))
    if space_check:
        close_in_space = np.sqrt((raw_cands['lpix']-inj['lpix'])**2 + (raw_cands['mpix']-inj['mpix'])**2) < 5
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


def plot_injection_param(injected, detected, parameter_name):
    """For one parameter plot the injected against the detected signal.

    Return a figure with two axis, the second holding the difference between injected and detected parameters.
    """
    height_ratios = (2, 1)
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.04, 'height_ratios' : (2, 1)},
                                  figsize=(4.8, 6.4))
    ax.plot(injected, detected, '.')
    ax2.plot(injected, detected-injected, '.')

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

    #fig.tight_layout()

    return fig


if __name__ == '__main__':
    obs_id =  "SB058479"  #"SB057623"  #"SB057472"
    fig_path='/data/craco/craco/jah011/'
    run='inj1'

    collated_data = get_injection_results(obs_id, run)

    beam_numbers = np.sort(collated_data['beam'].unique())
    all_found = beam_numbers[~collated_data.groupby('beam')['missed'].any()]

    # Print which beams missed detections.
    print(f"{len(all_found)} beams had no missed injections. These are beams {list_to_str(all_found)}.")

    beams_missed_injs = collated_data.loc[collated_data['missed'], 'beam'].to_list()
    print(f"{len(beams_missed_injs)} injections have been missed. These are in beams {list_to_str(beams_missed_injs)}.")

    # Print differing pixel positions.
    # print(collated_data[(collated_data['lpix'] != 100) & (collated_data[collated_data['mpix'] != 100])])

    # collated_data[collated_data['INJ_name']=='INJ_4']
    # # print(collated_data.groupby('INJ_name')['SNR'].apply(np.median))

    # collated_data['SNR/snr'] = collated_data['SNR'] / collated_data['snr']
    # # print(collated_data.groupby('INJ_name')['SNR/snr'].apply(np.median))

    # # Plot SNRs.
    # detected, injected = collated_data['snr'], collated_data['SNR'].fillna(5)
    # parameter_name = "SNR"
    # fig = plot_injection_param(detected, injected, parameter_name)
    # fig.savefig(fig_path + f"injection_SNR_{obs_id}_{run}.png")

    # # Plot DMs.
    # detected, injected = collated_data['dm_pccm3_inj'], collated_data['dm_pccm3']
    # parameter_name = "DM (pc/cm$^3$)"
    # fig = plot_injection_param(detected, injected, parameter_name)
    # fig.savefig(fig_path + f"injection_DM_{obs_id}.png")

    # # Plot lpix.
    # detected, injected = collated_data['lpix'], collated_data['lpix']
    # parameter_name = "lpix"
    # fig = plot_injection_param(detected, injected, parameter_name)
    # fig.savefig(f"injection_lpix_{obs_id}.png")

    # # Plot mpix.
    # detected, injected = collated_data['mpix'], collated_data['mpix']
    # parameter_name = "mpix"
    # fig = plot_injection_param(detected, injected, parameter_name)
    # fig.savefig(f"injection_mpix_{obs_id}.png")