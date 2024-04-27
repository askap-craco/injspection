import os
import numpy as np
import yaml

def main(num=1, rep=1, snr=[16, 16], dm=[200, 200], dm_samps=None, width_samps=[2, 2], width=None,
         inj_upix=[100, 100], inj_vpix=[100, 100], tstamp=None, tsep=None, tstart=0, t_res=None,
         blockoff=0, blockstep=1, toff=128, nt=256, subsample_phase=[0.5, 0.5],
         add_noise=False, seed=None, **kwargs):
    """Create an injection blueprint for Furby.

    Make a yaml file with property ranges, based on the command line input.
    """
    num_tot = num * rep

    # Set positions in pixel coordinates. or Sky
    inj_upixs = np.linspace(inj_upix[0], inj_upix[1], num)
    inj_vpixs = np.linspace(inj_vpix[0], inj_vpix[1], num)
    coord_val = [[float(pix_tuple[0]), float(pix_tuple[1])] for pix_tuple in zip(inj_upixs, inj_vpixs)]  # yaml cannot read its own written tuples uff. Also not numpy.
    coord_key = 'injection_pixels'

    # Set the range of widths either in s or samples.
    if width_samps is None:
        width_val = np.linspace(width[0], width[1], num)
        width_key = 'width'
    else:
        width_val = np.linspace(width_samps[0], width_samps[1], num)
        width_key = 'width_samps'

    # Set the DM range in DM units or samples.
    if dm_samps is None:
        dm_val = np.linspace(dm[0], dm[1], num)
        dm_key = 'dm'
    else:
        dm_val = np.linspace(dm_samps[0], dm_samps[1], num)
        dm_key = 'dm_samps'

    # Set injection times from inputs.
    if tstamp:
        inj_tstamps = np.linspace(tstamp[0], tstamp[1], num_tot, dtype=int)
    elif tsep:
        # Calculate individual separations between injections.
        assert dm_key == 'dm' and width_key == 'width_samps'
        before, after = total_burst_length(dm_val, width_val, bonus=tsep, t_res=t_res)
        before[1:] += after[:-1]
        inj_tstamps = tstart + before
    else:
        inj_tstamps = (blockoff + np.arange(num_tot) * blockstep) * nt + toff

    # Set all remaining parameters
    inj_snrs = np.linspace(snr[0], snr[1], num)
    inj_taus = np.full(num, 1e-16)
    inj_spectra = [r'flat'] * num
    inj_shapes = [r'tophat'] * num
    inj_noise_per_sample = np.full(num, 512)  # Hardcode the value of 512 in agreement with Andy.
    inj_subsample_phases = np.linspace(subsample_phase[0], subsample_phase[1], num)

    params = {
            'injection_tsamps' : inj_tstamps.tolist(),
            coord_key : rep * coord_val,
            'furby_props' : rep * [
                                {'snr': float(inj_snrs[ii]),
                                 width_key: float(width_val[ii]),
                                 dm_key: float(dm_val[ii]),
                                 'tau0': float(inj_taus[ii]),
                                 'spectrum' : str(inj_spectra[ii]),
                                 'shape': str(inj_shapes[ii]),
                                 'noise_per_sample': float(inj_noise_per_sample[ii]),
                                 'subsample_phase':float(inj_subsample_phases[ii])
                                 } for ii in range(num)
                            ],
            # Forward the noise and its seed into the file.
            'add_noise' : add_noise,
            'seed': seed,
            }

    return params


def randomized_injections(num=1, rep=1, rng=None, snr=[16, 16], dm=[200, 200], dm_samps=None, width_samps=[2, 2], width=None,
                          inj_upix=[100, 100], inj_vpix=[100, 100], tstamp=None, tsep=None, tstart=0, t_res=None,
                          blockoff=0, blockstep=1, toff=128, nt=256, subsample_phase=[0.5, 0.5],
                          add_noise=False, noise_seed=None, **kwargs):
    """Create an injection blueprint for Furby.

    Make a yaml file with property ranges, based on the command line input.
    """
    if rng is None:
        rng = np.random.default_rng()

    num_tot = num * rep

    # Set positions in pixel coordinates. or Sky
    inj_upixs = rng.integers(inj_upix[0], inj_upix[1], num, endpoint=True)
    inj_vpixs = rng.integers(inj_vpix[0], inj_vpix[1], num, endpoint=True)
    coord_val = [[float(pix_tuple[0]), float(pix_tuple[1])] for pix_tuple in zip(inj_upixs, inj_vpixs)]  # yaml cannot read its own written tuples uff. Also not numpy.
    coord_key = 'injection_pixels'

    # Set the range of widths either in s or samples.
    if width_samps is None:
        width_val = rng.uniform(width[0], width[1], num)
        width_key = 'width'
    else:
        width_val = rng.uniform(width_samps[0], width_samps[1], num)
        width_key = 'width_samps'

    # Set the DM range in DM units or samples.
    if dm_samps is None:
        dm_val = rng.uniform(dm[0], dm[1], num)
        dm_key = 'dm'
    else:
        dm_val = rng.uniform(dm_samps[0], dm_samps[1], num)
        dm_key = 'dm_samps'

    # Set injection times from inputs.
    if tstamp:
        inj_tstamps = np.linspace(tstamp[0], tstamp[1], num_tot, dtype=int)
    elif tsep:
        # Calculate individual separations between injections.
        assert dm_key == 'dm' and width_key == 'width_samps'
        before, after = total_burst_length(dm_val, width_val, bonus=tsep, t_res=t_res)
        before[1:] += after[:-1]
        inj_tstamps = tstart + np.cumsum(before * 1.1)
        inj_tstamps = inj_tstamps.astype(int)
    else:
        inj_tstamps = (blockoff + np.arange(num_tot) * blockstep) * nt + toff

    # Scale desired SNRs to detectable snrs
    desired_SNR = rng.uniform(snr[0], snr[1], num)
    inj_snrs = needed_snr(desired_SNR, lpix=inj_upixs, mpix=inj_vpixs, pc_effic=0.9, covered_beam=0.57)

    # Set all remaining parameters
    inj_taus = np.full(num, 1e-16)
    inj_spectra = [r'flat'] * num
    inj_shapes = [r'tophat'] * num
    inj_noise_per_sample = np.full(num, 512)  # Hardcode the value of 512 in agreement with Andy.
    inj_subsample_phases = rng.uniform(subsample_phase[0], subsample_phase[1], num)

    params = {
            'injection_tsamps' : inj_tstamps.tolist(),
            coord_key : rep * coord_val,
            'furby_props' : rep * [
                                {'snr': float(inj_snrs[ii]),
                                 width_key: float(width_val[ii]),
                                 dm_key: float(dm_val[ii]),
                                 'tau0': float(inj_taus[ii]),
                                 'spectrum' : str(inj_spectra[ii]),
                                 'shape': str(inj_shapes[ii]),
                                 'noise_per_sample': float(inj_noise_per_sample[ii]),
                                 'subsample_phase':float(inj_subsample_phases[ii])
                                 } for ii in range(num)
                            ],
            # Forward the noise and its seed into the file.
            'add_noise' : add_noise,
            'seed': noise_seed,
            }

    return params


def needed_snr(SNR, lpix, mpix, pc_effic=0.9, covered_beam=0.57):
    pc_pos = 128
    w = covered_beam
    snr = SNR/pc_effic/np.cos(np.pi*w*(lpix-pc_pos)/256)/np.cos(np.pi*w*(mpix-pc_pos)/256)

    return snr


def combine_injections(parameters):
    """Combine list of dicts into one dict"""
    injection_tsamps = []
    furby_props = []
    injection_pixels = []
    for params in parameters:
        injection_tsamps += params['injection_tsamps']
        injection_pixels += params['injection_pixels']
        furby_props += params['furby_props']

    combined = {
        'injection_tsamps' : injection_tsamps,
        'injection_pixels' : injection_pixels,
        'furby_props' : furby_props,
        'add_noise' : params['add_noise'],
        'seed': params['seed'],
    }

    return combined


def check_injection_times(params, t_res=13.8):
    """Check that no bursts overlap. Otherwise pipeline would throw errors."""
    inj_tstamps = np.array(params['injection_tsamps'])
    furby_props = params['furby_props']

    # Read in the parameters from the parameter list.
    widths = []
    dms = []
    for props in furby_props:
        if 'width' in props.keys():
            widths.append(props['width']*t_res)
        else:
            widths.append(props['width_samps'])

        if 'dm' in props.keys():
            dms.append(props['dm'])
        else:
            raise ValueError("No 'dm' key found. If the DM is not given in physical units correct burst separations cannot be guaranteed.")

    widths = np.array(widths)
    dms = np.array(dms)

    before, after = total_burst_length(dms, widths, bonus=30, t_res=t_res)
    sort_time = np.argsort(inj_tstamps)

    before, after = before[sort_time], after[sort_time]
    inj_tstamps = inj_tstamps[sort_time]

    before[1:] += after[:-1]
    before *= 1.1  # Add another 10% to be sure.
    tdiffs = np.diff(inj_tstamps, prepend=0)
    if not np.all(tdiffs > before):
        raise ValueError("The times between injections have to be larger than their duration, which is not the case "
                         f"for injections {np.nonzero(~(tdiffs > before))}, with time differences {tdiffs[~(tdiffs > before)]} but requiring {before[~(tdiffs > before)]}.")


def dm_sweep_length(dm, freq_bot=700, freq_top=1000):
    """Calculate the time of the maximum sweep to keep injections in the file. freqs in MHz

    The frequencies are the ones given on ASKAPS webpage. RACS e.g. uses 744MHz to 1032MHz,
    but this should give a good upper limit.
    """
    a = 4.14881e6  # ms. cause thats what furby uses.
    delay = a * dm * (freq_bot**(-2)-freq_top**(-2))
    return delay


def total_burst_length(dm, width, bonus=0, t_res=13.8):
    """Duration of the dispersed burst"""
    before = dm_sweep_length(dm, 700, 1000)/t_res + width/2 + bonus/2
    after = width/2 + bonus/2
    return before, after


if __name__ == '__main__':
    # # Test setup.
    # setups = [{'num' : 2, 'rep' : 2, 'snr' : (10, 50), 'tstamp' : (10, 15000)},
    #           {'num' : 3, 'rep' : 1, 'width_samps' : (0.5, 10), 'tstamp' : (15100, 30000)}]
    # Make a list of different setups to inject lots of pretty FRBs.
    # setups = ([{'num' : 21, 'rep' : 5, 'snr' : (10, 50), 'tstamp' : (100, 7500)},  # snr, keep tstamp >75 else it is outside file.
    #            {'num' : 22, 'rep' : 10, 'width_samps' : (0.5, 10), 'tstamp' : (7600, 15000)},  # width
    #            {'num' : 21, 'rep' : 10, 'dm' : (50, 1300), 'tstamp' : (15100, 22500)}  # DM
    #           ]
    #           + [{'num' : 256, 'rep' : 1, 'inj_upix' : (0, 255), 'inj_vpix' : (i*64, i*64), 'tstamp' : (22500+15000*i, 37500+15000*i)} for i in range(0, 3)]  # upix
    #           + [{'num' : 1, 'rep' : 1, 'inj_upix' : (300, 300), 'inj_vpix' : (100, 100), 'tstamp' : (22550, 22550)}]  # upix
    #          )
    # setups = ([#{'num' : 21, 'rep' : 5, 'snr' : (10, 50), },]  # snr, keep tstamp >75 else it is outside file.
    #         #    {'num' : 40, 'rep' : 5, 'width_samps' : (0.25, 10)}  # width
    #         #    {'num' : 21, 'rep' : 5, 'dm' : (50, 1500), 'blockstep' : 3}  # DM
    #           ]
    #           + [{'num' : 100, 'rep' : 1, 'inj_upix' : (0, 255), 'inj_vpix' : (i*64, i*64), 'blockoff' : 100*i} for i in range(0, 3)]  # upix
    #           + [{'num' : 1, 'rep' : 1, 'inj_upix' : (300, 100), 'inj_vpix' : (100, 100), 'blockoff' : 300}]  # upix
    #          )
    # setups = ([{'num' : 271, 'rep' : 1, 'dm' : (0, 0), 'inj_upix' : (0, 270), 'inj_vpix' : (127, 127)}])  # upix 2
    # setups = ([{'num' : 136, 'rep' : 1, 'dm' : (0, 0), 'inj_upix' : (0, 270), 'inj_vpix' : (127, 127) , 'subsample_phase' : (0,0), 'width_samps' : (4, 4)}])  # upix 3
    setups = ([{'num' : 201, 'rep' : 1, 'dm' : (0, 0), 'inj_upix' : (128, 128), 'inj_vpix' : (128, 128),
                'subsample_phase' : (0,0), 'width_samps' : (1, 8), 'add_noise' : True, 'seed' : 42
                }])  # Beam center
    setups = ([{'num' : 201, 'rep' : 1, 'snr' : (40, 200)
                }])  # snr up to 200 lets go
    random_setup = {'num' : 100, 'tsep' : 100, 't_res' : 13.8, 'snr' : (9, 20), 'dm' : (10, 1200),
                    'inj_upix' : (0, 255), 'inj_vpix' : (0, 255), 'width_samps' : (1, 8), 'subsample_phase' : (0,1)}  # 'blockstep' : 5, doesn't work
    # DM >~811 cannot be detected.
    random_setup = {'num' : 100, 'tsep' : 100, 't_res' : 13.8, 'snr' : (9, 20), 'dm' : (10, 800),
                    'inj_upix' : (0, 255), 'inj_vpix' : (0, 255), 'width_samps' : (1, 8), 'subsample_phase' : (0,1)}

    # Make the injection parameters with the given setup
    # injection_params = [main(**setup) for setup in setups]
    # params = combine_injections(injection_params)

    # Create a random paramameter file with the given random_setup.
    rng = np.random.default_rng(seed=42)
    params = randomized_injections(**random_setup, rng=rng)

    check_injection_times(params, t_res=13.8)

    print(os.path.dirname(os.path.abspath(__file__)))
    inj_dir = os.path.dirname(os.path.abspath(__file__))
    outfile = os.path.join(inj_dir, 'yaml_files/randomized_2_limit_DM.yml')
    with open(outfile, 'w') as f:
        yaml.dump(params, f)

    if yaml.safe_load(open(outfile, 'r')) == params:
        print("YAYY")
    else:
        print("Booo")
