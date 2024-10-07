#!/usr/bin/env python3
"""Do a full injection inspection like injspect.py, but for all sbids with
folders starting with inj.
"""
import os
import argparse
import warnings

from pathlib import Path
from glob import glob

from injspection.check_injections import injspect

if __name__ == '__main__':
    # Parse arguments with ArgumentParser from argpars as parser.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-s', '--sbid', default='SB0?????',
                        help="The schedulingblock ID to be analysed (e.g. 58479)")
    parser.add_argument('-r', '--run', type=str, default='inj*', help='Name of the run (default: "inj*").')
    parser.add_argument('-c', '--clustering_dir', type=str, default="clustering_output", help="Name of the directory "
                        'with clustering files (default: "clustering_output").')
    parser.add_argument('-p', '--fig_path', type=str, default='/data/craco/craco/jah011', help="Directory to save the plots and logs. By default a "
                        "directory in /data/craco/craco/jah011 will be created.")
    parser.add_argument('-f', '--force', action='store_true', help="Execute also if fig_path exists already.")

    args = parser.parse_args()

    sbid = args.sbid
    run = args.run
    obs_path_pattern = '/CRACO/DATA_01/craco/'
    scan_pattern = 'scans/??/*/'
    clustering_dir = args.clustering_dir
    # Define Data locations to be searched.
    inj_pattern = os.path.join(obs_path_pattern, sbid, scan_pattern, run, clustering_dir)

    # Get beam numbers that were used in this observation (usually 00 to 35).
    inj_paths = sorted(glob(inj_pattern))

    for path in inj_paths:
        path_dirs = Path(path).parts
        sbid = path_dirs[4]
        run = path_dirs[8]
        clustering_dir = path_dirs[-1]
        print(sbid, run, clustering_dir)

        # Give the arguments as a dictionary to the main function.
        fig_path = os.path.join(args.fig_path, sbid, f"{run}_{clustering_dir}")
        # print(fig_path)
        if args.force or not os.path.exists(fig_path):
            try:
                injspect(sbid, run=run, clustering_dir=clustering_dir, fig_path=fig_path)
            except Exception as err:
                warning = RuntimeWarning(*err.args)
                warning.with_traceback(err.__traceback__)
                print(warning)