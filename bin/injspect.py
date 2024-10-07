#!/usr/bin/env python3
"""Do a full injection inspection loading the injections from all scans and
beams, printing some statistics, saving the pixelresponse shape, and saving
various plots to disk.
"""
import argparse

from injspection.check_injections import injspect

if __name__ == '__main__':
    # Parse arguments with ArgumentParser from argpars as parser.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sbid',
                        help="The schedulingblock ID to be analysed (e.g. 58479)")
    parser.add_argument('-r', '--run', type=str, default='inj', help='Name of the run (default: "inj").')
    parser.add_argument('-c', '--clustering_dir', type=str, default="clustering_output", help="Name of the directory "
                        'with clustering files (default: "clustering_output").')
    parser.add_argument('-p', '--fig_path', type=str, help="Directory to save the plots and logs. By default a "
                        "directory in /data/craco/craco/jah011 will be created.")

    args = parser.parse_args()
    # print(vars(args))
    # Give the arguments as a dictionary to the main function.
    injspect(**vars(args))
    # injspect(65462, run='inj_r4')  # Debugging 64401
