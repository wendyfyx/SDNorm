'''
    Create density map template
'''

import os
import sys
import logging
import argparse
import tqdm

import numpy as np
from dipy.io.image import load_nifti, save_nifti

from sdnorm.streamline_utils import normalize_density_map, log_filter_density_map

def run(args):

    dm_template = None
    logging.info(f"Creating template from {len(args.input_fpaths)} subjects.")
    for path in tqdm.tqdm(args.input_fpaths):
        dm, affine = load_nifti(path)
        dm_norm = normalize_density_map(dm, min_density=2)
        if dm_template is None:
            dm_template = dm_norm
        else:
            dm_template += dm_norm
    dm_template = dm_template/len(args.input_fpaths)
    dm_template_filtered = log_filter_density_map(dm_template, sigma=args.smooth_sigma, threshold_percentile=50, use_smoothed=True)
    save_nifti(args.output_fpath, dm_template_filtered, affine)
    logging.info(f"Saved filtered template map to {args.output_fpath}")

def main():
    logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s,%(msecs)d [%(levelname)s] %(message)s', #[%(pathname)s %(funcName)s %(lineno)d]',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    encoding='utf-8', level=logging.INFO, force=True)
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fpaths', '-i', nargs='+', type=str, required=True,
                    help='Input file paths of subject density maps')
    parser.add_argument('--output_fpath', '-o', type=str, required=True, 
                        help="Output path of template density map")
    parser.add_argument('--min_density', '-min_density', type=int, required=False, default=2, 
                        help="Minimum density for threshold each individual density map")
    parser.add_argument('--smooth_sigma', '-sigma', type=float, required=False, default=0.5, 
                        help="Sigma for Gaussian smoothing. Larger value will produce larger smoother density map")
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()