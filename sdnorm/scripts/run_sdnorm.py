'''
    Run SDNorm with pruning
'''

import os
import sys
import logging
import argparse
import tqdm

import numpy as np
from dipy.io.image import load_nifti, save_nifti

from sdnorm.sdnorm import sdnorm_voxel_streamline_matrix, sdnorm_optimize, sdnorm_prune
from sdnorm.streamline_utils import load_streamlines, save_streamlines, qb_subsampling, resample_streamlines_by_step
from sdnorm.general_utils import save_json

def run(args):

    # Load streamlines
    streamlines, _ = load_streamlines(args.input_bundle_fpath)

    # Determine step size for resampling 
    if args.step_size is None:
        step_size = 2 / np.power(args.target_espd, 1/3)
    else:
        step_size = args.step_size

    # Subsample and resample streamlines if needed
    if len(streamlines) > args.qb_samples:
        streamlines = qb_subsampling(streamlines, n_samples=args.qb_samples)
    streamlines = resample_streamlines_by_step(streamlines, step_size)  

    # SDNorm step 1: get streamline weights
    dm_template, dm_affine, dm_voxsize = load_nifti(args.input_template_fpath, return_voxsize=True)
    A, f = sdnorm_voxel_streamline_matrix(streamlines, dm_template, dm_affine)
    weights, report = sdnorm_optimize(A, f, 
                                      reg_lambda=args.reg_lambda)

    # SDNorm step 2: prune streamlines to reach target eSPD
    streamlines_pruned, _, idx_pruned, report2 = sdnorm_prune(streamlines, weights, target_espd=args.target_espd,
                                                      ref_affine=dm_affine, ref_img=dm_template, 
                                                      ref_voxsize=dm_voxsize)
    report.update(report2)

    # Save outputs
    if args.output_bundle_fpath is not None:
        save_streamlines(streamlines_pruned, args.input_bundle_fpath, args.output_bundle_fpath)
    if args.output_weight_fpath is not None:
        np.savetxt(args.output_weight_fpath, weights, fmt='%1.10f')
        logging.info(f"Saved streamline weights to {args.output_weight_fpath}")
    if args.output_indices_fpath is not None and len(idx_pruned) < len(streamlines):
        np.savetxt(args.output_indices_fpath, idx_pruned, fmt='%d')
        logging.info(f"Saved pruned indices to {args.output_indices_fpath}")
    if args.output_report_fpath is not None:
        save_json(report, args.output_report_fpath)
        # logging.info(f"Saved SDNorm report to {args.output_report_fpath}")


def main():
    logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s,%(msecs)d [%(levelname)s] %(message)s', #[%(pathname)s %(funcName)s %(lineno)d]',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    encoding='utf-8', level=logging.INFO, force=True)
        
    parser = argparse.ArgumentParser()

    # Input args
    parser.add_argument('--input_bundle_fpath', '-i', type=str, required=True, 
                        help="Input bundle filepath")
    parser.add_argument('--input_template_fpath', '-temp', type=str, required=True, 
                        help="Input template density map filepath")

    # Output args
    parser.add_argument('--output_bundle_fpath', '-o', type=str, required=False, default=None, 
                        help="Output bundle filepath")
    parser.add_argument('--output_weight_fpath', '-ow', type=str, required=False, default=None, 
                        help="Output streamline weights filepath.")
    parser.add_argument('--output_indices_fpath', '-oi', type=str, required=False, default=None, 
                        help="Output pruning indices filepath")
    parser.add_argument('--output_report_fpath', '-or', type=str, required=False, default=None, 
                        help="Output SDNorm report filepath")

    # SDNorm args
    parser.add_argument('--reg_lambda', '-lambda', type=float, required=False, default=0.001,
                        help="Weights of ridge regularization term")
    parser.add_argument('--target_espd', '-espd', type=float, required=False, default=8,
                        help="Target eSPD")
    parser.add_argument('--step_size', '-step', type=float, required=False, default=None,
                        help="Step size for resampling streamlines. Default (target espd ^ {1/3}) / 2.")
    parser.add_argument('--qb_samples', '-qb_samples', type=int, required=False, default=10000,
                        help="Number of streamlines to subsample using QuickBundles before SDNorm if bundle is too large.")

    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()