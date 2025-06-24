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
from dipy.tracking.utils import density_map

from sdnorm.sdnorm import get_espd
from sdnorm.evaluation import *
from sdnorm.streamline_utils import load_streamlines, save_streamlines, get_step_size
from sdnorm.general_utils import save_json

def run(args):

    # Load inputs
    bundle, _ = load_streamlines(args.input_bundle_fpath)
    atlas, _ = load_streamlines(args.input_atlas_fpath)
    template, template_affine = load_nifti(args.input_template_fpath)
    fa, fa_affine, fa_voxsize = load_nifti(args.input_fa_fpath, return_voxsize=True)

    # Compute density map
    dm_bundle = density_map(bundle, fa_affine, fa.shape)
    dm_atlas = density_map(atlas, fa_affine, fa.shape)

    # Compute metrics
    report = {}
    report['n_lines'] = len(bundle)
    report['n_points'] = len(bundle.get_data())
    report['step_sze'] = get_step_size(bundle)
    report['espd'] = get_espd(bundle, fa_affine, fa, fa_voxsize)
    report['dice_atlas'] = dice(dm_bundle, dm_atlas)
    report['dice_template'] = dice(dm_bundle, template)
    report['overlap_atlas'] = overlap(dm_bundle, dm_atlas)[1]
    report['overlap_template'] = overlap(dm_bundle, template)[1]
    report['overreach_atlas'] = overreach(dm_bundle, dm_atlas)
    report['overreach_template'] = overreach(dm_bundle, template)
    report['NCC'] = normalized_cross_correlation(dm_bundle, template)
    report['bundle_similarity'] = bundle_sm(bundle, atlas) # threshold = 5
    report['bundle_adjacency'] = bundle_ba(bundle, atlas) # threshold=5, n_point_per_line=50

    # Compute FA profile
    n_segments = 100
    report['afq_profile'] = get_afq_profile(bundle, atlas, fa, fa_affine, n_segments=n_segments)
    buan_profile = get_buan_profile(bundle, atlas, fa, fa_affine, n_segments=n_segments)
    report.update(buan_profile)

    # Save report
    save_json(report, args.output_report_fpath)

def main():
    logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s,%(msecs)d [%(levelname)s] %(message)s', #[%(pathname)s %(funcName)s %(lineno)d]',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    encoding='utf-8', level=logging.INFO, force=True)
        
    parser = argparse.ArgumentParser()

    # Input args
    parser.add_argument('--input_bundle_fpath', '-i', type=str, required=True, 
                        help="Input bundle filepath")
    parser.add_argument('--output_report_fpath', '-o', type=str, required=True,
                        help="Output report filepath")
    parser.add_argument('--input_template_fpath', '-temp', type=str, required=True, 
                        help="Input template density map filepath (must be in the same space as input bundle)")
    parser.add_argument('--input_atlas_fpath', '-atlas', type=str, required=True, 
                        help="Input atlas bundle filepath (must be in the same space as input bundle)")
    parser.add_argument('--input_fa_fpath', '-fa', type=str, required=True, 
                        help="Input FA filepath (must be in the same space as input bundle)")
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()