import os
import logging

import numpy as np
from scipy.ndimage import gaussian_filter

from nibabel.streamlines.array_sequence import ArraySequence
from dipy.io.streamline import load_trk, save_trk, save_tractogram
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.tracking.streamline import set_number_of_points, Streamlines
from dipy.tracking.utils import length
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.segment.clustering import QuickBundles

from sdnorm.general_utils import random_select


def load_streamlines(trk_file):
    tractogram = load_trk(trk_file, reference='same', bbox_valid_check=False)
    logging.info(f"Loaded {len(tractogram.streamlines)} streamlines from {trk_file}.")
    return tractogram.streamlines, tractogram.affine


def save_streamlines(bundle, orig_fpath, new_fpath):
    new_tractogram = StatefulTractogram(bundle, orig_fpath, Space.RASMM)
    save_tractogram(new_tractogram, new_fpath, bbox_valid_check=False)
    logging.info(f"Saved {len(bundle)} streamlines to {new_fpath}.")


def get_step_size(streamlines):
    '''Get the average step size from a list of streamlines.'''
    all_step_sizes = []
    for sl in streamlines:
        if len(sl) < 2:
            continue
        steps = np.linalg.norm(np.diff(sl, axis=0), axis=1)
        all_step_sizes.extend(steps)

    if not all_step_sizes:
        return 0
    return np.mean(all_step_sizes)


def resample_streamlines_by_step(streamlines, new_step_size, tol=0.01):
    ''' 
        Resample streamlines with new step size
        Remove streamlines if their lengths are smaller than new_step_size
    '''

    # Check if current step size is within tolerance of new step size
    cur_step_size = get_step_size(streamlines)
    if abs(cur_step_size-new_step_size) < tol:
        logging.info(f"Current step size {cur_step_size:.3f} is within tolerance ({tol}mm) of target step size " \
                     f"{new_step_size}, skipping resampling ")
        return streamlines

    # Filter streamlines if they are shorter than new step size
    lengths = np.array(list(length(streamlines)))
    keep_sl =  np.where(lengths > new_step_size)[0]
    if len(keep_sl) < len(streamlines):
        logging.info(f"Keeping {len(keep_sl)} streamlines ")
        lengths = lengths[keep_sl]
    
    # Resample
    npoints = np.ceil(lengths / new_step_size).astype(int)
    resampled_streamlines = ArraySequence([set_number_of_points(s, n) for s, n in
                             zip(streamlines[keep_sl], npoints)])
    
    logging.info(f"Resampled streamlines with {cur_step_size:.3f} mm step size and {streamlines.get_data().shape[0]} points " \
                 f"to {new_step_size:.3f} mm and {resampled_streamlines.get_data().shape[0]} points")
    return resampled_streamlines


def normalize_density_map(dm, min_density=2):
    '''Normalize density map so they sum to 1'''
    dm_norm = dm.copy()
    dm_norm[dm_norm < min_density] = 0
    mask = dm_norm > 0
    dm_norm = dm_norm / np.sum(dm_norm)
    return dm_norm


def log_filter_density_map(dm, sigma=1, threshold_percentile=80, use_smoothed=True):
    '''Log density filtering on streamline density map'''

    logging.info(f"Smoothing with sigma={sigma:.2f} and thresholding at {threshold_percentile}th percentile")
    smoothed_density = gaussian_filter(dm.astype("float32"), sigma=sigma)
    log_density = np.log(smoothed_density + 1e-6)
    threshold_value = np.percentile(log_density[smoothed_density>0], threshold_percentile)
    thresholded_density = (log_density > threshold_value).astype("int32")
    if use_smoothed:
        logging.info(f"Thresholding log density at {threshold_value:.3f} with smoothing, filtered {(thresholded_density>0).sum()}/{(smoothed_density>0).sum()} voxels.")
        return smoothed_density * thresholded_density
    else:
        logging.info(f"Thresholding log density at {threshold_value:.3f}, filtered {(thresholded_density>0).sum()}/{(dm>0).sum()} voxels.")
        return dm * thresholded_density
    

def qb_subsampling(streamlines, n_samples=15000, threshold=5, min_cluster_size=2, rng=0):
    '''
        Use QuickBundles to select a subset of streamlines
    '''
    if len(streamlines) < n_samples:
        return streamlines
    
    feature = ResampleFeature(nb_points=20)
    metric = AveragePointwiseEuclideanMetric(feature=feature)  # a.k.a. MDF
    qb = QuickBundles(threshold=threshold, metric=metric)
    clusters = qb.cluster(streamlines)
    clusters = [cl for cl in clusters if len(cl.indices) >= min_cluster_size]
    
    def _sample_from_cluster(cluster, n_pct, rng=0):
        n_samp = int(np.ceil(len(cluster.indices) * n_pct))
        idx = random_select(cluster.indices, n_samp, rng=rng)
        return np.array(cluster.indices)[idx]
    
    indices = [_sample_from_cluster(cl, n_samples/len(streamlines), rng=rng) for cl in clusters]
    indices = np.sort(np.concatenate(indices))
    indices = indices[:n_samples] if len(indices) > n_samples else indices

    logging.info(f"Subsampled {len(indices)} streamlines from {len(clusters)} QuickBundles clusters.")

    return streamlines[indices]


def get_centroid(streamlines, n_points=20, threshold=100):
    '''Get one single centroid from streamlines'''
    streamlines = set_number_of_points(streamlines, n_points)
    metric = AveragePointwiseEuclideanMetric()
    qb = QuickBundles(threshold=threshold, metric=metric)
    clusters = qb.cluster(streamlines)
    centroids = Streamlines(clusters.centroids)
    if len(centroids) > 1:
        logging.warn("WARNING: number clusters > 1 ({})".format(len(centroids)))
    return centroids[0]