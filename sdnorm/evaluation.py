import logging

from collections import defaultdict

import numpy as np
from scipy import stats

from nibabel.streamlines.array_sequence import ArraySequence
from dipy.tracking.streamline import set_number_of_points, orient_by_streamline
from dipy.segment.bundles import bundle_adjacency, bundle_shape_similarity
from dipy.stats.analysis import gaussian_weights, afq_profile, assignment_map, values_from_volume

from sdnorm.general_utils import get_rng
from sdnorm.streamline_utils import get_centroid


def binary_mask(arr):
    '''Return binary mask (0/1) given array.'''
    return (arr>0).astype(int)


def bundle_sm(b1, b2, rng=None, threshold=5):
    '''Return bundle shape similarity score.'''

    if not isinstance(b1, ArraySequence):
        b1 = ArraySequence(b1)
    if not isinstance(b2, ArraySequence):
        b2 = ArraySequence(b2)
    return bundle_shape_similarity(b1, b2, get_rng(rng), threshold=threshold)


def bundle_ba(b1, b2, threshold=5, n_point_per_line=50):
    '''Comute bundle adjacency'''
    if not isinstance(b1, ArraySequence):
        b1 = ArraySequence(b1)
    if not isinstance(b2, ArraySequence):
        b2 = ArraySequence(b2)
    b1 = set_number_of_points(b1, n_point_per_line)
    b2 = set_number_of_points(b2, n_point_per_line)
    return bundle_adjacency(b1, b2, threshold=threshold)


def pearson_corr(map1, map2):
    '''Returns the Pearson correlation coefficient given two volumes of the same size.'''
    mask = (map1 > 0) | (map2 > 0)
    d1 = map1[mask].ravel()
    d2 = map2[mask].ravel()
    return stats.pearsonr(d1, d2)[0]


def normalized_cross_correlation(map1, map2, eps=1e-8):
    '''Returns the normalized corss correlation given two volumes of the same size.'''
    d1 = (map1 - np.mean(map1)) / (np.std(map1) + eps)
    d2 = (map2 - np.mean(map2)) / (np.std(map2) + eps)
    return np.sum(d1 * d2) / d1.size


def dice(map1, map2):
    '''Returns the dice coefficient given two volumes of the same size.'''
    assert map1.shape==map2.shape, "Shape mismatch"
    mask1 = binary_mask(map1)
    mask2 = binary_mask(map2)
    intersection = np.logical_and(mask1, mask2)
    return np.sum(intersection)*2.0 / (np.sum(mask1) + np.sum(mask2))


def overlap(map1, map2):
    '''
        Returns the overlap for two volumes of the same size, the fraction of 
        volume in map1 overlapping with map2, and the fraction of volume in map2 
        overlapping with map1.
    '''
    assert map1.shape==map2.shape, "Shape mismatch"
    mask1 = binary_mask(map1)
    mask2 = binary_mask(map2)
    intersection = np.logical_and(mask1, mask2)
    return np.sum(intersection) / np.sum(mask1), np.sum(intersection) / np.sum(mask2)


def overreach(map1, map2):
    '''
        Return overreach for two volumes of the same size
        OR = |map1 \ map2| / |map2|
    '''
    assert map1.shape==map2.shape, "Shape mismatch"
    mask1 = binary_mask(map1)
    mask2 = binary_mask(map2)
    intersection = np.logical_and(mask1, mask2)
    or_score = (np.count_nonzero(mask1)-np.count_nonzero(intersection))/np.count_nonzero(mask2)
    return or_score


def get_afq_profile(bundle, ref_bundle, scalar_map, scalar_affine, n_segments=50):
    '''
        Compute AFQ profile, given a reference bundle
    '''
    logging.info(f"Computing AFQ profile with {n_segments} points")
    centroid = get_centroid(ref_bundle, n_points=n_segments)
    bundle_orient = orient_by_streamline(bundle, centroid)
    weights = gaussian_weights(bundle_orient, n_points=n_segments)
    profile = afq_profile(scalar_map, bundle_orient, scalar_affine, 
                          weights=weights, n_points=n_segments)
    return profile


def ci_from_mean(ls, confidence=0.95):
    '''Return the confidence interval given a list of values'''
    if ls:
        se = stats.sem(ls)
        n = len(ls)
        t_crit = stats.t.ppf((1 + confidence) / 2., n-1)
        ci = t_crit * se
        return ci
    else:
        return 0


def get_buan_profile(bundle, ref_bundle, scalar_map, scalar_affine, n_segments=50):
    '''
        Compute BUAN profile, given a reference bundle
        Returns mean, standard deviation, 95% and 99% confidence interval
    '''
    logging.info(f"Computing BUAN profile with {n_segments} segments")
    segments = assignment_map(bundle, ref_bundle, n_segments)
    scalar_values = sum(values_from_volume(scalar_map, bundle, scalar_affine), [])
    
    segment_map = defaultdict(list)
    for idx, val in zip(segments, scalar_values):
        segment_map[idx].append(val)
    
    profile = defaultdict(list)
    for i in range(n_segments):
        values = segment_map.get(i, [])
        mean_val = np.mean(values) if values else 0.0
        std_val = np.std(values) if values else 0.0
        profile['mean'].append(mean_val)
        profile['std'].append(std_val)
        profile['ci95'].append(ci_from_mean(values, confidence=0.95))
        profile['ci99'].append(ci_from_mean(values, confidence=0.99))
    return profile