import logging
import tqdm

import numpy as np
import cvxpy as cp
from scipy.sparse import lil_matrix, vstack
from scipy.optimize import lsq_linear

from dipy.tracking.utils import density_map, length

from sdnorm.streamline_utils import get_step_size


def get_espd(streamlines, ref_affine, ref_img, ref_voxsize):
    '''
        Compute effect streamline point density (eSPD)
        eSPD = (N_lines * Length) / (Volume * Step size)
    '''
    dm = density_map(streamlines, ref_affine, ref_img.shape)
    nvox = np.count_nonzero(dm[dm>1])
    if nvox == 0:
        logging.error("No voxel in density map")
    volume = np.prod(ref_voxsize) * nvox
    sl_len = np.mean(list(length(streamlines)))
    step_size = get_step_size(streamlines)
    return (len(streamlines) * sl_len) / (volume * step_size)


def get_nlines_from_espd(streamlines, espd, ref_affine, ref_img, ref_voxsize):
    '''
        Get number of streamlines required to reach target ESPD
    '''
    dm = density_map(streamlines, ref_affine, ref_img.shape)
    nvox = np.count_nonzero(dm[dm>1])
    if nvox == 0:
        logging.error("No voxel in density map")
    volume = np.product(ref_voxsize) * nvox
    sl_len = np.mean(list(length(streamlines)))
    step_size = get_step_size(streamlines)
    return int(np.ceil(((espd * volume * step_size) / (sl_len))))


def lsq_report(X, w, y):
    '''
        Evaluate model fit of least squared
    '''

    report = {}
    y_pred = X @ w

    # MSE and MAE
    report['lsq_rmse'] = np.sqrt(np.mean((y - y_pred) ** 2))
    report['lsq_mae'] = np.mean(np.abs(y - y_pred))

    # r2
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    report['lsq_r2'] = r2

    # residuals
    report['residual_norm'] = cp.sum_squares(y_pred - y).value
    report['weight_norm'] = cp.sum_squares(w).value

    return report


def sdnorm_voxel_streamline_matrix(streamlines, ref_img, ref_affine):
    '''
        Create voxel streamline matrix given a template density map image 
    '''
    # Compute mask from ref_img
    mask = ref_img > 0
    mask_indices = np.where(mask.flatten())[0]
    
    # Make streamline voxel matrix (N_vox x N_sl)
    n_voxels = len(mask_indices)
    n_streamlines = len(streamlines)
    A = lil_matrix((n_voxels, n_streamlines), dtype=np.float32)
    
    for i, sl in enumerate(tqdm.tqdm(streamlines, desc="Create Voxel-Streamline Matrix")):
        sl_map = density_map([sl], ref_affine, ref_img.shape) > 0
        intersect_voxels = np.logical_and(mask, sl_map)
        if not np.any(intersect_voxels):
            continue
        vox_coords = np.where(intersect_voxels)
        lin_idx = np.ravel_multi_index(vox_coords, ref_img.shape)
        mask_idx = np.searchsorted(mask_indices, lin_idx)
        A[mask_idx, i] = 1.0
    A = A.tocsr()
    logging.info(f"Created streamline voxel matrix of size {A.shape}")

    # Get density values from template map
    f = ref_img.flatten()[mask_indices]

    return A, f


def sdnorm_optimize(A, f, reg_lambda=0.1, verbose=False):
    '''
        Compute streamline weights
    '''

    # scale A to match f
    mu = np.sum(f) / np.sum(A)
    A = A * mu
    
    w = cp.Variable(A.shape[1], nonneg=True)  # non-negative weights
    # w_prior = np.full(A.shape[1], np.mean(f) / np.mean(A.sum(axis=0))) # uniform prior
    
    reg_scale = cp.sum_squares(f).value / A.shape[1] # for scaling regularization
    objective = cp.Minimize( cp.sum_squares(A @ w - f) + reg_lambda * reg_scale * cp.norm2(w) )
        
    problem = cp.Problem(objective)
    problem.solve(verbose=verbose)
    weights = w.value
    
    report = lsq_report(A, weights, f)
    report['reg_scale'] = reg_scale
    report['global_scale'] = mu
   
    return weights, report


def sdnorm_prune(streamlines, weights, ref_affine, ref_img, ref_voxsize, 
                 target_espd=8, min_weight=2e-3, 
                 max_iter=20, espd_tol=0.1, lines_tol=5):
    '''
        Prune streamlines based on SDNorm weights until it reaches target eSPD
        
        At least one iteration of pruning will be done to remove streamlines with small weights.
        If the bundle eSPD is still larger than target, continue pruning. Use max_iter, espd_tol 
        and lines_tol to control early stopping
    '''

    init_espd = get_espd(streamlines, ref_affine, ref_img, ref_voxsize)
    logging.info(f"Init: pruning {len(streamlines)} streamlines, current eSPD is {init_espd:.3f}")
    
    # Inital pruning based on minimum weight
    i = 0
    idx = weights > min_weight
    weights_pruned = weights[idx]
    sl_pruned = streamlines[idx]
    indices = np.arange(len(streamlines))[idx]
    espd = get_espd(sl_pruned, ref_affine, ref_img, ref_voxsize)
    logging.info(f"Iter {i+1} (w>{min_weight:.3f}): kept {len(sl_pruned)}/{len(streamlines)} streamlines, current eSPD is {espd:.3f}")
    if len(sl_pruned) < len(streamlines):
        i += 1
    
    # If espd is higher than target and smaller than last iteration, continuing pruning
    while espd - target_espd > espd_tol:
        n_lines_init = len(sl_pruned)
        n_sl = get_nlines_from_espd(sl_pruned, target_espd, ref_affine, ref_img, ref_voxsize)
        if len(sl_pruned)-n_sl < lines_tol or (i+1) >= max_iter:
            break
        idx = np.argsort(weights_pruned)[-n_sl:][::-1]
        espd_old = espd
        espd = get_espd(sl_pruned[idx], ref_affine, ref_img, ref_voxsize)
        if espd > espd_old:
            break
        sl_pruned = sl_pruned[idx]
        weights_pruned = weights_pruned[idx]
        indices = indices[idx]
        logging.info(f"Iter {i+1}: kept {len(sl_pruned)}/{n_lines_init} streamlines, current eSPD is {espd:.3f}")
        i += 1
    
    report = {}
    report['espd_init'] = init_espd
    report['espd_pruned'] = espd
    report['nlines_init'] = len(streamlines)
    report['nlines_pruned'] = len(sl_pruned)
    report['n_prune_iterations'] = i
        
    return sl_pruned, weights_pruned, indices, report