# Streamline Density Normalization (SDNorm)

## Overview
SDNorm is a supervised method for reducing bundle variability by normalizing streamline density ([preprint](https://doi.org/10.1101/2025.08.18.670965), [twitter thread](https://x.com/wendyfyx/status/1960067687796863252)). It computes streamline weights to match an individual bundle to a template density map using ridge regression, followed by a pruning procedure to reach a target effective Streamline Point Density (eSPD). We've shown that SDNorm can
- Reduce variability in streamline density
- Improve consistency in along-tract microstructure profiles
- Provide useful metrics for automated bundle quality control

## Prerequisites
- Python (version 3.9+)
- Poetry ([official installation instructions](https://python-poetry.org/docs/#installing-with-pipx))

## Installation
```
git clone https://github.com/wendyfyx/SDNorm.git
cd SDNorm
poetry install
```

## Running SDNorm
To run SDNorm, the input bundle must be in the same space as the template map and they do not need to be in the MNI space. For testing purposes, we provided the template maps of 10 bundles in MNI space (1mm voxel size) used in our paper in `test_data/templates_mni`, and an example AF_L bundle in the same space. If you are doing tractometry, we recommend warping template maps to the subject space and running SDNorm there.
```
poetry run sdnorm -i test_data/AF_L.trk \
                  -temp test_data/templates_mni/AF_L_template_dm.nii.gz \
                  -o test_data/sdnorm_outputs/AF_L_sdnorm.trk \
                  -ow test_data/sdnorm_outputs/AF_L_sdnorm_weights.txt \
                  -oi test_data/sdnorm_outputs/AF_L_sdnorm_indices.txt \
                  -or test_data/sdnorm_outputs/AF_L_sdnorm_report.json \
                  -lambda 0.001 -espd 8 -step 0.5
```
If you already ran SDNorm, but want to prune bundles with different parameters without refitting the model, you can run
```
poetry run sdnorm_prune -i test_data/AF_L.trk \
                        -w test_data/sdnorm_outputs/AF_L_sdnorm_weights.txt \
                        -temp test_data/templates_mni/AF_L_template_dm.nii.gz \
                        -o test_data/sdnorm_outputs/AF_L_sdnorm_2.trk \
                        -oi test_data/sdnorm_outputs/AF_L_sdnorm_indices_2.txt \
                        -espd 12 -step 0.5

```
but make sure you save the streamline weights when you first run SDNorm!

Here's the sample AF_L bundle before running SDNorm with 2900 streamlines,
<div>
  <img
  src="test_data/AF_L_orig.png"
  width="300">
</div>
and here's the same bundle after running SDNorm with espd=12, and it has 1112 streamlines.
<div>
<img
  src="test_data/AF_L_espd12.png"
  width="300">
</div>



## Creating Custom Template Maps
To create your own template maps, you can create density maps from existing bundles using `dipy.tracking.utils.density_map`. Then run this command to generate the template map
```
poetry run sdnorm_template -i your-folder/sub-*_AF_L.nii.gz \
                           -o AF_L_template.nii.gz \
```

## Citing SDNorm

@misc{feng_streamline_2025,  
	title = {Streamline {Density} {Normalization}: {A} {Robust} {Approach} to {Mitigate} {Bundle} {Variability} in {Multi}-{Site} {Diffusion} {MRI}},  
	copyright = {http://creativecommons.org/licenses/by-nc-nd/4.0/},  
	shorttitle = {Streamline {Density} {Normalization}},  
	url = {http://biorxiv.org/lookup/doi/10.1101/2025.08.18.670965},  
	doi = {10.1101/2025.08.18.670965},  
	language = {en},  
	urldate = {2025-08-25},  
	author = {Feng, Yixue and Shuai, Yuhan and Villalón-Reina, Julio E. and Chandio, Bramsh Q. and Thomopoulos, Sophia I. and Nir, Talia M. and Jahanshad, Neda and Thompson, Paul M.},  
	month = aug,  
	year = {2025},  
}

