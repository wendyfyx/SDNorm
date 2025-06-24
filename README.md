# Streamline Density Normalization (SDNorm)

## Overview
SDNorm is a supervised method for reducing bundle variability by normalizing streamline density.

<div style="display: flex">
  <div>
    <img
    src="test_data/AF_L_orig.png"
    width="300">
  </div>
  <figcaption>AF_L before SDNorm (2900 streamlines)</figcaption>
  <div>
  <img
    src="test_data/AF_L_espd12.png"
    width="300">
  </div>
  <figcaption>AF_L after SDNorm with eSPD=12 (1112 streamlines)</figcaption>
</div>


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

To create your own template maps, you can create some density maps from extract bundles using `dipy.tracking.utils.density_map`. Then run this command to generate the template map
```
poetry run sdnorm_template -i your-folder/sub-*_AF_L.nii.gz \
                           -o AF_L_template.nii.gz \
```
