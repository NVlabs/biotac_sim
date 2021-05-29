# BioTac Sim
This repo contains simulation scripts and assets for the ICRA 2021 paper, "Sim-to-real for robotic tactile sensing via physics-based simulation and learned latent projections." ([paper](https://arxiv.org/abs/2103.16747) | [website](https://sites.google.com/nvidia.com/tactiledata2))

The script provides a simple example of how to import the BioTac assets into NVIDIA Isaac Gym, launch a simulation with multiple indenters across multiple parallel environments, and extract features (net forces, nodal coordinates, and element-wise stresses).

## Installation:
- Clone repo
- Download [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym/download)
    - Follow provided instructions to create and activate `rlgpu` Conda environment for Isaac Gym
- Install `h5py` package via Conda

## Usage:
- Execute `sim_biotac.py`
    - See code for available command line switches
- View `results.hdf5`
    - File structure is `timestep / feature / environment / data`

## FAQ:
- Error: `cannot open shared object file`
    - Add `/home/username/anaconda3/envs/rlgpu/lib` to `LD_LIBRARY_PATH`
- Warning: `Degenerate or inverted tet`
    - Safely ignore

## Additional:
- For questions related to NVIDIA Isaac Gym, please see the [official forum](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/isaac-gym/322)
