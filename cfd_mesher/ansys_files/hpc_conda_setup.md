# Conda environment for `main.py` on the HPC

`main.py` exports Fluent EnSight results to `blood_data.pt` / `wall_data.pt` and can render
sanity images. It needs:

| package | needed for |
|---|---|
| `numpy`, `pytorch` | the export itself (always) |
| `ensight-reader` (imported as `ensightreader`) | reading the EnSight case (always); pip only |
| `pyvista`, `vtk`, `imageio` | sanity images and GIFs (`sanity_check=True`) |
| `scipy` | only `test_wall_coords_in_blood()` |

## Install

```bash
module load miniforge/3          # or however conda is provided on your cluster

conda create -n new -c conda-forge python=3.11 -y
conda activate new

# core: numpy + CPU-only PyTorch (the export needs no GPU)
conda install -c conda-forge numpy scipy "pytorch=*=cpu*" -y

# EnSight reader -- not on conda, pip only
pip install ensight-reader

# sanity rendering, HEADLESS: compute nodes have no display, and a normal VTK build
# segfaults there ("bad X server connection ... EGL ... OSMesa not found").
# The osmesa build renders in software with no display and no GPU.
conda install -c conda-forge pyvista "vtk=*=osmesa*" mesalib imageio -y

pip install --upgrade huggingface_hub
```

## Check

```bash
python -c "import numpy, torch, ensightreader, pyvista, vtk, imageio; print('ok', vtk.__version__)"
python -c "import pyvista as pv; pl = pv.Plotter(off_screen=True); pl.add_mesh(pv.Sphere()); pl.screenshot('t.png'); print('headless render ok')"
```

If the second command crashes, the OSMesa build did not install. In that case, skip
rendering on the HPC by setting `sanity_check=False` in `main.py`'s `__main__`, and render
the sanity images later on the workstation from the exported `.pt` files.

## Versions this was developed against (workstation)

python 3.11.6, numpy 2.3.5, torch 2.4.0, scipy 1.16.3, pyvista 0.42.3, vtk 9.3.1,
ensight-reader 0.13.1. Newer versions should work. If PyVista complains about VTK, pin
both to these versions.
