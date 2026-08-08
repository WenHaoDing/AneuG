"""Filesystem locations used by the mesh regularizer.

Every path can be overridden by an environment variable, so the package runs on a machine
other than the one it was developed on without editing source. Defaults point at the
layout used during development.

    export MESHREG_REFERENCE_ROOT=/path/to/AneuX
    export MESHREG_INPUT_ROOT=/path/to/shapes_to_regularize

The one exception is MODEL_DIR, which defaults to the reference_models/ folder shipped
inside this package -- the cached models are part of the deliverable, because rebuilding
one costs ~15 minutes and requires the reference dataset to be present.
"""

import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def _env(name, default):
    return os.environ.get(name, default)


# -- reference dataset: what "physiological smoothness" is learned from -----------------
REFERENCE_ROOT = _env(
    "MESHREG_REFERENCE_ROOT",
    "/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/AneuX")
REFERENCE_FILENAME = _env("MESHREG_REFERENCE_FILENAME",
                          "merged_reconstruction_remeshed.obj")

# -- cached reference models, shipped with the package ----------------------------------
MODEL_DIR = _env("MESHREG_MODEL_DIR", os.path.join(PACKAGE_DIR, "reference_models"))

# -- shapes to regularize ---------------------------------------------------------------
INPUT_ROOT = _env("MESHREG_INPUT_ROOT",
                  "/media/yaplab2/wd8tb/wenhao/angioflow/cfd/ImperialNHS")
INPUT_FILENAME = _env("MESHREG_INPUT_FILENAME", "ghd_smoothed_reconstruction.obj")

# -- downstream training data, used only to measure the export resolution ---------------
DOWNSTREAM_ROOT = _env("MESHREG_DOWNSTREAM_ROOT",
                       "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged")
DOWNSTREAM_WALL_FILE = _env("MESHREG_DOWNSTREAM_WALL_FILE", "wall_data.pt")

# -- where ablation sweeps write their results ------------------------------------------
OUTPUT_ROOT = _env("MESHREG_OUTPUT_ROOT",
                   "/media/yaplab2/wd8tb/wenhao/angioflow/cfd/_regularizer_validation")
