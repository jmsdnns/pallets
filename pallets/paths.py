import os

PALLETS_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(PALLETS_ROOT, ".."))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved")

# https://github.com/jmsdnns/cpunks
CPUNKS_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "cpunks"))
