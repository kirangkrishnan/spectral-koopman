"""Direct regressor baseline, trained on the training values of a spectral run.

    python scripts/train_regressor.py vdp results/vdp/spectral runs/vdp/regressor
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spectral_koopman import get_config, get_system
from spectral_koopman.identification import training_targets
from spectral_koopman.protocol import budgets
from spectral_koopman.regressor import save_regressor, train_regressor

parser = argparse.ArgumentParser()
parser.add_argument("system")
parser.add_argument("spectral_run", type=Path)
parser.add_argument("out", type=Path)
args = parser.parse_args()

system = get_system(args.system)
thetas = np.load(args.spectral_run / "training_theta.npy")
# same targets as the spectral model (identification is seeded by theta)
targets, mask = training_targets(system, thetas, budgets(get_config(args.system))[0])
model, _ = train_regressor(system, thetas, targets, mask)
args.out.mkdir(parents=True, exist_ok=True)
save_regressor(model, args.out / "model.pth", system)
