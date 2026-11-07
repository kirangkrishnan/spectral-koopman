"""Train and calibrate one model.

    python scripts/run_protocol.py vdp spectral runs/vdp/spectral

Takes a few hours per model on a CPU node. The paper's models are in results/.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spectral_koopman import get_config, get_system
from spectral_koopman.config import VARIANTS
from spectral_koopman.protocol import run

parser = argparse.ArgumentParser()
parser.add_argument("system", help="vdp, arm, cartpole, quadrotor or double_pendulum")
parser.add_argument("variant", choices=list(VARIANTS))
parser.add_argument("out", type=Path)
parser.add_argument("--threads", type=int, default=4)
args = parser.parse_args()

torch.set_num_threads(args.threads)
summary = run(get_system(args.system), get_config(args.system, args.variant), args.out)
print(json.dumps({k: v for k, v in summary.items() if k != "settings"}, indent=1, default=float))
