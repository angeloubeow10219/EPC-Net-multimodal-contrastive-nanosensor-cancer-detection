set -euo pipefail
torchrun --standalone --nproc-per-node=4 -m epcnet.cli.train --config configs/main.yaml "$@"
