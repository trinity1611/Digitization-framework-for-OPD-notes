#!/bin/bash
# Environment setup for a stand-alone NVIDIA GB10 system (DGX Spark class: ARM64, unified memory).
#
# STATUS: NOT VERIFIED. No GB10 machine was available when this project was written. The script is
# deliberately conservative: it inspects first, installs into an isolated virtual environment only,
# makes no system-wide change, and stops if PyTorch cannot use CUDA.
#
#   bash scripts/setup_gb10.sh                 # step 1: hardware report only (installs nothing)
#   bash scripts/setup_gb10.sh install <URL>   # step 2: create .venv; <URL> = PyTorch wheel index for this machine
#
# Choose <URL> from the hardware report (CUDA version, aarch64) and the current PyTorch / NVIDIA
# instructions for the system, for example https://download.pytorch.org/whl/cu130 . Do not guess: an
# x86-64 or CPU-only wheel will install but cannot run the models.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "===== uname -m: $(uname -m)"
python3 --version
python3 -m rxbench.hardware.diagnostics --out docs/environment/HARDWARE_REPORT_gb10_pre_env.md \
    --role-note "GB10 system, inspected before any environment was created." | sed -n '1,40p'
echo
echo "Full report: docs/environment/HARDWARE_REPORT_gb10_pre_env.md"

if [ "${1:-}" != "install" ]; then
    echo "Nothing was installed. Review the report, then run: bash scripts/setup_gb10.sh install <pytorch-index-url>"
    exit 0
fi
INDEX_URL="${2:?Give the PyTorch wheel index URL for this machine as the second argument}"

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url "$INDEX_URL"
python - <<'PY'
import platform, sys, torch
print("machine", platform.machine(), "| torch", torch.__version__, "| CUDA build", torch.version.cuda,
      "| CUDA available", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("ERROR: this PyTorch build cannot use CUDA on this machine. Stop here and choose another build.")
print("GPU:", torch.cuda.get_device_name(0))
PY
pip install -r requirements.txt
pip freeze > docs/environment/gb10_pip_freeze.txt
python -m rxbench hardware-report --out docs/HARDWARE_REPORT.md \
    --role-note "GB10 system, inside the project virtual environment." > /dev/null
python -m pytest -q --ignore=tests/test_app_pages.py | tail -n 5
echo "Done. Next: python -m rxbench models download primary   (asks for confirmation)"
