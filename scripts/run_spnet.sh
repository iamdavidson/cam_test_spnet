#!/usr/bin/env bash
# Install SPNet and its inference dependencies, then run one configured benchmark.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

config="${1:-config/spnet.yaml}"
venv="${SPNET_VENV:-.venv-spnet}"
python_bin="${PYTHON_BIN:-python3}"
torch_index="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

command -v git >/dev/null
test -f "$config"
if [[ ! -x "$venv/bin/python" ]]; then
    "$python_bin" -m venv "$venv"
fi
python="$venv/bin/python"
"$python" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url "$torch_index"
"$python" -m pip install -r requirements-spnet.txt
"$python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable. Check nvidia-smi and the selected PyTorch wheel index.')
PY

settings="$("$python" - "$config" <<'PY'
from pathlib import Path
import sys
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
checkpoints = {
    'Tiny': '1ivmCX-i9lej4uJhT0Yyk2Nq9ZmoQlsB9',
    'Small': '1Ba-W3oX62lCjx5MvvGkn91LXP6SuCnV6',
    'Base': '1B9uPRVPGm1F8F-isVDVzEdHgxXmp43hn',
    'Large': '11dujPviL4pKLEXytXK0mEmPBNQDqgEak',
}
model = cfg['model']
if model not in checkpoints:
    raise SystemExit('model must be Tiny, Small, Base or Large')
print(Path(cfg['spnet_checkout']).expanduser())
print(Path(cfg.get('checkpoint') or f'checkpoints/{model}_300.pth').expanduser())
print(checkpoints[model])
PY
)"
mapfile -t values <<< "$settings"
checkout="${values[0]}"
checkpoint="${values[1]}"
drive_id="${values[2]}"

if [[ ! -e "$checkout" ]]; then
    mkdir -p "$(dirname "$checkout")"
    git clone https://github.com/Wang-xjtu/SPNet.git "$checkout"
fi
if [[ ! -f "$checkout/src/networks.py" || ! -e "$checkout/.git" ]]; then
    echo "Invalid SPNet checkout: $checkout" >&2
    exit 1
fi
if [[ ! -s "$checkpoint" ]]; then
    if ! "$python" - "$config" <<'PY'
from pathlib import Path
import sys
import yaml
if yaml.safe_load(Path(sys.argv[1]).read_text()).get('checkpoint'):
    raise SystemExit('Custom checkpoint is missing or empty. Upload it to the configured path.')
PY
    then
        exit 1
    fi
    mkdir -p "$(dirname "$checkpoint")"
    download="$(mktemp "${checkpoint}.part.XXXXXXXX")"
    trap 'rm -f -- "$download"' EXIT
    if ! "$venv/bin/gdown" "$drive_id" -O "$download" || [[ ! -s "$download" ]]; then
        echo "Download failed. Save https://drive.google.com/file/d/$drive_id/view to $checkpoint and retry." >&2
        exit 1
    fi
    mv -- "$download" "$checkpoint"
    trap - EXIT
fi

exec "$python" benchmark_spnet.py --config "$config"
