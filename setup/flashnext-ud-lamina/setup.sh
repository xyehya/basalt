#!/usr/bin/env bash
# setup.sh: build Basalt and a Qwen3.8-Flash-Next model file made of Unsloth's UD-IQ3_XXS weights plus Lamina's
# retrained MTP drafter, then write a config for one RTX 5080-class GPU (16 GB) and 64 GB of RAM. Run it from anywhere;
# it works on the checkout it lives in. Every download is pinned to a revision and checked against its SHA-256.
#
#   setup/flashnext-ud-lamina/setup.sh
#
# Environment (all optional):
#   MODELS_DIR    where the downloads and the .basalt file go (default: <checkout>/models); needs ~180 GB free
#   JOBS          build jobs (default: nproc)
#   SKIP_BUILD=1  keep an existing build in bin/
#   NO_VISION=1   leave out the image projector (saves 0.9 GB; image input is then off)
#   SHARD4_FROM   path to ISTA-DASLab's GSQ-RCO IQ3_S shard 2 of 2: its n-gram table is Unsloth's shard 4 of 4 byte for
#                 byte once re-stamped, which saves a 28.8 GB download (checked against Unsloth's SHA-256)
#   UD_GGUF, MMPROJ, MTP_SOURCE, LAMINA_DENSE, DRAFT_VOCAB   use files you already have instead of downloading them
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
B=$(cd "$HERE/../.." && pwd)
MODELS_DIR=${MODELS_DIR:-$B/models}
JOBS=${JOBS:-$(nproc)}
PY=$B/.venv/bin/python
PACK=$MODELS_DIR/Qwen3.8-FN-UD-IQ3_XXS-lamina-mtp.basalt
CONFIG=$B/basalt_config-flashnext.json

# pinned sources
UNSLOTH_REPO=unsloth/Qwen3.8-Flash-Next-GGUF;            UNSLOTH_REV=d96201194f103c906994f283f504209dc8d2a10c
LAMINA_REPO=eddoursul/Lamina-Qwen3.8-Flash-Next-MTP;     LAMINA_REV=362952648189e083ee413b3cdcac53eb5f0d183e
BASALT_REPO=jesdga/Qwen3.8-Flash-Next-Basalt;            BASALT_REV=21b8357546172bf19a685ccf3353e7da7aeadde0
GSQ_REPO=ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF;    GSQ_REV=ed59f92082b1e93c0e96d60a8b11aab089b52f09
SHA_LAMINA=a46561c5d3e76e78eebb9b54bbf4ae87ba1f99a62cad85562cbb516141cd5e47
SHA_VOCAB=b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91
SHA_MMPROJ=b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0
SHA_SHARD4=5de8b5dbf89409acc39468decafd4a4a95b13c29d88a14b0356107897829ec36

say()  { printf '\n== %s\n' "$*"; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
warn() { printf 'warning: %s\n' "$*" >&2; }
check_sha() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ] || die "$1: SHA-256 mismatch (delete it and run again)"; }
ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }

say "checking the machine"
[ "$(uname -s)" = Linux ] || die "this script is for Linux (Basalt also builds on Windows: see docs/platforms.md)"
grep -qw avx2 /proc/cpuinfo || die "the CPU has no AVX2"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: install the NVIDIA driver"
CC_GPU=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
case "$CC_GPU" in 12.*) ;; *) die "Basalt needs an NVIDIA Blackwell GPU (compute capability 12.x); found $CC_GPU" ;; esac
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
command -v nvcc >/dev/null || [ -x /usr/local/cuda/bin/nvcc ] || die "nvcc not found: install CUDA 13"
NVCC=$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)
CUDA_V=$("$NVCC" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')
ver_ge "$CUDA_V" 13.0 || die "CUDA $CUDA_V found; Basalt needs CUDA 13"
command -v cmake >/dev/null && ver_ge "$(cmake --version | awk 'NR==1 {print $3}')" 3.24 || die "CMake 3.24 or newer is needed"
GCC_V=$(${CXX:-g++} -dumpfullversion 2>/dev/null || echo 0)
ver_ge "$GCC_V" 13 || die "${CXX:-g++} $GCC_V found; the Linux build needs GCC 13 or newer (set CXX/CC to pick one)"
PYV=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
ver_ge "$PYV" 3.10 || die "Python $PYV found; 3.10 or newer is needed"
RAM_GIB=$(awk '/MemTotal/ {printf "%d", $2 / 1048576}' /proc/meminfo)
[ "$RAM_GIB" -ge 60 ] || warn "${RAM_GIB} GiB of RAM: 64 GB is the tested size; with less, experts spill to the SSD and decode slows"
mkdir -p "$MODELS_DIR"
FREE_GB=$(df -P -BG "$MODELS_DIR" | awk 'NR==2 {print $4+0}')
[ "$FREE_GB" -ge 180 ] || warn "${FREE_GB} GB free in $MODELS_DIR: the downloads and the model file need about 180 GB"
echo "GPU $GPU (cc $CC_GPU), CUDA $CUDA_V, gcc $GCC_V, Python $PYV, RAM ${RAM_GIB} GiB, ${FREE_GB} GB free in $MODELS_DIR"

if [ "${SKIP_BUILD:-0}" != 1 ]; then
    say "building Basalt (bin/basalt, bin/basalt-bench, bin/basalt-probe, bin/basalt-vision)"
    cmake -S "$B" -B "$B/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$NVCC" -DCMAKE_CUDA_ARCHITECTURES=120
    cmake --build "$B/build" -j "$JOBS"
fi
[ -x "$B/bin/basalt" ] || die "bin/basalt is missing: the build did not finish"

say "Python environment (.venv): the server's requirements and the Hugging Face CLI"
[ -x "$PY" ] || python3 -m venv "$B/.venv"
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q -r "$B/requirements.txt" "huggingface_hub[cli]>=0.34"
HF=$B/.venv/bin/hf
dl() { "$HF" download "$1" "${@:3}" --revision "$2" --local-dir "$DL"; }

DL=$MODELS_DIR/src; mkdir -p "$DL"
UD_DIR=$DL/UD-IQ3_XXS
if [ -z "${UD_GGUF:-}" ]; then
    say "Unsloth UD-IQ3_XXS (rev ${UNSLOTH_REV:0:8}): 83.6 GB in 4 files"
    S4=$UD_DIR/Qwen3.8-Flash-Next-UD-IQ3_XXS-00004-of-00004.gguf
    if [ -n "${SHARD4_FROM:-}" ] && [ ! -s "$S4" ]; then
        mkdir -p "$UD_DIR"; cp --reflink=auto "$SHARD4_FROM" "$S4"
        "$PY" - "$S4" <<'PY'
import struct, sys
with open(sys.argv[1], "r+b") as f:
    head = f.read(1 << 24)
    for key, val in ((b"split.no", 3), (b"split.count", 4)):     # GGUF uint16 values, rewritten in place
        pat = struct.pack("<Q", len(key)) + key + struct.pack("<I", 2)
        f.seek(head.index(pat) + len(pat)); f.write(struct.pack("<H", val))
PY
        check_sha "$S4" "$SHA_SHARD4"; echo "shard 4 made from $SHARD4_FROM"
        dl "$UNSLOTH_REPO" "$UNSLOTH_REV" --include "UD-IQ3_XXS/*-0000[123]-of-00004.gguf"
    else
        dl "$UNSLOTH_REPO" "$UNSLOTH_REV" --include "UD-IQ3_XXS/*"
    fi
    UD_GGUF=$UD_DIR/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf
fi
[ -s "$UD_GGUF" ] || die "no GGUF at $UD_GGUF"

if [ "${NO_VISION:-0}" != 1 ] && [ -z "${MMPROJ:-}" ]; then
    say "image projector (ISTA-DASLab GSQ-RCO repo, the one the official Basalt files carry)"
    dl "$GSQ_REPO" "$GSQ_REV" mmproj-Qwen3.8-Flash-Next-BF16.gguf
    MMPROJ=$DL/mmproj-Qwen3.8-Flash-Next-BF16.gguf
fi
[ "${NO_VISION:-0}" = 1 ] || check_sha "$MMPROJ" "$SHA_MMPROJ"

if [ -z "${LAMINA_DENSE:-}" ]; then
    say "Lamina's MTP dense weights (rev ${LAMINA_REV:0:8}, 116 MB)"
    dl "$LAMINA_REPO" "$LAMINA_REV" dense.bin LICENSE README.md
    LAMINA_DENSE=$DL/dense.bin
fi
check_sha "$LAMINA_DENSE" "$SHA_LAMINA"

if [ -z "${DRAFT_VOCAB:-}" ]; then
    say "the official draft vocabulary (rev ${BASALT_REV:0:8})"
    dl "$BASALT_REPO" "$BASALT_REV" convert/draft_vocab.bin
    DRAFT_VOCAB=$DL/convert/draft_vocab.bin
fi
check_sha "$DRAFT_VOCAB" "$SHA_VOCAB"

if [ -z "${MTP_SOURCE:-}" ]; then
    say "the draft layer's BF16 tensors from Qwen's checkpoint (range reads, about 5 GB, resumable)"
    MTP_SOURCE=$MODELS_DIR/mtp-src
    "$PY" "$B/tools/convert/mtp_fetch.py" fetch --out "$MTP_SOURCE"
fi
"$PY" "$B/tools/convert/mtp_fetch.py" verify --out "$MTP_SOURCE"

if [ -s "$PACK" ]; then
    say "the model file exists: $PACK (delete it to rebuild)"
else
    say "converting to one .basalt file (about 82 GB, a few minutes)"
    VISION=(--mmproj "${MMPROJ:-}"); [ "${NO_VISION:-0}" = 1 ] && VISION=(--no-vision)
    (cd "$B" && "$PY" tools/convert/convert.py --gguf "$UD_GGUF" "${VISION[@]}" --mtp-source "$MTP_SOURCE" \
        --mtp-dense "$LAMINA_DENSE" --draft-vocab "$DRAFT_VOCAB" --compat-bf16 \
        --name "Qwen3.8-Flash-Next UD-IQ3_XXS" --out "$PACK")
fi

say "writing $CONFIG"
POOL=""
grep -q "Core(TM) Ultra 9 285K" /proc/cpuinfo && POOL='"pool-workers": 21,'   # tuned on a 285K (8P + 16E)
cat > "$CONFIG" <<EOF
{
 "server": { "model": "$PACK", "gpu": "0", "host": "127.0.0.1", "port": 8080 },
 "engine": {
  "max-context": 262144, "kv": "int8", "expert-cache": "auto", "prefill": "auto", "resident-experts": true,
  "vram-reserve-mib": 300, "conversation-cache-mib": 0, "conversation-cache-disk-mib": 32768,
  "tail-role-token": 846, $POOL "adapt-every": 1, "adapt-margin": 3,
  "expert-profile": "$HERE/qwen4exp-48x512-v1pack.prof"
 }
}
EOF
"$PY" -m json.tool "$CONFIG" > /dev/null || die "the config is not valid JSON"

cat <<EOF

Done. Start the server (OpenAI and Anthropic API at http://127.0.0.1:8080/v1, web UI at http://127.0.0.1:8080/):

  cd $B && bin/basalt serve --config $CONFIG --python $PY --alias qwen3.8-flash-next

Add --host 0.0.0.0 to reach it from the LAN. The first start pins ~47 GB of experts in RAM and takes about 15 s.
Sampling comes from the model (temperature 1.0, top_p 0.95, top_k 20). Settings: docs/configuration/index.md.
EOF
