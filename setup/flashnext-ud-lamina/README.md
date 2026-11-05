# Qwen3.8-Flash-Next on one RTX 5080 + 64 GB: Unsloth UD-IQ3_XXS + Lamina drafter + Basalt

This branch is the exact engine we run day to day (Basalt `0.2.1` plus our open upstream PRs), and `setup.sh` rebuilds
our model file from public parts:

| part | what it brings | source (pinned) |
|---|---|---|
| Basalt (this branch) | the engine: experts split between VRAM and a CPU pool, adaptive expert swaps, MTP drafting, elastic KV | jesdga95/basalt v0.2.1 + the PRs listed below |
| Unsloth UD-IQ3_XXS ("Dynamic v3.1") | the weights: same size and the same expert bytes as GSQ-RCO IQ3_S, more accurate | `unsloth/Qwen3.8-Flash-Next-GGUF` @ `d9620119` |
| Lamina MTP | the drafter: Basalt's self-distilled MTP layer trained further, more drafts accepted | `eddoursul/Lamina-Qwen3.8-Flash-Next-MTP` @ `36295264` |
| draft vocabulary, image projector | the same files the official Basalt packs carry | `jesdga/Qwen3.8-Flash-Next-Basalt` @ `21b83575`, `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` @ `ed59f920` |
| expert start profile | which experts start in VRAM (`qwen4exp-48x512-v1pack.prof`, measured +2% over the v0.2.0 file on our box) | this folder |

## Run it

```sh
git clone -b flashnext-5080-setup https://github.com/xyehya/basalt && cd basalt
setup/flashnext-ud-lamina/setup.sh
bin/basalt serve --config basalt_config-flashnext.json --python .venv/bin/python --alias qwen3.8-flash-next
```

The script checks the machine, builds, makes `.venv`, downloads about 90 GB (83.6 GB of GGUF, 5 GB of drafter tensors
by range reads, the rest small), verifies every file's SHA-256, converts to one ~82 GB `.basalt` file and writes
`basalt_config-flashnext.json`. Reruns skip what is done. Options are at the top of `setup.sh` (`MODELS_DIR`,
`NO_VISION=1`, `SHARD4_FROM=` to reuse a GSQ IQ3_S download for the 28.8 GB n-gram shard, local file overrides).

Needs: Linux, an NVIDIA Blackwell GPU (RTX 50xx, compute capability 12.x) with 16 GB, CUDA 13, GCC 13+, CMake 3.24+,
Python 3.10+, AVX2, 64 GB RAM, ~180 GB free disk. The config is tuned for a Core Ultra 9 285K (21 pool workers); on
another CPU the script leaves the pool size to Basalt's default.

## What we measured (RTX 5080 at 360 W, Core Ultra 9 285K, 2x32 GB DDR5-7200, PCIe 5.0, Fedora)

Live agent coding (opencode, 3 prompts building a SQL engine, 151 requests up to 202K context): 195.6 tok/s average
decode, 263 tok/s best request, 156 tok/s slowest, ~5,500 tok/s prompt reads.

Same UD-IQ3_XXS file, standard sweep (fresh code prompts, 384-token replies; Basalt rows average two runs), tok/s at 1K / 4K / 40K / 150K context:

| engine | decode | prompt tok/s |
|---|---|---|
| Basalt + Lamina drafting | 213 / 219 / 217 / 165 | 1,482 / 4,019 / 5,855 / 5,572 |
| Basalt, drafting off | 131 / 138 / 137 / 124 | 1,431 / 3,833 / 5,838 / 5,601 |
| llama.cpp (Unsloth's b11985-mix build, best of our tuning, no drafter) | 40 / 42 / 38 / 31 | 171 / 706 / 858 / 832 |

Wikitext-2 perplexity through Basalt, 100 chunks (lower is better): GSQ-RCO IQ3_S 3.382, UD-IQ3_XXS 3.247, BF16
reference 3.273 (the Basalt docs' number on the same rows). UD is better on 86 of 100 chunks and costs 1 to 3.5% decode.

## What is in this branch over upstream v0.2.1

All of it is in open PRs on jesdga95/basalt: host K/V prefault before the crossover and ordinary pages for its pool
(#117), elastic KV experts without RAM copies (#114), spin-then-yield in the decode loop's waits (#246), `--adapt-margin`
(#231), the streamed conversation disk tier (#149), deferred checkpoint copy (#66), mixed-format warm-up (#101), priced
stream-all prompt chunks (#107), draft-length rule (#128), opt-in FP8 prompt path (#129, off here), timing keys and
first-token phases (#127, #232), help text (#159), prompt GEMM width invariance (#157, older form).

## Licenses

Qwen3.8-Flash-Next weights and Lamina's MTP weights: Qwen Community License 1.0 (`tools/convert/qwen-community-license-1.0.txt`).
Basalt: MIT. Unsloth's quantization and ISTA-DASLab's projector file: see their model cards.
