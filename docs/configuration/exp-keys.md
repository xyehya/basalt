# Experiment keys (`--exp`)

`--exp NAME=VALUE[,NAME=VALUE...]` sets experiment switches in the engine. They exist to test a change or force a
path, carry no stability promise, and go away when the experiment ends. A key the engine does not read is ignored;
`--exp` may be given more than once. In a config file: `"engine": { "exp": "kv_host=0,draft_th=0.4" }`.

| Key | Default | What it does |
| --- | --- | --- |
| `kv_host` | auto | The per-request host K/V of a single-GPU server ([running](../running.md#concurrent-requests---parallel-n)). `0` keeps every request's K/V in VRAM; `1` turns it on without the experts' RAM copy too, and says why when it still cannot apply. |
| `kv_host_cross` | 99328 | A request that starts decoding at or past this many cells streams its K/V from RAM; a shallower one uses the VRAM tail. |
| `kv_host_fresh` | 32768 | A session not yet streamed takes a host pool only for a request that reads at most this many new prompt tokens. |
| `kv_host_out` | 16384 | The output cells a host pool is pinned for; a reply that writes past them grows the pool between windows. |
| `kv_host_ram_mib`, `kv_host_grow_ram_mib` | -1 (the RAM actually free) | Test stand-ins for the free RAM, in MiB, at admission and at a mid-request grow; they force the fallbacks. |
| `kv_floor` | 8192 | The cells each session backs at start under the elastic KV (single-stream; batch slots use 4096). |
| `kv_grow` | 8192 | How many cells past a request's need the elastic KV backs, and its later steps (single-stream; batch slots use 4096). |
| `kv_tail_copies` | 0 (1 with `--batch`) | `1`: the experts' RAM copy also covers the lowest reach of the prompt path's loan, instead of giving those experts a RAM home when a request takes the cache's tail. |
| `draft_th` | 0.50 | The draft-length rule's threshold: a draft joins the window while the chance that it and every draft before it are accepted is at least this. A value above 0 replaces the default. |
| `fp8_prompt_min` | 1024 | With `--prefill-fp8`: the smallest prompt chunk, in tokens, that runs in FP8 (at least 1). |
| `fp8_prompt_kinds` | 91 | With `--prefill-fp8`: which projections run in FP8, as a sum of 1 GDN qkv, 2 GDN z, 4 GDN out, 8 QSA q, 16 QSA k and v, 32 QSA o, 64 shared expert. |
| `concur9_logit_hash` | 0 | `1`: print a hash of each accepted row's head logits on stderr (single-stream and batch slots). |
| `concur5_digest` | 0 | `1`: batch slots print a `BDIG` digest of their state after each window, and windows do not overlap. |
| `spec_log` | 0 | `1` with `--draft-log FILE`: write `FILE.tk`, each verified row's target distribution beside the drafter's top 4. |
| `host_ev` | 0 | `1`: record events at the decode loop's graph launches and print each request's GPU spans and idle gaps per window. |

[Precision](../precision/index.md#prefill) covers `--prefill-fp8` and what `fp8_prompt_min=1` measures.
