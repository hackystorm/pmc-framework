# llama.cpp patch: h and logits in one decode

Path C (the optional live runtime) reads the hidden state and the logits from
the same decode. On llama.cpp commit `1248fd8`, the LFM2 graph skips the output
projection when `cparams.embeddings` is true, so `llama_get_logits_ith` returns
a non-NULL all-zero buffer. The control model BitNet-2B passes the same gate;
the failure is specific to LFM2.

The patch builds the output projection unconditionally. `res->t_embd` is set
to the post-final-norm tensor before the projection, so the hidden state is
unchanged. The extra compute is the output projection for embedding-only
callers; the live runtime needs it.

The patch applies to the stock upstream commit, not to the private fork's
uncommitted research changes. It is for the live runtime only: the frozen GGUF
artifact does not need it. The state-bearing 350M file is the measured
stock-load case. The fused 8B file (128008 tokens, state metadata) also loads
in a stock `libllama` built from `1248fd8` without this patch: a C probe
printed `n_vocab=128008`, `n_embd=2048`, `n_params=8467873216` and
`model_size=3326681856`, and the loader logged the `pmc.state.*`,
`pmc.version`, `pmc.manifest` and `pmc.dream_id` metadata before ignoring it.
The CLI entry point in this upstream tree is the unified `llama-app`, whose
build needs the UI asset pipeline, so the measured check uses the stock C API
rather than a CLI log. The 8B `lfm2moe` model uses the same LFM2 base graph.
Both the 350M `lfm2` probe and the 8B `lfm2moe` target pass the same-decode
gate with this patch; the measured fields are below.

## Measured result

Applied to stock `1248fd8` and built with the shim in
`src/pmc/runtime/llama_spike_shim.c`:

| field | 350M `lfm2` | 8B `lfm2moe` |
| --- | --- | --- |
| `gate_ok` | true | true |
| `same_decode` | true | true |
| `n_embd` | 1024 | 2048 |
| `n_vocab` | 65536 | 128000 |
| `h_sha256` | 489c41836649 | 06ced724d445 |
| `logits_nonzero` | 65536 | 128000 |
| `fw_caused_flip` | true | true |
| `decode_ms` | 35.15 | 791.56 |

The 350M `h_sha256` matches the stock embeddings-only decode. The 8B run used
`--ctx 128 --batch 32 --threads 2`; the 350M used the defaults. The full table
and the JSON fields are in `src/pmc/runtime/README.md`.

Apply and build:

```sh
git -C /path/to/llama.cpp apply /path/to/pmc-framework/patches/llama-lfm2-logits-with-embeddings.patch
cmake -S /path/to/llama.cpp -B /path/to/llama.cpp/build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build /path/to/llama.cpp/build --target llama -j2
```

The path placeholders are generic. The patch is a local research patch for the
private fork, not an upstream submission. llama.cpp is MIT-licensed; this patch
file is distributed under the repository's GPL-3.0-only licence.

The same-decode test on the 350M and 8B is in `src/pmc/runtime/README.md`.
