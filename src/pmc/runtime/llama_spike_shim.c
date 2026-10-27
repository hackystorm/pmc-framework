// SPDX-License-Identifier: GPL-3.0-only
//
// Minimal C shim for the PMC WP7a live-runtime spike.
//
// Purpose: keep the fragile llama_context_params / llama_model_params structs
// inside C, where they are compiled against the exact header of the build in
// use. Python only ever handles opaque pointers and plain scalar arguments.
//
// Build against the live-runtime build:
//   LLAMA=/path/to/llama.cpp
//   cc -O2 -fPIC -shared -o libpmc_spike.so llama_spike_shim.c \
//      -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
//      -L"$LLAMA/build/bin" -lllama \
//      -Wl,-rpath,"$LLAMA/build/bin"

#include "llama.h"

#include <stdint.h>
#include <stdio.h>

void * pmc_model_load(const char * path, char * err, int32_t errlen) {
    struct llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.use_mmap = true;

    struct llama_model * model = llama_model_load_from_file(path, params);
    if (model == NULL && err != NULL) {
        snprintf(err, (size_t) errlen, "llama_model_load_from_file returned NULL for %s", path);
    }
    return model;
}

void * pmc_ctx_new(void * model, uint32_t n_ctx, uint32_t n_batch, int32_t n_threads,
                   int32_t embeddings, int32_t no_perf, char * err, int32_t errlen) {
    if (model == NULL) {
        if (err != NULL) snprintf(err, (size_t) errlen, "pmc_ctx_new: model is NULL");
        return NULL;
    }
    struct llama_context_params params = llama_context_default_params();
    params.n_ctx           = n_ctx;
    params.n_batch         = n_batch;
    params.n_ubatch        = n_batch;
    params.n_threads       = n_threads;
    params.n_threads_batch = n_threads;
    params.embeddings      = embeddings != 0;
    params.no_perf         = no_perf != 0;

    struct llama_context * ctx = llama_init_from_model((struct llama_model *) model, params);
    if (ctx == NULL && err != NULL) {
        snprintf(err, (size_t) errlen, "llama_init_from_model returned NULL");
    }
    return ctx;
}

void pmc_ctx_free(void * ctx) {
    if (ctx != NULL) llama_free((struct llama_context *) ctx);
}

void pmc_model_free(void * model) {
    if (model != NULL) llama_model_free((struct llama_model *) model);
}
