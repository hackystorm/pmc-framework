/*
 * Stock-provider load probe for a fused PMC GGUF.
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * This probe links only against the stock libllama built from llama.cpp at
 * 1248fd8. It contains no PMC code: it loads the file, prints the vocabulary,
 * embedding size, parameter count and tensor count, and exits. The point is to
 * prove that a fused file is accepted by an unmodified loader.
 */

#include "llama.h"

#include <stdio.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 2;
    }

    llama_backend_init();

    struct llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.use_mmap = true;

    struct llama_model *model = llama_model_load_from_file(argv[1], params);
    if (model == NULL) {
        fprintf(stderr, "load_failed\n");
        llama_backend_free();
        return 1;
    }

    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    printf("n_vocab=%d\n", llama_vocab_n_tokens(vocab));
    printf("n_embd=%d\n", llama_model_n_embd(model));
    printf("n_params=%lld\n", (long long) llama_model_n_params(model));
    printf("model_size=%lld\n", (long long) llama_model_size(model));

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
