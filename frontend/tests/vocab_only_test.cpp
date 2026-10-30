// The exact file's vocabulary and template, with no fabricated architecture.
// Full inference through llama remains refused; only strata-poly owns it.
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    llama_backend_init();
    auto p = llama_model_default_params(); p.vocab_only = true;
    llama_model * m = llama_model_load_from_file(argv[1], p);
    if (!m) return 1;
    char arch[128]{};
    const bool identity = llama_model_meta_val_str(m, "general.architecture", arch, sizeof arch) > 0 &&
                          std::strcmp(arch, "deepseek41") == 0;
    const auto * v = llama_model_get_vocab(m);
    std::vector<llama_token> ids(64);
    const char * text = "The capital of France is Paris.";
    const int n = llama_tokenize(v, text, (int) std::strlen(text), ids.data(), (int) ids.size(), true, true);
    const char * tm = llama_model_chat_template(m, nullptr);
    const bool ok = identity && llama_vocab_n_tokens(v) == 129280 && n > 0 && tm && std::strlen(tm) == 7423;
    std::printf("architecture=%s vocabulary=%d tokenized=%d template_bytes=%zu\n", arch,
                llama_vocab_n_tokens(v), n, tm ? std::strlen(tm) : 0);
    llama_model_free(m);
    p.vocab_only = false;
    m = llama_model_load_from_file(argv[1], p);
    const bool refused = m == nullptr;
    if (m) llama_model_free(m);
    std::printf("full llama inference refused=%d; vocabulary-only test=%s\n", refused, ok && refused ? "PASS" : "FAIL");
    llama_backend_free();
    return ok && refused ? 0 : 1;
}
