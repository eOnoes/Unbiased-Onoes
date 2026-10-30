// Distinct architecture entry; the donor implements V4.1's shared KV/index,
// 0/2/1 compression and external Engram history/cache. Never use DeepSeek4.
#include "strata/artifact/gguf_reader.hpp"
#include <cstdio>
#include <string>
#include <vector>

int strata_deepseek41_main(int argc, char ** argv);
int strata_deepseek41_dispatch(int argc, char ** argv) {
    std::string model, engram;
    std::vector<char *> args{argv[0]};
    // The donor's FP16 chunk test fails for IQ1_M on this card. Use the
    // separately verified token/window kernels until chunk parity is fixed.
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == "--prefill-chunk" && std::string(argv[i + 1]) != "0") {
            std::fprintf(stderr, "DeepSeek41: only --prefill-chunk 0 is qualified in this port\n"); return 2;
        }
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--architecture" || a == "--model" || a == "--engram") {
            if (++i >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); return 2; }
            if (a == "--model") model = argv[i];
            if (a == "--engram") engram = argv[i];
        } else args.push_back(argv[i]);
    }
    try {
        if (model.empty() || engram.empty()) throw std::runtime_error("deepseek41 requires --model FIRST_SHARD --engram IQ4_NL_GGUF and --pack INDEX_DIR");
        const auto m = strata::GgufModel::open(model);
        const auto e = strata::GgufModel::open(engram);
        const auto * a = m.meta().get("general.architecture");
        const auto * ea = e.meta().get("general.architecture");
        const auto * s = m.meta().get("deepseek41.engram.source");
        const auto * es = e.meta().get("deepseek41.engram.source");
        if (!a || a->s != "deepseek41" || !ea || ea->s != "deepseek41-engram" || !s || !es || s->s != es->s)
            throw std::runtime_error("DeepSeek41 architecture/Engram source mismatch");
        for (size_t i = 0; i < m.size(); ++i)
            for (const auto & t : m.shard(i).tensors())
                if (!m.in_bounds(t, i)) throw std::runtime_error(t.name + ": invalid GGUF payload bounds");
        for (const auto & t : e.meta().tensors())
            if (t.type != 20 || t.shape.size() != 2 || t.shape[0] != 256 || !e.in_bounds(t, 0))
                throw std::runtime_error(t.name + ": invalid IQ4_NL Engram table");
    } catch (const std::exception & e) { std::fprintf(stderr, "DeepSeek41: %s\n", e.what()); return 2; }
    static char prefill_flag[] = "--prefill-chunk", prefill_value[] = "0";
    args.push_back(prefill_flag); args.push_back(prefill_value);
    args.push_back(nullptr);
    return strata_deepseek41_main((int) args.size() - 1, args.data());
}
