// Offline probe of UnbiasedStrata's existing GGUF reader. No engine or CUDA.
#include "strata/artifact/gguf_reader.hpp"
#include <cstdio>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: probe FIRST_SHARD ENGRAM_GGUF");
        auto m = strata::GgufModel::open(argv[1]);
        auto e = strata::GgufModel::open(argv[2]);
        size_t n = 0, unsupported = 0;
        for (size_t s = 0; s < m.size(); ++s)
            for (const auto& t : m.shard(s).tensors()) {
                if (!m.in_bounds(t, s)) {
                    ++unsupported;
                    std::printf("REJECTED: %s type=%u payload_bytes=%llu\n", t.name.c_str(), t.type,
                                (unsigned long long) strata::tensor_payload_bytes(t));
                }
                ++n;
            }
        for (const auto& t : e.meta().tensors())
            if (!e.in_bounds(t, 0)) throw std::runtime_error(t.name + ": Engram payload out of bounds");
        if (m.size() != 4 || n != 1008 || e.meta().tensors().size() != 2)
            throw std::runtime_error("unexpected shard/tensor count");
        const auto* a = m.meta().get("general.architecture");
        const auto* ea = e.meta().get("general.architecture");
        const auto* src = m.meta().get("deepseek41.engram.source");
        const auto* esrc = e.meta().get("deepseek41.engram.source");
        if (!a || a->s != "deepseek41" || !ea || ea->s != "deepseek41-engram" ||
            !src || !esrc || src->s != esrc->s) throw std::runtime_error("architecture/source mismatch");
        const std::string refusal = strata::check_engine_architecture(m.meta());
        if (refusal.empty()) throw std::runtime_error("unexpected runtime acceptance");
        for (const auto [id, block_bytes] : {std::pair{19, 50}, {29, 56}, {16, 66}, {22, 82}, {20, 18}}) {
            int block = 0, bytes = 0;
            if (!strata::block_geometry(id, block, bytes) || bytes != block_bytes || block != (id == 20 ? 32 : 256))
                throw std::runtime_error("incorrect mixed-quant block geometry");
        }
        std::printf("4 shards, 1008 backbone tensors, 2 Engram tables; rejected backbone payloads=%zu\n", unsupported);
        std::printf("PASS: exact IQ1_S/IQ1_M/IQ2_XXS/IQ2_S/IQ4_NL block geometry\n");
        std::printf("PASS: engine guard still refuses deepseek41: %s\n", refusal.c_str());
        std::puts("No model load, GPU allocation or inference performed.");
        return unsupported ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
