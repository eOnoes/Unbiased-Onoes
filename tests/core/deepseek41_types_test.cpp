#include "gguf_fixture.hpp"
#include <limits>
#include <iostream>
#include <tuple>
#include <tuple>

int main() {
    const auto root = std::filesystem::temp_directory_path() / "strata-deepseek41-types-test";
    std::filesystem::create_directories(root);
    int fail = 0;
    auto check = [&](bool ok) { if (!ok) ++fail; };
    for (auto [id, block, bytes] : {std::tuple{27u, 1, 8}, {19u, 256, 50}, {29u, 256, 56},
                                  {16u, 256, 66}, {22u, 256, 82}, {20u, 32, 18}}) {
        int b = 0, n = 0;
        check(strata::block_geometry(id, b, n) && b == block && n == bytes);
        auto path = root / (std::to_string(id) + ".gguf");
        fixture::write(path, {}, {{"weight", {(uint64_t) block * 2, 3}, id}});
        {
            auto g = strata::GgufModel::open(path.string());
            const auto & t = g.meta().tensors().front();
            check(strata::tensor_payload_bytes(t) == (uint64_t) bytes * 6 && g.in_bounds(t, 0));
        }
        std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
        {
            auto g = strata::GgufModel::open(path.string());
            check(!g.in_bounds(g.meta().tensors().front(), 0));
        }
    }
    strata::TensorInfo t;
    t.type = 27; t.shape = {std::numeric_limits<uint64_t>::max(), 2};
    check(strata::tensor_payload_bytes(t) == 0);
    check(std::string(strata::ggml_type_name(27)) == "I64");
    check(std::string(strata::ggml_type_name(29)) == "IQ1_M");
    std::filesystem::remove_all(root);
    std::cout << "deepseek41_types_test: " << fail << " failures\n";
    return fail ? 1 : 0;
}
