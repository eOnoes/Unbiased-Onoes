// Tokenizer-only adapter for a child-owned architecture. No architecture key
// is overridden; this class cannot load weights or build a llama graph.
struct strata_vocab_only_model final : llama_model_base {
    explicit strata_vocab_only_model(const llama_model_params & p) : llama_model_base(p) {}
    void load_arch_hparams(llama_model_loader &) override {
        throw std::runtime_error("DeepSeek41 is vocabulary-only in the frontend");
    }
    void load_arch_tensors(llama_model_loader &) override {
        throw std::runtime_error("DeepSeek41 tensors belong to strata-poly");
    }
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params &) const override {
        throw std::runtime_error("DeepSeek41 graphs belong to strata-poly");
    }
};
