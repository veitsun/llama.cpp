#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// 大模型负责给最终答案把关，小模型负责强跑起草接下来若干个 token
int main(int argc, char ** argv) {
    common_params params; // 承载所有运行参数的结构体 （模型路径，采样，speculative，子配置等）

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    common_init(); // 通用初始化，(日志，线程亲和力之类的全局设定)

    if (params.speculative.model.path.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__); // 没有 draft model 就没有办法推测解码
        return 1;
    }

    // init llama.cpp
    llama_backend_init(); // 初始化后端（注册算子，设备，加速后端等）
    llama_numa_init(params.numa); // 如启用 NUMA，做内存绑定/亲和力设置

    llama_model * model_tgt = NULL;  // 目标模型
    //llama_model * model_dft = NULL;

    llama_context * ctx_tgt = NULL; // 目标模型的上下文
    llama_context * ctx_dft = NULL; // 草稿模型的上下文

    // load the target model
    common_init_result llama_init_tgt = common_init_from_params(params); // 依据当前 params 加载目标模型

    model_tgt = llama_init_tgt.model.get(); // 拿到模型对象 （智能指针转裸指针用于 C API）
    ctx_tgt   = llama_init_tgt.context.get(); // 拿到上下文 （里面有 KVCache， Logits 等）

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt); // 从目标模型中拿到词表

    // load the draft model 根据 params.speculative 里的配置加载草稿模型
    // 这里把参数切到草稿模型的设定（硬件设备，上下文长度，批大小，GPU层数，CPU线程等），然后再加载一次作为草稿模型
    params.devices      = params.speculative.devices;
    params.model        = params.speculative.model;
    params.n_ctx        = params.speculative.n_ctx;
    params.n_batch      = params.speculative.n_ctx > 0 ? params.speculative.n_ctx : params.n_batch;
    params.n_gpu_layers = params.speculative.n_gpu_layers;

    if (params.speculative.cpuparams.n_threads > 0) {
        params.cpuparams.n_threads = params.speculative.cpuparams.n_threads;
    }

    params.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;
    common_init_result llama_init_dft = common_init_from_params(params); // 依据当前 params 加载草稿模型

    //model_dft = llama_init_dft.model.get();
    ctx_dft   = llama_init_dft.context.get();

    if (!common_speculative_are_compatible(ctx_tgt, ctx_dft)) { // 兼容性检查确保两边的 tokenizer ，特殊 token，上下文配置等一致或可协同，否则退出
        return 1;
    }

    // Tokenize the prompt 分词与 prompt 预处理
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true); // 用目标模型的 tokenizer 对 prompt 编码

    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) { // 超过上下文长度报错
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));

        return 1;
    }

    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) { // 超过批大小报错
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));

        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {  // 打印 prompt token 对应的 piece
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    // how many tokens to draft each time
    // 设置推测式参数与采样器
    int n_draft     = params.speculative.n_max; // 每轮最多草拟多少 token
    int n_draft_min = params.speculative.n_min; // 小于这个值就当作太小的草稿，直接丢弃不算（如果草稿长度太短，不值得让目标模型批量前向，浪费吞吐，就清空丢弃）
    float p_min = params.speculative.p_min; // 最小接受概率阈值

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // used to determine end of generation
    bool has_eos = false;

    // ================================================
    // everything until here is standard initialization
    // the relevant stuff for speculative decoding starts here

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

    // eval the prompt 先用目标模型的上下文对 prompt 进行解码
    llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), inp.size() - 1)); // 先用目标模型编码完整 prompt

    // note: keep the last token separate!
    llama_token id_last = inp.back(); // 注意：把最后一个 token 留出来用于下一步接着生成
    // 典型技巧：把 prompt 的最后一个 token 保留下来，作为生成序列的起始 token（也能避免重复前向）
    // 本质原因：下一步的 logits 是由 上一个token 的前向计算产生的。把prompt的最后一个 token留出来不先算，能把它和草稿序列一起一次性送进目标模型前向，从而做一次小而多余的解码
    // 把最后一个 prompt token 留到生成环节，让它和草稿一并计算，可在每轮用一次前向就获得“从 L + 1 开始的一串 logits”，既满足草稿校验的连续性需求，又避免了为了拿 l + 1 logits 而单独再跑一次 T_L 的重复计算

    // all tokens currently in the target context
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    int n_past = inp.size() - 1;

    // init the speculator 初始化推测器与目标批
    struct common_speculative_params params_spec;
    params_spec.n_draft = n_draft;  // 每次推测解码时，草稿模型生成的token数量（即每次“草拟”的最大 token 数）。它是草稿模型一次性可以推测的最大 token 数量
    // n_reuse 表示当前推测结算草稿模型可以复用的上下文步数。复用历史信息而不是每次都从头开始生成（或重新计算）可以显著提高计算效率并节省内存，因为你不需要频繁地重新计算已生成过的 token
    params_spec.n_reuse = llama_n_ctx(ctx_dft) - n_draft; // llama_n_ctx(ctx_dft) 草稿模型上下文的最大容量，也就是草稿模型最多可以存放多少个token。它决定了草稿模型的最大上下文长度
    params_spec.p_min   = p_min;

    // 草稿模型上下文中存储了草稿生成的历史 token，越多的历史信息可以提供给草稿模型以便生成更多合理的后续 token
    // 然而每次推测时，并不是所有的历史信息都需要从头计算。剩余的历史 token 是可以复用的，用于后续推测而不需要再做额外的计算
    // 通过调整 n_draft 的大小 （每次草拟的最大 token 数），可以灵活控制上下文复用的长度。这意味着如果草稿模型能够生成足够多的有效 token（n_draft），那么可以复用更多的历史信息以优化推测过程
    // n_reuse 计算的是草稿模型可以复用的上下文步数，即在每轮推测中，草稿模型能够“带入”的历史 token 数量，以避免在推测时从头计算所有的历史信息，提升计算效率并减少内存消耗

    struct common_speculative * spec = common_speculative_init(ctx_dft);

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    const auto t_enc_end = ggml_time_us();

    const auto t_dec_start = ggml_time_us();

    while (true) {
        // optionally, generate draft tokens that can be appended to the target batch
        //
        // this is the most important part of the speculation. the more probable tokens that are provided here
        // the better the performance will be. in theory, this computation can be performed asynchronously and even
        // offloaded to a remote device. it doesn't even have to be based on an LLM. instead, it can provide tokens
        // from a cache or lookup tables.
        //
        // 草稿模型先跑，从 id_last 开始草拟若干 token
        llama_tokens draft = common_speculative_gen_draft(spec, params_spec, prompt_tgt, id_last); // 通过草稿模型从当前的 id_last （即上一个已生成的 token）开始，生成一系列草稿 token。这个生成的草稿序列会作为候选项供目标模型验证

        //LOG_DBG("draft: %s\n", string_from(ctx_dft, draft).c_str());

        // always have a token to evaluate from before - id_last
        // 构造目标模型的批；总是把 id_last 先放进去
        common_batch_clear(batch_tgt); // 清空目标模型的输入批次 batch_tgt
        common_batch_add  (batch_tgt, id_last, n_past++, { 0 }, true); // 将 id_last （即最后一个 token）加入批次。这是目标模型生成的基础 token，后续会与草稿生成的 token 一起送入目标模型

        // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
        {
            // do not waste time on small drafts
            // 如果草稿太短，直接丢弃
            if (draft.size() < (size_t) n_draft_min) {
                draft.clear();
            }

            for (size_t i = 0; i < draft.size(); ++i) {
                common_batch_add(batch_tgt, draft[i], n_past + i, { 0 }, true);
            }

            //LOG_DBG("target batch: %s\n", string_from(ctx_tgt, batch_tgt).c_str());

            llama_decode(ctx_tgt, batch_tgt); // 这个前向推理的目标是生成当前 token 以及后续草稿 token 的 logits （预测的概率分布）
        }

        // sample from the full target batch and return the accepted tokens based on the target sampler
        //
        // for each token to be accepted, the sampler would have to sample that same token
        // in such cases, instead of decoding the sampled token as we normally do, we simply continue with the
        // available logits from the batch and sample the next token until we run out of logits or the sampler
        // disagrees with the draft
        //
        const auto ids = common_sampler_sample_and_accept_n(smpl, ctx_tgt, draft); // 从目标模型的 logits 中进行采样，接受符合条件的草稿 token 。 采样器根据目标模型生成的 logits和 草稿 token 的预测值，选择哪些草稿 token 是可以被接受的。目标模型与草稿模型的预测结果相符时，草稿 token 会被接受

        //LOG_DBG("ids: %s\n", string_from(ctx_tgt, ids).c_str());

        GGML_ASSERT(ids.size() > 0); // there will always be at least one accepted token 确保至少有一个 token 被接受

        // 更新计数器
        n_past    += ids.size() - 1; // 记录已经生成的 token 数量。 ids.size() - 1 是因为 ids 包含了 id_last ，而我们只关心新生成的 token 数量
        n_drafted += draft.size(); // note: we ignore the discarded small drafts 记录草稿模型生成的 token 数量
        n_accept  += ids.size() - 1; // 记录被接受的草稿 token 数（排除第一个作为目标模型采样结果的 token）
        n_predict += ids.size(); // 记录总共生成的 token 数

        // process the accepted tokens and update contexts
        //
        // this is the standard token post-processing that we normally do
        // in this case, we do it for a group of accepted tokens at once
        //
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);

            id_last = ids[i];

            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }

            const std::string token_str = common_token_to_piece(ctx_tgt, id_last);

            if (params.use_color && i + 1 < ids.size()) {
                LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            } else {
                LOG("%s", token_str.c_str());
            }
        }

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, (int) draft.size(), id_last);

        {
            // 清理目标模型上下文中的 KVCache，避免过多的上下文导致内存溢出或性能下降。它移除不再需要的 “过期” 的 token 信息
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);

            llama_kv_self_seq_rm(ctx_tgt, 0, n_past, -1);
        }

        // 如果生成的 token 数量已经超过了指定的 n_predict ，或者遇到结束标志，则退出生成循环
        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }
    }

    auto t_dec_end = ggml_time_us();

    const int n_input = inp.size();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", n_draft);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_INF("\n");
    LOG_INF("draft:\n\n");

    llama_perf_context_print(ctx_dft);

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl);

    common_sampler_free(smpl);
    common_speculative_free(spec);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
