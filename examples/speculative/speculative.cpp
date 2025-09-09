#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

struct seq_draft {
    bool active   = false;  // 是否仍在参与
    bool drafting = false;  // 是否继续起草
    bool skip     = false;  // 是否跳过本轮

    int i_batch_dft = 0;   // 对于草稿分支s，下次要从草稿模型 logits 缓冲的哪一行继续采样（草稿模型每层只在罪行的那个位置继续采样）
    std::vector<int> i_batch_tgt; // 就是一个映射表：记录这条草稿分支的某个 token 在目标模型 batch 输出里对应哪一行 logits （在目标模型里，每个草稿分支要验证的 token 不止一个）

    std::vector<llama_token> tokens;  // 该分支已经起草的 token 列表
    std::vector<std::vector<llama_token_data>> dists;  // 每步起草时候的候选分布（用于之后的概率比对与残差）

    struct common_sampler * smpl = nullptr;  // 这条分支自己的 sampler （温度/Top-p 设定）
};

int main(int argc, char ** argv) {
    common_params params;

    // needed to get candidate probs even for temp <= 0.0
    params.sampling.n_probs = 128;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    common_init();

    if (params.speculative.model.path.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    // max number of parallel drafting sequences (i.e. tree branches)
    const int n_seq_dft = params.n_parallel;  // 并行草稿的分支数

    // probability threshold for splitting a draft branch (only for n_seq_dft > 1)
    const float p_draft_split = params.speculative.p_split;  // 当草稿候选某些 token 概率高于阈值时，复制分支进行分叉

    std::default_random_engine rng(params.sampling.seed == LLAMA_DEFAULT_SEED ? std::random_device()() : params.sampling.seed);
    std::uniform_real_distribution<> u_dist;

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;
    llama_model * model_dft = NULL;

    llama_context * ctx_tgt = NULL;
    llama_context * ctx_dft = NULL;

    // load the target model
    common_init_result llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt.model.get();
    ctx_tgt   = llama_init_tgt.context.get();

    // load the draft model
    params.devices = params.speculative.devices;
    params.model = params.speculative.model;
    params.n_gpu_layers = params.speculative.n_gpu_layers;
    if (params.speculative.cpuparams.n_threads > 0) {
        params.cpuparams.n_threads = params.speculative.cpuparams.n_threads;
    }

    params.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;
    common_init_result llama_init_dft = common_init_from_params(params);

    model_dft = llama_init_dft.model.get();
    ctx_dft   = llama_init_dft.context.get();

    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("vocab_type tgt: %d\n", vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("vocab_type dft: %d\n", vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_ERR("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_ERR("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return 1;
    }

    if (
        llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)
    ) {
        LOG_ERR("%s: draft model special tokens must match target model to use speculation\n", __func__);
        return 1;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_ERR("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_ERR("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return 1;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);
            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_ERR("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_ERR("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(ctx_tgt, i).c_str(),
                        common_token_to_piece(ctx_dft, i).c_str());
                return 1;
            }
        }
    }


    // Tokenize the prompt
    std::vector<llama_token> inp;
    // common_tokenize 是 llama.cpp 示例里常见的包装函数，等价于底层的 llama_tokenize 思路：把文本 -> token ID 列表，给后续推理用
    inp = common_tokenize(ctx_tgt, params.prompt, true, true); // 用目标模型的 tokenizer 对 prompt 编码

    const int max_context_size     = llama_n_ctx(ctx_tgt); // llama_n_ctx 读出模型的上下文窗口大小（能容纳的 token 总数）。如果 prompt token 数太多，就直接报错退出
    const int max_tokens_list_size = max_context_size - 4; // 这里预留了 4 个 token 安全余量（常见的做法是给后面要生成、拼接的 token 留点空间），因此用 max_context_size - 4 做上限

    if ((int) inp.size() > max_tokens_list_size) { // 检查上下文窗口是否够用
        LOG_ERR("%s: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    const int n_input = inp.size(); // n_input 是 prompt 的 token 数量

    const auto t_enc_start = ggml_time_us();

    // eval the prompt with both models 用两个模型把 prompt 送进 KVCache 前向预填/prefill
    // 很多示例会把”最后一个输入 token“单独作为一个 batch 解一次，这样可以确保：把 prompt 全部吃完之后，目标模型的最后一个位置有可用的 logits，好立刻进行首个采样/验证
    // llama_batch_get_one 是一个便捷构造器：快速做出”单序列，连续位置“ 的 batch 结构，供 llama_decode 使用（它在项目里长期存在，但社区计划弃用）
    llama_decode(ctx_tgt, llama_batch_get_one( inp.data(), n_input - 1));
    llama_decode(ctx_tgt, llama_batch_get_one(&inp.back(),           1)); // 用目标模型单独把最后一个 prompt token 再前向一次
    llama_decode(ctx_dft, llama_batch_get_one( inp.data(), n_input));  // 草稿模型直接把整个 prompt 作为一个 batch 送进 llama_decode, 这样草稿模型也完成了 prefill，后续就能从同样的上下文开始“起草”候选 token 供目标模型验证
    // 上面这三行是整段里最关键的 prefill 部分，目的是让目标模型和草稿模型都先把 prompt 吃完，把注意力缓存（KVCache）建立好，为后续采样做准备

    const auto t_enc_end = ggml_time_us();

    // the 2 models should have the same vocab
    //GGML_ASSERT(n_vocab == llama_vocab_n_tokens(model_dft));

    /**
         * @brief 在开启推理主循环之前，把推测解码所需要的状态全都准备好：定义本轮最多起草的 token 数，统计计数器，两套模型（target / draft）的“已处理 token 位置。采样器、并行草稿分支容器，以及两份批处理 batch（分别喂给草稿模型与目标模型）。
         * 
         */
    // how many tokens to draft each time
    int n_draft = params.speculative.n_max; // 每轮最多起草多少 token，（控制草稿分支一次向前猜多远）

    // 之后可以用这些统计的数据，来计算接受率与打印性能统计
    int n_predict = 0; // 最终真正生成了多少个 token
    int n_drafted = 0; // 草稿模型一共起草了多少个 token
    int n_accept  = 0; // 目标模型接受了多少草稿 token

    // 前面已经把 prompt 送进了两套上下文（prefill 了 kvcache），所以已经处理过的 token 数对 target 和 draft 都是 inp.size()，接下来新增的位置都是从这个位置继续
    int n_past_tgt = inp.size();
    int n_past_dft = inp.size();

    // used to determine end of generation
    bool has_eos = false;

    // target model sampling context (reuse the llama_context's sampling instance)
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling); // 目标  

    // draft sequence data
    std::vector<seq_draft> drafts(n_seq_dft);  // n_seq_dft = params.n_parallel 并行 草稿分支数（树宽度）。speculative 解码里，草稿模型可能会分叉（例如当多个候选都很稳时），所以需要为每条分支维护独立状态。

    for (int s = 0; s < n_seq_dft; ++s) {
        // allocate llama_sampler for each draft sequence
        // 给每条分支各分配一个 common_sampler(绑定到草稿模型)，这样不同分支的采样历史，温度/Top-K 等策略互不干扰
        drafts[s].smpl = common_sampler_init(model_dft, params.sampling);
    }

    // 为两套模型准备 batch : 容量与每 token 可挂的序列数
    // llama_n_batch(ctx) 返回的是这个上下文允许一次 llama_decode 调用里处理的最大 token 数量（即该 context 的 n_batch配置）。这是在创建 llama_context 时通过 llama_context_params.n_batch 设定的上限；
    // 告诉我们单次解码最多能往 batch 里塞多少个 token
    // 把 batch 的 token 容量设成各自 context 的 n_batch ，避免超出一次性可解码的上限
    // 在很多封装里，n_batch 都被描述为“并行处理的 token 数”，通常需要在 1 … n_ctx 之间选择。设太小会更频繁地分批 decode，设太大则会吃更多内存。
    llama_batch batch_dft = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, n_seq_dft);

    const auto t_dec_start = ggml_time_us();

    // sample from the last token of the prompt
    drafts[0].i_batch_tgt.resize(1); // i_batch_tgt 这组索引用于“从目标模型最近一次解码得到的 logits 缓冲里取第几个位置来采样”。
    drafts[0].i_batch_tgt[0] = 0;
    // 这里先把第 0 条草稿分支的 i_batch_tgt 扩成 1 个元素，并设置为 0：意思是“下一次我们会从目标模型当前 batch 输出的第 0 行 logits 位置开始采样/验证”。等到后面把草稿 token 批量加入 batch_tgt 时，会持续把草稿第 i 个 token ↔ 目标 batch 第 idx 行建立一一对应关系，以便快速做“draft→target”的逐位校验。
    // llama.cpp 把“设置了 logits[i] 的那些 token”的输出按出现顺序连续存放，你可以通过下标 i或者 -1（最后一个）去取对应 logit 行。示例与文档里都有“按索引取第 i 个 logits”的 API（llama_get_logits_ith）。common_sampler_sample(ctx, idx) 的 idx 就是指这个第几行输出。

    while (true) {
        std::set<int> active_seqs = {}; // 定义一个集合，来记录当前仍然处在活跃状态的 draft 序列编号。speculative decoding 中，通常会同时维护多个 draft 序列（候选的 预测 token 序列），活跃表示这个序列在本轮推理中还有效，还需要继续考虑

        // print current draft sequences
        // 遍历所有候选序列分支，如果这个 draft 已经被标记为 inactive（失效，就跳过），否则就把它的序号存进 active_seqs 集合
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }

            active_seqs.insert(s);
            const auto & tokens = drafts[s].tokens; // 取出这个 draft 中的 tokens

            LOG_DBG("draft %d: %s\n", s, string_from(ctx_dft, tokens).c_str()); // 将 token 序列转成字符串，并打印出来
        }

        // 准备解码时用到的临时变量
        int i_dft  = 0; // 一般用作 draft token 的索引，表示当前在 draft 序列里的位置
        int s_keep = 0; // 可能表示在对比验证过程中要保留的序列编号

        llama_token token_id;  // 表示当前正在处理的 token id
        std::string token_str; // 存储当前  token 的字符串表示

        // loop until we fail to accept a drafted token or we run out of drafted tokens
        while (true) {

            // check if the target token matches any of the drafts
            // for stochastic sampling, attempt to match the token with the drafted tokens
            {
                bool accept = false;
                if (params.sampling.temp > 0) {
                    // stochastic verification
                    common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft], true);

                    auto & dist_tgt = *common_sampler_get_candidates(smpl);

                    float p_tgt = 0.0f;
                    float p_dft = 0.0f;

                    while (active_seqs.size() > 0) {
                        // randomly select a sequence to verify from active sequences
                        std::uniform_int_distribution<unsigned int> u_int_dist(0, active_seqs.size() - 1);
                        int s = *std::next(active_seqs.begin(), u_int_dist(rng));
                        if (i_dft >= (int) drafts[s].tokens.size()) {
                            drafts[s].active = false;
                            active_seqs.erase(s);
                            continue;
                        }
                        if (accept) {
                            // if we already accepted a token, we can skip the rest
                            if (drafts[s].tokens[i_dft] != drafts[s_keep].tokens[i_dft]) {
                                drafts[s].active = false;
                                active_seqs.erase(s);
                            }
                            continue;
                        }

                        LOG_DBG("verifying sequence #%d at pos #%d from %d active sequence(s)\n", s, i_dft, (int) active_seqs.size());
                        float r = u_dist(rng);
                        llama_token_data_array dist_dft = { drafts[s].dists[i_dft].data() , drafts[s].dists[i_dft].size(), LLAMA_TOKEN_NULL, true };

                        //GGML_ASSERT(dist_tgt.size <= dist_dft.size);

                        // acquire the token probabilities assigned by the draft and target models
                        for (size_t i = 0; i < dist_tgt.size; i++) {
                            if (dist_tgt.data[i].id == drafts[s].tokens[i_dft]) {
                                p_tgt = dist_tgt.data[i].p;
                                break;
                            }
                        }
                        for (size_t i = 0; i < dist_dft.size; i++) {
                            if (dist_dft.data[i].id == drafts[s].tokens[i_dft]) {
                                p_dft = dist_dft.data[i].p;
                                break;
                            }
                        }
                        LOG_DBG("r = %f, p_dft = %f, p_tgt = %f\n", r, p_dft, p_tgt);
                        if (r <= p_tgt / p_dft) {
                            s_keep = s;
                            accept = true;
                            token_id = drafts[s].tokens[i_dft];
                            token_str = common_token_to_piece(ctx_tgt, token_id);
                            common_sampler_accept(smpl, token_id, true);

                            LOG_DBG("draft token %d of sequence %d (%d, '%s') accepted\n", i_dft, s, token_id, token_str.c_str());
                            break;
                        } else {
                            LOG_DBG("draft token %d of sequence %d (%d, '%s') rejected\n", i_dft, s, drafts[s].tokens[i_dft], common_token_to_piece(ctx_tgt, drafts[s].tokens[i_dft]).c_str());
                            drafts[s].active = false;

                            // calculate residual probability
                            GGML_ASSERT(dist_tgt.sorted);
                            GGML_ASSERT(dist_dft.sorted);

                            // sort dist by id
                            std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.id < b.id;
                            });
                            std::sort(dist_dft.data, dist_dft.data + dist_dft.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.id < b.id;
                            });

                            float sum_probs = 0.0f;

                            for (size_t i = 0; i < dist_tgt.size; i++) {
                                if (i < dist_dft.size) {
                                    dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p - dist_dft.data[i].p);
                                } else {
                                    dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p);
                                }

                                sum_probs += dist_tgt.data[i].p;
                            }

                            for (size_t i = 0; i < dist_tgt.size; i++) {
                                dist_tgt.data[i].p /= sum_probs;
                            }

                            // sort dist_tgt by p desc
                            std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.p > b.p;
                            });
                        }

                        active_seqs.erase(s);
                        for (int i = 0; i < n_seq_dft; i++) {
                            if (i == s) {
                                continue;
                            }
                            if (drafts[i].active && drafts[i].tokens[i_dft] == drafts[s].tokens[i_dft]) {
                                // synchronize active status for sequences with the same drafted token
                                drafts[i].active = drafts[i].active && accept;
                                if (!drafts[i].active) {
                                    active_seqs.erase(s);
                                }
                            }
                        }
                    }

                    if (!accept) {
                        // all drafted tokens were rejected
                        // sample from the target model
                        LOG_DBG("all drafted tokens were rejected, sampling from residual distribution\n");
                        std::vector<float> probs(dist_tgt.size);
                        for (size_t i = 0; i < dist_tgt.size; ++i) {
                            probs[i] = dist_tgt.data[i].p;
                        }

                        std::discrete_distribution<> dist(probs.begin(), probs.end());

                        const int idx = dist(rng);

                        token_id = dist_tgt.data[idx].id;
                        common_sampler_accept(smpl, token_id, true);
                        token_str = common_token_to_piece(ctx_tgt, token_id);
                    }
                } else { 
                    // 用目标模型在已预计算好的某个位置上取 argmax token，然后把这个 token 与各条活跃草稿在同一位置的 token 做逐一对比，匹配的草稿保留，不匹配的草稿淘汰
                    // greedy verification

                    // sample from the target model
                    // i_dft 是当前验证到草稿的第几个 token （草稿位置索引）
                    // drafts[s_keep].i_batch_tgt[i_dft] 是为了后面能逐 token 验证草稿，必须要知道：某条草稿的第 i_dft 个 token，对应目标模型 logits 数组中的哪一行/哪一个位置。否则，目标模型的结果和草稿就对不上了
                    LOG_DBG("sampling target: s_keep = %3d, i_dft = %3d, i_batch_tgt = %3d\n", s_keep, i_dft, drafts[s_keep].i_batch_tgt[i_dft]);
                    token_id = common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft]);  // token_id 是本轮最终要输出/使用的 token（可能来自接受的草稿/目标模型采样）， drafts[s_keep] 第 s_keep 条草稿分支（当前被保留的草稿）。i_batch_tgt 记录草稿分支里每个 token 在目标模型 batch 中对应的索引

                    common_sampler_accept(smpl, token_id, true);

                    token_str = common_token_to_piece(ctx_tgt, token_id);

                    for (int s = 0; s < n_seq_dft; ++s) { // 这段循环是当前位置 i_dft 上做一次“生死筛选” -- 谁的第 i_dft 个草稿 token 跟目标模型采样到的 token_id 相等，谁就可以继续存活；不等的话或者说是长度不够的分支会被当场淘汰（active = false）
                        if (!drafts[s].active) {
                            continue;
                        }

                        if (i_dft < (int) drafts[s].tokens.size() && token_id == drafts[s].tokens[i_dft]) { // 只要某条草稿分支的第 i_dft 个 token == 目标模型 argmax token，就可以视为匹配成功，该分支保持 active = true，同时 accept = true；s_keep 被更新为最后一次命中的分支编号
                            LOG_DBG("the sampled target token matches the %dth drafted token of sequence %d (%d, '%s') - accepted\n", i_dft, s, token_id, token_str.c_str());

                            s_keep = s; //（记录最新命中的分支编号） s_keep 被更新为最后一次命中的分支编号，所以，如果有多条分支都命中的话，它们都会继续存活，只不过 s_keep 指向最后一个命中的分支
                            accept = true;
                        } else {
                            drafts[s].active = false; // 不命中，或者分支太短，就会被淘汰
                        }
                    }
                    // 如果draft分支全部被淘汰，就会触发重建draft tree的操作
                }

                // 这段代码是推测解码内存验证循环里的提交本次 token 并决定是否继续沿用现有草稿的分水岭
                // 结束是否达到结束符
                if (llama_vocab_is_eog(vocab_tgt, token_id)) {
                    has_eos = true;
                }
                ++n_predict;

                if (accept) { // 本轮是否接受草稿 token
                    ++n_accept; // 接受的草稿 token 计数
                    ++n_past_tgt; // 目标模型上下文和草稿模型上下文都前进一步
                    ++n_past_dft;
                    ++i_dft;        // 在草稿模型序列里的当前位置前进（下一次要验证草稿的下一个 token）
                    if (params.use_color) {
                        // Color token according to its origin sequence
                        LOG("\u001b[%dm%s\u001b[37m", (36 - s_keep % 6), token_str.c_str());
                    } else {
                        LOG("%s", token_str.c_str());
                    }
                    continue;
                } else {
                    LOG("%s", token_str.c_str());
                    break;
                }
            }
        }

        { // 这段代码是在内层验证循环 break 之后立刻执行的“对齐上下文 + 清空重建草稿” 步骤。它完成了三件事情
            /**
                         * @brief 
                           -1 把 KV 缓存（ctx_dft/ctx_tgt） 收拢成一条统一的序列，并且保证两边都处于同一前缀位置
                           -2 清空旧的 drafts
                           -3 用刚刚最终确定的 token_id 给草稿侧播种，并让草稿模型前向一次，为下一轮 tree-based 草稿扩展做好准备
                         * 
                         */
            // 含义：要么目标模型采样到的 token_id 不等于任何草稿在当前位置的 token（贪婪验证失败），要么草稿已经用完。因此需要结束本轮推测，收拢状态，重建草稿。
            LOG_DBG("the sampled target token (%d, '%s') did not match, or we ran out of drafted tokens\n", token_id, token_str.c_str());

            // TODO: simplify
            {
                LOG_DBG("keeping sequence %d, n_past_tgt = %d, n_past_dft = %d\n", s_keep, n_past_tgt, n_past_dft);

                llama_kv_self_seq_keep(ctx_dft, s_keep); // 只保留 seq 这条序列的 KV （把其他序列的条目都丢弃）
                llama_kv_self_seq_cp  (ctx_dft, s_keep, 0, -1, -1); // （把这条权威分支的 KV 全量复制到标准序列 id = 0）把 src 这条序列在 [t0, t1] 范围的 KV 复制到 dst 序列（-1 表示整段）
                llama_kv_self_seq_keep(ctx_dft, 0); // 之后只保留 id = 0 的序列，这样草稿侧上下文被收拢为单序列 0

                llama_kv_self_seq_rm  (ctx_tgt, s_keep, n_past_tgt, -1); // （从目标侧的 s_keep 上把[n_past_tgt, 结束] 这段 KV 移除。含义：只保留到 n_past_tgt -1 的已确认前缀，把当前这一轮在验证阶段多算出来的那批草稿位点清掉，避免后续上下文错位）删除seq 在 [t0, t1] 范围的 KV 项
                llama_kv_self_seq_keep(ctx_tgt, s_keep);
                llama_kv_self_seq_cp  (ctx_tgt, s_keep, 0, -1, -1);
                llama_kv_self_seq_keep(ctx_tgt, 0);
            }

            for (int s = 0; s < n_seq_dft; ++s) { // 清空旧草稿结构，彻底结束上一轮的推测树，不再沿用任何旧的分支/位置信息
                drafts[s].active = false;  // 把所有草稿分支的毁约标志清零， token 列表，目标 batch 索引映射 i_batch_tgt 以及分布快照 dists 全部清空
                drafts[s].tokens.clear();
                drafts[s].i_batch_tgt.clear();
                drafts[s].dists.clear();
            }
            
            // note: will be erased after the speculation phase
            // 用 token_id 给草稿侧播种，并让草稿模型前向一次，为下一轮 tree-based 草稿扩展做好准备
            drafts[0].tokens.push_back(token_id); // 将刚刚最终采用的 token_id（它不是被草稿命中的 token，而是“目标模型自己决定的 token”，或“残差采样出来的 token”）作为新一轮草稿的起始点。
            drafts[0].dists.push_back(std::vector<llama_token_data>());
            drafts[0].i_batch_tgt.push_back(0);

            common_batch_clear(batch_dft); // 清空草稿侧 batch，然后把这个 token_id 作为 seq=0 在位置 n_past_dft 上的输入，加入 batch_dft
            common_batch_add  (batch_dft, token_id, n_past_dft, { 0 }, true);

            // 确保从 n_past_dft 之后的 KV 被移除（如果之前有残留/超前内容就清掉），以免上下文“越界/重复”。
            llama_kv_self_seq_rm(ctx_dft, 0, n_past_dft, -1);
            // LOG_DBG("dft batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_dft, batch_dft).c_str());

            // 让草稿模型前向一次，把这个新确认的 token 写进草稿侧的 KV
            llama_decode(ctx_dft, batch_dft);

            ++n_past_dft; // 草稿侧的“已过去长度”自增 1，与实际 KV 进度对齐。
        } // 这样下一轮就能从“（目标刚确认的）新 token 之后”开始，让草稿模型进行 tree-based sampling 扩展出新的分支树，再交由目标模型批量验证。

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) { // 若已生成的 token 数 n_predict 超过上限，或命中 EOS，则整轮生成结束。
            break;
        }

        // 准备下一轮草稿生成的 状态
        if (drafts[0].smpl) {
            common_sampler_free(drafts[0].smpl);
        }
        drafts[0].smpl = common_sampler_clone(smpl); // 把 smpl（主/目标侧采样器的配置与惩罚状态）克隆给草稿分支 drafts[0]，保证规则一致（重复惩罚、top-k/p 等）。

        int n_seq_cur  = 1; // 当前已有 1 个草稿序列（id=0）
        int n_past_cur = n_past_dft; // 草稿模型（ctx_dft）当前上下文长度（KV 位置）。

        for (int s = 0; s < n_seq_dft; ++s) { // 重置所有草稿分支的状态，除了 drafts[0] 之外，其他都设为 inactive
            drafts[s].active   = false;
            drafts[s].drafting = false;
        }
        drafts[0].active      = true;
        drafts[0].drafting    = true;
        drafts[0].i_batch_dft = 0;  // 草稿模型批处理索引起始位

        // 为目标模型批量验证预埋一个种子
        common_batch_clear(batch_tgt); // 清空目标侧 batch（batch_tgt），把播种 token（上一轮确定的 token）加入。
        common_batch_add  (batch_tgt, drafts[0].tokens[0], n_past_tgt, { 0 }, true); // batch_tgt 内部会记录每个位置对应的 seq id，便于后续分叉时把“之前的 token 历史”共享给新分支
        // 后续我们会把新起草的 token 也不断追加到 batch_tgt , 等本轮草稿扩展完毕后，一次性在目标模型上前向，得到所有这些位置的 logits 做验证

        // sample n_draft tokens from the draft model using tree-based sampling
        // 草稿生成/扩展 （真正“生成多个草稿” 的地方）
        // 外层循环：起草 n_draft 层 （“树式起草”）
        for (int i = 0; i < n_draft; ++i) { // 外层循环：起草 n_draft 层 （“树式起草”）
            // 这是第 i 层 的起草（从根往下扩一层）。 按层扩展草稿树，第 i 层每条在起草的分支各追加 1 个 token
            batch_dft.n_tokens = 0;

            for (int s = 0; s < n_seq_dft; ++s) {
                // skip 标志就是用来防止刚分裂出来的新分支在同一层被再次采样（本层：它只分配一个候选 token（继承父分支的候选分布））
                drafts[s].skip = false; // 避免“同一层里，新分裂出的分支又被重复采样一遍”。
            }

            // 对每个仍在起草的分支s；在草稿模型上 ctx_dft 上做一次采样（不前向网络，直接用上一层的 logits 缓存位置 i_batch_dft）
            for (int s = 0; s < n_seq_dft; ++s) {
                if (!drafts[s].drafting || drafts[s].skip) { // 只处理 还在起草且 本层未跳过的 分支
                    continue;
                }

                // 在这行 logits 上按照 sampler 规则 （温度，top-p/top-k ，重复惩罚）生成候选清单
                common_sampler_sample(drafts[s].smpl, ctx_dft, drafts[s].i_batch_dft, true);

                const auto * cur_p = common_sampler_get_candidates(drafts[s].smpl); // 该分支当前位点的候选分布（按概率降序）

                for (int k = 0; k < std::min(n_seq_dft + 3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - draft candidate %3d for seq %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            k, s, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                std::vector<int> sa(1, s); // 本层要 加 token 的分支列表，初始包含原分支 s

                // attempt to split the branch if the probability is high enough
                // 按概率阈值分叉（KV复制 + 状态克隆）
                for (int f = 1; f < 8; ++f) { // 从top-2开始考虑分裂（f=1表示第二高）
                    if (n_seq_cur < n_seq_dft && cur_p->data[f].p > p_draft_split) {
                        // 1) 复制草稿模型KV：新分支拥有与父分支完全相同的历史上下文
                        LOG_DBG("splitting seq %3d into %3d\n", s, n_seq_cur);

                        llama_kv_self_seq_rm(ctx_dft,    n_seq_cur, -1, -1);
                        llama_kv_self_seq_cp(ctx_dft, s, n_seq_cur, -1, -1);

                        // all previous tokens from this branch are now also part of the new branch
                        // 2) 共享目标batch的历史归属：
                        //    把 batch_tgt 中此前各token的 seq归属里凡是含 s 的，也加上新分支 n_seq_cur
                        for (int t = 0; t < batch_tgt.n_tokens; ++t) {
                            for (int p = 0; p < batch_tgt.n_seq_id[t]; ++p) {
                                if (batch_tgt.seq_id[t][p] == s) {
                                    batch_tgt.seq_id[t][batch_tgt.n_seq_id[t]] = n_seq_cur;
                                    batch_tgt.n_seq_id[t]++;
                                    break;
                                }
                            }
                        }

                        // copy the draft state
                        // 3) 复制草稿分支的“高层状态”
                        drafts[n_seq_cur].active   = true;
                        drafts[n_seq_cur].drafting = true;
                        drafts[n_seq_cur].skip     = true;// 本层不再让它单独采样

                        drafts[n_seq_cur].tokens      = drafts[s].tokens;
                        drafts[n_seq_cur].dists       = drafts[s].dists;
                        drafts[n_seq_cur].i_batch_dft = drafts[s].i_batch_dft;
                        drafts[n_seq_cur].i_batch_tgt = drafts[s].i_batch_tgt;

                        if (drafts[n_seq_cur].smpl) {
                            common_sampler_free(drafts[n_seq_cur].smpl);
                        }
                        drafts[n_seq_cur].smpl = common_sampler_clone(drafts[s].smpl);

                        sa.push_back(n_seq_cur); // 本层等会儿也给它分配一个候选token

                        n_seq_cur++;
                    } else {
                        break;// 超过分支上限或概率不达阈值，停止继续分裂
                    }
                }

                // add drafted token for each sequence
                for (int is = 0; is < (int) sa.size(); ++is) {
                    const llama_token id = cur_p->data[is].id;

                    const int s = sa[is];

                    // 1) 接受到分支采样器（更新重复惩罚等内部状态）
                    common_sampler_accept(drafts[s].smpl, id, true);

                    // 2) 记录“起草的token”与该步候选分布快照（后续随机验证要用到 p_dft）
                    drafts[s].tokens.push_back(id);
                    // save cur_p.data into drafts[s].dists
                    drafts[s].dists.push_back({cur_p->data, cur_p->data + cur_p->size});

                    // add unique drafted tokens to the target batch
                    // 3) 登记到目标模型的batch：并记录该token在目标batch中的“行号”索引
                    drafts[s].i_batch_tgt.push_back(batch_tgt.n_tokens);

                    common_batch_add(batch_tgt, id, n_past_tgt + i + 1, { s }, true);

                    // add the token to the batch for batched decoding with the draft model
                    // 4) 登记到草稿模型的batch：并设置这条分支的“下一次采样要看的logits行号”
                    drafts[s].i_batch_dft = batch_dft.n_tokens;

                    common_batch_add(batch_dft, id, n_past_cur, { s }, true);

                    if (batch_tgt.n_tokens > n_draft) {
                        drafts[s].drafting = false;
                    }
                }
            }

            // no sequence is drafting anymore
            // 5) 限制目标batch规模：超过 n_draft 就不再扩这条分支
            if (batch_dft.n_tokens == 0) {
                break; // 说明没有分支继续起草（可能都被 drafting=false 或容量/阈值卡住），无需再前向，直接退出“按层扩展”的大循环
            }

            // evaluate the drafted tokens on the draft model
            // 层末统一让草稿模型 llama_decode(ctx_dft, batch_dft) 前向，把本层所有分支新起草的 token 全部写进草稿侧 KV, 推进到下一层可继续采样
            llama_decode(ctx_dft, batch_dft);
            ++n_past_cur; // 草稿侧“已过去长度”推进一格。
            ++n_drafted;  // 统计指标。

            if (batch_tgt.n_tokens > n_draft) { // 若 batch_tgt.n_tokens 超过 n_draft，就不再扩更多层了，目标 batch 已经够本轮验证用。
                break;
            }
        }

        // evaluate the target model on the drafted tokens
        {
            llama_kv_self_seq_keep(ctx_tgt, 0); // 保留目标模型里序列 0 的 KV 缓存（已确认的上下文）
            for (int s = 1; s < n_seq_dft; ++s) {
                llama_kv_self_seq_cp(ctx_tgt, 0, s, -1, -1); // 把序列 0 的 KV 拷贝到 s，这样所有分支的上下文呢都和 seq=0 一样，保证验证时公平对齐
            }

            // LOG_DBG("target batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_tgt, batch_tgt).c_str());
            llama_decode(ctx_tgt, batch_tgt); // 对刚才草稿扩展出来的 token 批量做前向，让目标模型得到它们对应的 logits 分布。这些 logits 稍后会用来做验证，比较目标模型分布与草稿 token 是否一致
            ++n_past_tgt; // 目标模型上下文的“已过去 token 数”加一，表示它现在也往前走了一步。
        }

        // the first token is always proposed by the target model before the speculation loop so we erase it here
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }
            // 草稿序列里剩下的 token 就只包含真正由草稿模型推测出来的部分，后续验证时不会误把目标模型的种子 token 拿来对比。
            drafts[s].tokens.erase(drafts[s].tokens.begin());
            drafts[s].dists.erase(drafts[s].dists.begin());
        }
    }

    auto t_dec_end = ggml_time_us();

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
    // TODO: print sampling/grammar timings for all drafts
    llama_perf_context_print(ctx_dft);

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl);

    common_sampler_free(smpl);
    for (int s = 0; s < n_seq_dft; ++s) {
        common_sampler_free(drafts[s].smpl);
    }

    llama_batch_free(batch_dft);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
