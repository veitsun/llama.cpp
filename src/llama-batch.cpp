#include "llama-batch.h"

#include <cassert>
#include <cstring>
#include <algorithm>

llama_ubatch llama_sbatch::reserve_ubatch(size_t n_ubatch, bool has_embd) {
    // clear empty sequences
    // the previous ubatch is assumed to be gone,
    // so nothing should refer to values in these sequences anymore.
    for (size_t i = seq.size(); i-- > 0;) {
        if (seq[i].length == 0) {
            seq.pop_back();
        } else {
            break;
        }
    }
    ubatch_token.resize(!has_embd ? n_ubatch : 0);
    ubatch_embd.resize(has_embd ? n_embd * n_ubatch : 0);
    ubatch_pos.resize(n_ubatch);
    ubatch_n_seq_id.resize(n_ubatch);
    ubatch_seq_id.resize(n_ubatch);
    ubatch_output.resize(n_ubatch);
    llama_ubatch ubatch = {
        /*equal_seqs   =*/ true,
        /*n_tokens     =*/ 0,
        /*n_seq_tokens =*/ 0,
        /*n_seqs       =*/ 0,
        /*token        =*/ !has_embd ? ubatch_token.data() : nullptr,
        /*embd         =*/ has_embd  ? ubatch_embd.data()  : nullptr,
        /*pos          =*/ ubatch_pos.data(),
        /*n_seq_id     =*/ ubatch_n_seq_id.data(),
        /*seq_id       =*/ ubatch_seq_id.data(),
        /*output       =*/ ubatch_output.data(),
    };
    return ubatch;
}

void llama_sbatch::add_seq_to_ubatch(llama_ubatch & ubatch, llama_sbatch_seq & seq, size_t length) {
    GGML_ASSERT(batch != nullptr);
    GGML_ASSERT(length <= seq.length);
    // Can only add sequences of equal lengths to a batch,
    // otherwise it isn't clear to which sequence a token belongs
    GGML_ASSERT(seq.n_seq_id == 0 || ubatch.n_seqs == 0 || length == (size_t) ubatch.n_tokens / ubatch.n_seqs);
    GGML_ASSERT((seq.n_seq_id != 0) == ubatch.equal_seqs);
    // NOTE: loops are separated for cache-friendliness
    if (batch->token) {
        if (ubatch.equal_seqs) {
            for (size_t i = 0; i < length; ++i) {
                ubatch.token[ubatch.n_tokens + i] = batch->token[ids[seq.offset + i]];
            }
        } else {
            // simple split
            ubatch.token = batch->token + seq.offset;
        }
    } else {
        ubatch.token = nullptr;
    }
    if (batch->embd) {
        if (ubatch.equal_seqs) {
            for (size_t i = 0; i < length; ++i) {
                memcpy(
                        ubatch.embd + (n_embd * (ubatch.n_tokens + i)),
                        batch->embd + (n_embd * ids[seq.offset + i]),
                        n_embd * sizeof(float)
                      );
            }
        } else {
            // simple split
            ubatch.embd = batch->embd + (n_embd * seq.offset);
        }
    } else {
        ubatch.embd = nullptr;
    }
    if (ubatch.equal_seqs) {
        for (size_t i = 0; i < length; ++i) {
            ubatch.pos[ubatch.n_tokens + i] = batch->pos[ids[seq.offset + i]];
        }
    } else {
        // simple split
        ubatch.pos = batch->pos + seq.offset;
    }
    if (ubatch.equal_seqs) {
        ubatch.n_seq_id[ubatch.n_seqs] = seq.n_seq_id;
        if (seq.seq_id) {
            ubatch.seq_id[ubatch.n_seqs] = seq.seq_id;
        }
    } else {
        // simple split
        if (batch->n_seq_id) {
            ubatch.n_seq_id = batch->n_seq_id + seq.offset;
        } else {
            for (size_t i = 0; i < length; ++i) {
                ubatch.n_seq_id[ubatch.n_seqs + i] = 1;
            }
        }
        if (batch->seq_id) {
            ubatch.seq_id = batch->seq_id + seq.offset;
        }
    }
    if (logits_all) {
        for (size_t i = 0; i < length; ++i) {
            ubatch.output[ubatch.n_tokens + i] = 1;
            out_ids.push_back(ids[seq.offset + i]);
        }
    } else if (batch->logits) {
        if (ubatch.equal_seqs) {
            for (size_t i = 0; i < length; ++i) {
                size_t id = ids[seq.offset + i];
                int8_t is_output = batch->logits[id];
                ubatch.output[ubatch.n_tokens + i] = is_output;
                if (is_output) { out_ids.push_back(id); }
            }
        } else {
            // simple split
            ubatch.output = batch->logits + seq.offset;
            for (size_t i = 0; i < length; ++i) {
                if (ubatch.output[i] != 0) { out_ids.push_back(seq.offset + i); }
            }
        }
    } else {
        // only get last output
        for (size_t i = 0; i < length; ++i) {
            size_t id = ids[seq.offset + i];
            int8_t is_last = id == ids.size() - 1;
            ubatch.output[ubatch.n_tokens + i] = is_last;
            if (is_last) { out_ids.push_back(id); }
        }
    }
    if (ubatch.n_tokens == 0 && ubatch.n_seqs == 0) {
        ubatch.n_seq_tokens = ubatch.equal_seqs ? length : 1;
    }
    ubatch.n_tokens += length;
    ubatch.n_seqs += ubatch.equal_seqs ? 1 : length; // virtual sequences for simple splits
    seq.offset += length;
    seq.length -= length;
    n_tokens -= length;
    GGML_ASSERT(ubatch.n_tokens == ubatch.n_seq_tokens * ubatch.n_seqs);
}

llama_ubatch llama_sbatch::split_simple(size_t n_ubatch) {
    n_ubatch = n_tokens < n_ubatch ? n_tokens : n_ubatch;
    llama_ubatch ubatch = reserve_ubatch(n_ubatch, /* has_embd */ batch->embd != nullptr);
    ubatch.equal_seqs = false;
    if (!seq.empty()) {
        llama_sbatch_seq & s = seq[0];
        size_t length = s.length < n_ubatch ? s.length : n_ubatch;
        GGML_ASSERT(seq.size() == 1 && s.n_seq_id == 0); // don't mix with other splits
        add_seq_to_ubatch(ubatch, s, length);
    }
    return ubatch;
}

llama_ubatch llama_sbatch::split_equal(size_t n_ubatch) {
    n_ubatch = n_tokens < n_ubatch ? n_tokens : n_ubatch;
    llama_ubatch ubatch = reserve_ubatch(n_ubatch, /* has_embd */ batch->embd != nullptr);
    if (!seq.empty()) {
        size_t length = 0;
        size_t n_tokens_in_ubatch = 0;
        GGML_ASSERT(seq[0].n_seq_id > 0); // should not be mixed with simple splits
                                          // smallest first, because it's easier to split this way;
                                          // starting from the end to pop in constant time.
        for (size_t i = seq.size(); i-- > 0;) {
            llama_sbatch_seq & s = seq[i];
            GGML_ASSERT(s.length > 0);
            if (length == 0) {
                length = s.length < n_ubatch ? s.length : n_ubatch;
            }
            add_seq_to_ubatch(ubatch, s, length);
            n_tokens_in_ubatch += length;
            // shared prompts can't be mixed with any of their sequences,
            // so it's safer to compute them in their own ubatch
            if (s.n_seq_id > 1) { break; }
            // stop when there isn't enough space for another sequence
            if (length + n_tokens_in_ubatch > n_ubatch) { break; }
        }
    }
    return ubatch;
}

llama_ubatch llama_sbatch::split_seq(size_t n_ubatch) {
    n_ubatch = n_tokens < n_ubatch ? n_tokens : n_ubatch;
    llama_ubatch ubatch = reserve_ubatch(n_ubatch, /* has_embd */ batch->embd != nullptr);
    if (!seq.empty()) {
        llama_sbatch_seq & s = seq[seq.size() - 1];
        size_t length = s.length < n_ubatch ? s.length : n_ubatch;
        GGML_ASSERT(s.n_seq_id > 0); // should not be mixed with simple splits
        add_seq_to_ubatch(ubatch, s, length);
    }
    return ubatch;
}


/**
 * @brief 把一锅“混合了多条序列的 token”的 llama_batch，重排→分段→归类，做成后续易于下发给底层算子的“微批（ubatch）原料表”。之所以要这么折腾，是因为 llama.cpp 的批处理模型允许同一批里每个 token 有各自的序列 ID 集合与位置，从而支持“连续批处理”和“共享前缀复用”。这些能力都建立在“序列 ID（seq_id）+ 位置（pos）决定注意力掩码”的机制上
 * 
 * @param batch 
 * @param n_embd 
 * @param simple_split 
 * @param logits_all 
 */
llama_sbatch::llama_sbatch(const llama_batch & batch, size_t n_embd, bool simple_split, bool logits_all) {
    GGML_ASSERT(batch.n_tokens >= 0);
    this->batch = &batch;
    this->n_embd = n_embd;
    this->logits_all = logits_all;

    n_tokens = batch.n_tokens;
    ids.resize(n_tokens);
    out_ids.clear();
    // TODO: reserve out_ids and seq

    for (size_t i = 0; i < n_tokens; ++i) {
        ids[i] = i;
    }

    if (simple_split) {
        seq.resize(1);
        llama_sbatch_seq & s = seq[0];
        s.n_seq_id = 0;
        s.seq_id = nullptr;
        s.offset = 0;
        s.length = n_tokens;
        return;
    }

    std::sort(ids.begin(), ids.end(),
            [&batch](size_t a, size_t b) {
                int32_t n_seq_a = batch.n_seq_id ? batch.n_seq_id[a] : 1;
                int32_t n_seq_b = batch.n_seq_id ? batch.n_seq_id[b] : 1;
                // sort by seq_id, then by pos
                if (n_seq_a == n_seq_b) {
                    if (batch.seq_id) {
                        for (int32_t i = 0; i < n_seq_a; ++i) {
                            llama_seq_id seq_id_a = batch.seq_id[a][i];
                            llama_seq_id seq_id_b = batch.seq_id[b][i];
                            // smaller seq_ids go first
                            if (seq_id_a != seq_id_b) {
                                return seq_id_a < seq_id_b;
                            }
                        }
                    }
                    // when all else is equal, sort by pos
                    if (batch.pos) {
                        return batch.pos[a] < batch.pos[b];
                    }
                    // no pos, sort by id
                    return a < b;
                }
                // shared prompts go first
                return n_seq_a > n_seq_b;
            }
    );

    // init seq
    llama_sbatch_seq * last_seq = nullptr;

    for (size_t i = 0; i < n_tokens; ++i) {
        const size_t bi = ids[i];
        const int32_t n_seqs = batch.n_seq_id[bi];
        llama_seq_id * seq_ids = batch.seq_id[bi];
        if (last_seq != nullptr) {
            bool same = n_seqs == last_seq->n_seq_id;
            for (int32_t j = 0; same && j < n_seqs; ++j) {
                if (seq_ids[j] != last_seq->seq_id[j]) {
                    same = false;
                }
            }
            if (same) {
                last_seq->length += 1;
                continue;
            }
        }
        llama_sbatch_seq new_seq = {n_seqs, seq_ids, i, 1};
        seq.push_back(new_seq);
        last_seq = &seq.back();
    }

    // keep shared prompts first at the end, then sort by length descending.
    std::sort(seq.begin(), seq.end(),
            [](llama_sbatch_seq & a, llama_sbatch_seq & b) {
                if (a.n_seq_id == b.n_seq_id) {
                    return a.length > b.length;
                }
                return a.n_seq_id < b.n_seq_id;
            }
            );
}

llama_batch_allocr::llama_batch_allocr(struct llama_batch in_batch, llama_pos p0) {
    // 这段构造函数接过一个 llama_batch ，检查几个关键指针是否为空；若为空，就在内部分配并填充默认值，把缺的都补上，然后把batch里的指针指向这些内部缓冲区。
    batch = in_batch;
    GGML_ASSERT(batch.n_tokens > 0);
    if (!batch.pos) { // 如果没有提供 pos，就用 p0 ,p0 + 1 、、、连续填充每个 token 的位置。并把 batch.pos 指向内部 pos 缓冲。
        // 为什么要位置：因为每个 token 都有自己的位置和序列 id，这两者共同决定它会跟哪部分 KV 交互。连续位置就是这是一段接在一起的序列
        assert(p0 >= 0);
        pos.resize(batch.n_tokens);
        for (int32_t i = 0; i < batch.n_tokens; i++) {
            pos[i] = p0 + i;
        }
        batch.pos = pos.data();
    }
    if (!batch.n_seq_id) {
        // n_seq_id 每个 token 关联了多少个 sequence id
        n_seq_id.resize(batch.n_tokens);
        for (int32_t i = 0; i < batch.n_tokens; i++) {
            n_seq_id[i] = seq_id_0.size(); // 默认设置每个 token 都属于一个序列（即每个 token 属于一个序列。llama_batch 允许“一个 token 归属于多个 seq”，因此这里的类型是“每 token 一个计数”）
        }
        batch.n_seq_id = n_seq_id.data();
    }
    if (!batch.seq_id) { // seq_id 这里是二级指针，每个 token 指向一段它自身的 seq_id 数组
        seq_id.resize(batch.n_tokens + 1); // 分配一个长度是 n_tokens + 1 的指针数组，最后一个置 NULL 当哨兵
        seq_id[batch.n_tokens] = NULL;
        for (int32_t i = 0; i < batch.n_tokens; i++) { // 让每个 token 的 seq_id[i] 都指向同一段默认的 seq_id_0 ，等价于所有 token 都属于同一个默认序列（通常是 0）
            seq_id[i] = seq_id_0.data();
        }
        batch.seq_id = seq_id.data();
    }
    if (!batch.logits) { // 若没有提供 logits 标志数组，就默认只给最后一个 token 算 logits。 这和项目里的实践一致：如果你不指定 logits_all ，通常只需要最后一步的分布用来采样。 llama_get_logits 的行为也基于这组布尔位：只有 logits[i] != 0 的 token 才会在输出的 logits 矩阵里出现，顺序与 batch 内出现的顺序一致
        logits.resize(batch.n_tokens);
        logits[logits.size() - 1] = true; // 只有最后一步的分布需要用来采样
        batch.logits = logits.data();
    }
}

///
// interface implementation
//

struct llama_batch llama_batch_get_one(
             llama_token * tokens,
                 int32_t   n_tokens) {
    return {
        /*n_tokens       =*/ n_tokens,
        /*tokens         =*/ tokens,
        /*embd           =*/ nullptr,
        /*pos            =*/ nullptr,
        /*n_seq_id       =*/ nullptr,
        /*seq_id         =*/ nullptr,
        /*logits         =*/ nullptr,
    };
}

struct llama_batch llama_batch_init(int32_t n_tokens_alloc, int32_t embd, int32_t n_seq_max) {
    llama_batch batch = {
        /*n_tokens       =*/ 0,
        /*tokens         =*/ nullptr,
        /*embd           =*/ nullptr,
        /*pos            =*/ nullptr,
        /*n_seq_id       =*/ nullptr,
        /*seq_id         =*/ nullptr,
        /*logits         =*/ nullptr,
    };

    if (embd) {
        batch.embd = (float *) malloc(sizeof(float) * n_tokens_alloc * embd);
    } else {
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_tokens_alloc);
    }

    batch.pos      = (llama_pos *)     malloc(sizeof(llama_pos)      * n_tokens_alloc);
    batch.n_seq_id = (int32_t *)       malloc(sizeof(int32_t)        * n_tokens_alloc);
    batch.seq_id   = (llama_seq_id **) malloc(sizeof(llama_seq_id *) * (n_tokens_alloc + 1));
    for (int i = 0; i < n_tokens_alloc; ++i) {
        batch.seq_id[i] = (llama_seq_id *) malloc(sizeof(llama_seq_id) * n_seq_max);
    }
    batch.seq_id[n_tokens_alloc] = nullptr;

    batch.logits   = (int8_t *)        malloc(sizeof(int8_t)         * n_tokens_alloc);

    return batch;
}

void llama_batch_free(struct llama_batch batch) {
    if (batch.token)    free(batch.token);
    if (batch.embd)     free(batch.embd);
    if (batch.pos)      free(batch.pos);
    if (batch.n_seq_id) free(batch.n_seq_id);
    if (batch.seq_id) {
        for (int i = 0; batch.seq_id[i] != nullptr; ++i) {
            free(batch.seq_id[i]);
        }
        free(batch.seq_id);
    }
    if (batch.logits)   free(batch.logits);
}
