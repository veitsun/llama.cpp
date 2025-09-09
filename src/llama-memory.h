#pragma once

#include "llama.h"

struct llama_memory_params {
    // kv cache
    ggml_type type_k;
    ggml_type type_v;

    // use full-size SWA cache
    bool swa_full;
};

// general concept of LLM memory
// the KV cache is a type of LLM memory, but there can be other types
// 这是 LLM 内存的抽象接口。 llama_memory_i 把记住了什么，怎么改，怎么丢的动作统一成一套 API， 这样上层推理流程不用关心底下用的是那种记忆体，随时可以替换实现而不改调用代码。在社区绑定文档里面，这类 memory api 就被描述为统一封装多种记忆实现，取代老的只面向 KV-cache 的接口
class llama_memory_i {
public:
    virtual ~llama_memory_i() = default;

    virtual void clear() = 0; // 清空所有内部记忆，（比如整块 KV 状态或循环层的隐藏状态） -- 重置对话 / 解码用

    virtual bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) = 0; // 从序列 seq_id 里移除 [p0, p1） 这个位置区间的记忆，典型用途是裁剪上下文或丢弃回溯分支的尾部。在一些讨论/修复还能看到对 " seq_id < 0 表示任意序列" 的约定与实现差异，这也侧面说明它就是区间删除的通用入口。 
    virtual void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) = 0; // 把源序列在 [p0, p1) 这个位置区间的记忆复制到目标序列。从主轨复制一段历史到新分支继续生成。
    virtual void seq_keep(llama_seq_id seq_id) = 0;
    virtual void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos delta) = 0;
    virtual void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) = 0;

    virtual llama_pos seq_pos_min(llama_seq_id seq_id) const = 0;
    virtual llama_pos seq_pos_max(llama_seq_id seq_id) const = 0;

    virtual bool get_can_edit() const = 0;
};
