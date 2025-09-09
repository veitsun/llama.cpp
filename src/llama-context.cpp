#include "llama-context.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-kv-cache.h"

#include <cstring>
#include <stdexcept>
#include <cinttypes>

//
// llama_context
//

// 把模型与运行参数落地为一次可用的推理上下文，并把后续推理所需的后端，内存，调度器，图以及缓冲区都准备好，从而在生成时尽可能避免临时分配和兼容性问题
llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model) {
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    // 把模型侧的计时信息拷贝过来（用于后续性能统计）
    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    // 把调用者传进来的 params 转成内部使用的 cparams ， 并做一系列未指定则用模型默认/训练值 处理
    const auto & hparams = model.hparams;
 
    cparams.n_seq_max        = std::max(1u, params.n_seq_max);
    cparams.n_threads        = params.n_threads;
    cparams.n_threads_batch  = params.n_threads_batch;
    cparams.yarn_ext_factor  = params.yarn_ext_factor;
    cparams.yarn_attn_factor = params.yarn_attn_factor;
    cparams.yarn_beta_fast   = params.yarn_beta_fast;
    cparams.yarn_beta_slow   = params.yarn_beta_slow;
    cparams.defrag_thold     = params.defrag_thold;
    cparams.embeddings       = params.embeddings;
    cparams.offload_kqv      = params.offload_kqv;
    cparams.flash_attn       = params.flash_attn;
    cparams.no_perf          = params.no_perf;
    cparams.pooling_type     = params.pooling_type;
    cparams.warmup           = false;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    auto rope_scaling_type = params.rope_scaling_type;
    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    // the batch has to be at least GGML_KQ_MASK_PAD because we will be padding the KQ_mask
    // this is required by GPU kernels in order to avoid out-of-bounds accesses (e.g. ggml_flash_attn_ext)
    // ref: https://github.com/ggerganov/llama.cpp/pull/5021
    // TODO: this padding is not needed for the cache-less context so we should probably move it to llama_context_kv_self
    if (cparams.n_batch < GGML_KQ_MASK_PAD) {
        LLAMA_LOG_WARN("%s: n_batch is less than GGML_KQ_MASK_PAD - increasing to %d\n", __func__, GGML_KQ_MASK_PAD);
        cparams.n_batch = GGML_KQ_MASK_PAD;
    }

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    cparams.op_offload = params.op_offload; // 算子异地执行

    const uint32_t n_ctx_per_seq = cparams.n_ctx / cparams.n_seq_max; // 多序列并发时等分上下文窗口

    LLAMA_LOG_INFO("%s: n_seq_max     = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx         = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_per_seq = %u\n",   __func__, n_ctx_per_seq);
    LLAMA_LOG_INFO("%s: n_batch       = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch      = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn   = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn    = %d\n",   __func__, cparams.flash_attn);
    LLAMA_LOG_INFO("%s: freq_base     = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale    = %g\n",   __func__, cparams.rope_freq_scale);

    if (n_ctx_per_seq < hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_per_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, n_ctx_per_seq, hparams.n_ctx_train);
    }

    if (n_ctx_per_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_per_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, n_ctx_per_seq, hparams.n_ctx_train);
    }

    // 若非仅词表模式，初始化各类后端
    if (!hparams.vocab_only) {
        // GPU backends
        // 对每个设备 ggml_backend_dev_init , 失败抛异常，成功放入 backends
        for (auto * dev : model.devices) { // 遍历 model.devices 里的每一个设备句柄 （ggml_backend_dev_t） 。在 GGML 里，ggml_backend_dev_t 本质上就是指向后端设备的指针类型
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr); // 把每个设备实例化成一个可用的后端。失败就抛异常，成功就收集起来。
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev))); // 如果初始化失败，直接抛异常，并打印哪个设备失败
            }
            backends.emplace_back(backend); // 前面 ggml_backend_dev_init 函数返回的值就是可用的后端句柄，成功的话就把创建好的后端放进 backends 容器中，后面执行计算/分配张量都会用到这些后端
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend 必须要有一个 CPU 后端（兜底，协调），初始化失败则抛异常，并 push 到 backends
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends 收集 set_n_threads ：对每个后端，若能拿到 ggml_backend_set_n_threads 函数指针，就记下来（后续可以动态调线程）
        // 从已经初始化的后端中，找出是否可以动态设置线程数的函数指针，然后把这些函数收集起来，方便后续调用
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) { // 如果函数指针有效，就放进 set_n_threads_fns 容器，记下这个后端和它设置线程数的函数指针
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }
        // GGML 把后端，设备，注册表设计得像一个棋盘上的棋子，职责分明。 register_backend 注册所有后端， get_proc_address 把可选能力暴露给调用方

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data); // 支持外部中止长时间推理

        // graph outputs buffer
        {
            // resized during inference when a batch uses more outputs
            if ((uint32_t) output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        llama_memory_params params_mem = {
            /*.type_k   =*/ params.type_k,
            /*.type_v   =*/ params.type_v,
            /*.swa_full =*/ params.swa_full,
        };

        // 初始化记忆模块，把注意力的 KV 历史存储形式，量化方式，是否 swa （滑动窗口注意力）等落实好，是增量解码的核心数据结构
        memory.reset(model.create_memory(params_mem, cparams)); // reset(p) 删除当前对象，memory 再接管原始指针 p 的所有权
    }

    // init backends 初始化调度器，与计算元数据缓冲
    // 下面这段代码：枚举后端 -> 预备图/元数据缓冲 -> 决定是否开启流水线并行并创建调度器
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__); 

        backend_buft.clear();
        backend_ptrs.clear();

        for (auto & backend : backends) {
            // 对每个后端取默认的 buffer_type ，若有 GPU 设备， CPU 侧有限用第一个设备的 host buffer 作为 CPU 缓冲
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            // 如果当前是 CPU 后端且存在 GPU 设备，就把 CPU 的 buffer type 改成设备的 host buffer type。 host buffer 是配合设备传输优化的主机侧内存类型，用它给 CPU 侧的中间状态做落脚点，能更快地和 GPU 互拷。
            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                auto * dev = model.devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            // 把挑好的 buffer type 和后端指针分别塞进 backend_buft、backend_ptrs 数组，后面创建调度器会用到。调度器会基于 后端 + buffer type 来决定张量驻留与算子派发
            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        const size_t max_nodes = this->graph_max_nodes();

        LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

        // buffer used to store the computation graph and the tensor meta data
        // 一次性预分配一块连续内存（CPU端），专门用来放计算图对象 + 张量元数据（metadata）
        // 不是权重数据，也不是中间激活的数值，只是描述信息（结构体，指针，列表等）。这样做能避免构图时的零碎分配，提升稳定性与性能
        // 把 buf_compute_meta 的容量改到足够容纳 max_nodes 个张量描述 + 一个计算图对象自身的开销之和
        // ggml_tensor_overhead() 返回单个 ggml 张量描述需要的字节数（相当于 struct ggml_tensor 以及必要对齐/附加的大小，不含张量的数据区）
        // ggml_graph_overhead_custom(max_nodes, false) 返回计算图对象本身以及其内部数组/表需要的总开销，按计划的节点 max_nodes 来估计
        buf_compute_meta.resize(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        // 是否启用流水线并行的判定，（仅当满足以下全部时）
        /*
        1. 设备数量大于 1
        2. n_gpu_layers > n_layer（按层切分到多设备的场景）
        3. 切分模式为层级切分
        4. 启用 KQV 的异地执行( KQV 可下放)
        */
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.params.n_gpu_layers > (int) model.hparams.n_layer &&
            model.params.split_mode == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            !model.has_tensor_overrides();

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        // 创建调度器，把后端数组， buffer types, 最大节点数， 是否流水线并行， 以及 op_offload 传进去。调度器是把计算图切分到不同后端执行的大脑
        sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, pipeline_parallel, cparams.op_offload));

        // 流水线并行能提升多设备吞吐，但占用更多内存
        // 若启用流水线并行，打印 n_copies (流水线副本数)
        if (pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled (n_copies=%d)\n", __func__, ggml_backend_sched_get_n_copies(sched.get()));
        }
    }

    // 在推理前阶段和生成阶段中，模型的计算图和内存缓冲区会有所不同
    // 为了避免在推理过程中，频繁地重新分配内存，llama.cpp 采用，预先构建计算图（为预填充阶段和生成阶段分别构建计算图），预留内存缓冲区（通过调度器为计算图预留内存缓冲区），避免内存重分配，三种策略
    // reserve worst-case graph
    if (!hparams.vocab_only && memory) {
        const uint32_t n_seqs = 1; // TODO: worst-case number of sequences 最坏情况下的序列数
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch); // 即预填充阶段的最大 token 数，这通常是最吃内存的，越长的 prefill 越占内存，因此提前按这个上限预留，最坏情况下的最大 token数 

        llama_token token = model.vocab.token_bos(); // not actually used by llama_build_graph, but required to choose between token and embedding inputs graph

        // restore later
        // TODO: something cleaner
        const auto n_outputs_save = n_outputs;

        LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

        int n_splits_pp = -1;
        int n_nodes_pp  = -1;

        int n_splits_tg = -1;
        int n_nodes_tg  = -1;

        // 将 kv 缓存标记为满，模拟历史对话已写满 KV 的状态（模拟 KV 缓存已满的状态，通过将 KV 缓存标记为已满，模拟历史对话已写满 KV 的状态，以测试在这种情况下的内存分配）
        // simulate full KV cache
        llama_kv_cache * kv_self = static_cast<llama_kv_cache *>(memory.get());

        kv_self->set_full(); 

        cross.v_embd.clear();

        // reserve pp graph first so that buffers are only allocated once
        // 先为 prefill（pp）大批量前向建图并预留，然后为 token-generation （TG，bs = 1 ） 建图并预留，最后再预留一次 PP， 以确保 ggml-alloc 在真实推理中的图切换不会触发 reallocate
        // 先为 PP 阶段按最坏情况建计算图，并向调度器要一整套计算缓冲，并把图的一些统计信息（节点数，切分数）记下来，如果申请失败就抛出异常。这样能在真正推理时避免临时分配/重排，运行更稳定
        {
            // 构建一份 ubatch 描述，告诉后面的建图逻辑：这次要跑的是 prefill 形态（一次吃很多个 token）
            /*
            n_tokens 是本次预留按“最坏情况” 取的 token数，通常等于 min(n_ctx, n_ubatch)， prefill 最吃内存
            */
            llama_ubatch ubatch_pp = { true, n_tokens, n_tokens / n_seqs, n_seqs, &token, nullptr, nullptr, nullptr, nullptr, nullptr};

            // max number of outputs
            n_outputs = ubatch_pp.n_tokens; // 暂时把 n_outputs 设成这次图的最大输出数（对 PP 来说就是这批 token 数）。有些缓冲/图结构会依据可产生的输出数量来定尺寸。

            LLAMA_LOG_DEBUG("%s: reserving graph for n_tokens = %d, n_seqs = %d\n", __func__, ubatch_pp.n_tokens, ubatch_pp.n_seqs);

            auto * gf = graph_init(); // 创建一张空的计算图句柄
            graph_build(ctx_compute.get(), gf, ubatch_pp, LLM_GRAPH_TYPE_DEFAULT); // 构建计算图

            if (!ggml_backend_sched_reserve(sched.get(), gf)) { // 这张图交给后端调度器 sched 去 reserve。这会在各个后端上预分配这张图运行所需的中间计算缓冲/工作区，并在必要时对图做切分以适配显存/内存。失败就报异常。
                // 另外一个相关事实：每次为一张新图做分配，之前那张图所绑定的临时张量/缓冲可能会被调度器重用或者覆盖，因此后面代码会先预留 PP，再预留 TG，然后再预留一次 PP，让最终布局稳定在能同时覆盖两类场景的状态
                throw std::runtime_error("failed to allocate compute pp buffers");
            }

            n_splits_pp = ggml_backend_sched_get_n_splits(sched.get());
            n_nodes_pp  = ggml_graph_n_nodes(gf);
        } // llama.cpp 里会为 PP（多 token） 和 TG（单 token）各建一套图并预留缓冲，这是常见做法

        // reserve with tg graph to get the number of splits and nodes  为预填充阶段构建计算图并预留内存
        {
            llama_ubatch ubatch_tg = { true, 1, 1, n_seqs, &token, nullptr, nullptr, nullptr, nullptr, nullptr};

            n_outputs = ubatch_tg.n_tokens; // 设置输出数量

            LLAMA_LOG_DEBUG("%s: reserving graph for n_tokens = %d, n_seqs = %d\n", __func__, ubatch_tg.n_tokens, ubatch_tg.n_seqs);

            auto * gf = graph_init();  // 创建计算图句柄
            graph_build(ctx_compute.get(), gf, ubatch_tg, LLM_GRAPH_TYPE_DEFAULT); // 构建计算图

            if (!ggml_backend_sched_reserve(sched.get(), gf)) { // 预留内存
                throw std::runtime_error("failed to allocate compute tg buffers");
            }

            n_splits_tg = ggml_backend_sched_get_n_splits(sched.get());
            n_nodes_tg  = ggml_graph_n_nodes(gf);
        }

        // reserve again with pp graph to avoid ggml-alloc reallocations during inference 为生成阶段构建计算图并预留内存
        {
            llama_ubatch ubatch_pp = { true, n_tokens, n_tokens / n_seqs, n_seqs, &token, nullptr, nullptr, nullptr, nullptr, nullptr};

            n_outputs = ubatch_pp.n_tokens; // 设置输出数量

            LLAMA_LOG_DEBUG("%s: reserving graph for n_tokens = %d, n_seqs = %d\n", __func__, ubatch_pp.n_tokens, ubatch_pp.n_seqs); // 

            auto * gf = graph_init(); // 创建计算图句柄
            graph_build(ctx_compute.get(), gf, ubatch_pp, LLM_GRAPH_TYPE_DEFAULT); // 构建计算图

            if (!ggml_backend_sched_reserve(sched.get(), gf)) { // 预留内存
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        n_outputs = n_outputs_save;

        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];
            size_t size = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size > 1) {
                LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                        ggml_backend_buft_name(buft),
                        size / 1024.0 / 1024.0);
            }
        }

        // 输出计算图的节点数和切分数，以供调试和优化
        if (n_nodes_pp == n_nodes_tg) {
            LLAMA_LOG_INFO("%s: graph nodes  = %d\n", __func__, n_nodes_pp);
        } else {
            LLAMA_LOG_INFO("%s: graph nodes  = %d (with bs=%d), %d (with bs=1)\n", __func__, n_nodes_pp, n_tokens, n_nodes_tg);
        }

        if (n_splits_pp == n_splits_tg) {
            LLAMA_LOG_INFO("%s: graph splits = %d\n", __func__, n_splits_pp);
        } else {
            LLAMA_LOG_INFO("%s: graph splits = %d (with bs=%d), %d (with bs=1)\n", __func__, n_splits_pp, n_tokens, n_splits_tg);
        }
    }
}

llama_context::~llama_context() {
    ggml_opt_free(opt_ctx);
}

/**
 * @brief 主要负责同步计算并更新与模型推理相关的额性能统计数据。它涉及了模型推理（例如语言模型的生成） 的一些关键时序管理和性能计时。
 * 
 */
void llama_context::synchronize() {
    // sched 可能包含待执行的任务
    ggml_backend_sched_synchronize(sched.get()); // 确保在统计数据更新或下一步操作前，所有计算任务都已经完成

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats  
    if (n_queued_tokens == 1) {
        // 如果处理的是单个 token， t_eval_us 会记录从 t_compute_start_us 到当前时间的插值
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0; // 重置排队的 tokens 数量，表示当前没有待处理的 tokens
    t_compute_start_us = 0; // 重置计算开始的时间戳，准备下一轮的计算
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

ggml_context * llama_context::get_ctx_compute() const {
    return ctx_compute.get();
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_per_seq() const {
    return cparams.n_ctx / cparams.n_seq_max;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_kv_cache * llama_context::get_kv_self() {
    llama_kv_cache * kv_self = static_cast<llama_kv_cache *>(memory.get());
    return kv_self;
}

const llama_kv_cache * llama_context::get_kv_self() const {
    llama_kv_cache * kv_self = static_cast<llama_kv_cache *>(memory.get());
    return kv_self;
}

/**
 * @brief llama_context::kv_self_update() 用来同步并检查自有 KV 缓存（self KV cache，注意不是 cross-attention 的 KV），如果发现 KV 的布局或容量变了，需要预先为最坏情况构建一张计算图（graph）并把后端调度器的计算/显存缓冲区一次性预留好。这样后面真实推理时就不会因为 batch 长度或上下文长度变化而频繁重新分配显存/内存，减少抖动。
 * 
 */
void llama_context::kv_self_update() {
    bool need_reserve = false;

    llama_kv_cache * kv_self = static_cast<llama_kv_cache *>(memory.get());

    need_reserve = kv_self->update(*this); // 更新 KV 状态，并告诉你是否需要重新做一次容量级的准备工作

    // reserve a worst case graph if needed
    if (need_reserve) {
        LLAMA_LOG_DEBUG("%s: reserving a worst case graph\n", __func__);

        // build worst-case graph
        uint32_t n_seqs = 1; // TODO: worst-case number of sequences
        uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        // simulate full KV cache
        kv_self->set_full(); // 把 KV 标记为满来驱动最大需求的形状推导；这一步一般不去改动真实数据，只是影响图构建时的维度计算

        llama_token token = model.vocab.token_bos(); // not actually used by llama_build_graph, but required to choose between token and embedding inputs graph
        llama_ubatch ubatch = { true, n_tokens, n_tokens / n_seqs, n_seqs, &token, nullptr, nullptr, nullptr, nullptr, nullptr};

        auto * gf = graph_init();
        graph_build(ctx_compute.get(), gf, ubatch, LLM_GRAPH_TYPE_DEFAULT);

        // initialize scheduler with the worst-case graph
        ggml_backend_sched_reset(sched.get());
        if (!ggml_backend_sched_reserve(sched.get(), gf)) {  // 后端把需要的缓冲区一口气申请出来，避免后面频繁 realloc。
            LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        }
    }
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    return logits;
}

float * llama_context::get_logits_ith(int32_t i) {
    int32_t j = -1;

    try {
        if (logits == nullptr) {
            throw std::runtime_error("no logits");
        }

        if (i < 0) {
            j = n_outputs + i;
            if (j < 0) {
                throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
            }
        } else if ((size_t) i >= output_ids.size()) {
            throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
        } else {
            j = output_ids[i];
        }

        if (j < 0) {
            throw std::runtime_error(format("batch.logits[%d] != true", i));
        }
        if (j >= n_outputs) {
            // This should not happen
            throw std::runtime_error(format("corrupt output buffer (j=%d, n_outputs=%d)", j, n_outputs));
        }

        return logits + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings() {
    return embd;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    int32_t j = -1;

    try {
        if (embd == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        if (i < 0) {
            j = n_outputs + i;
            if (j < 0) {
                throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
            }
        } else if ((size_t) i >= output_ids.size()) {
            throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
        } else {
            j = output_ids[i];
        }

        if (j < 0) {
            throw std::runtime_error(format("batch.logits[%d] != true", i));
        }
        if (j >= n_outputs) {
            // This should not happen
            throw std::runtime_error(format("corrupt output buffer (j=%d, n_outputs=%d)", j, n_outputs));
        }

        return embd + j*model.hparams.n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
        if (set_abort_callback_fn) {
            set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    cparams.embeddings = value;
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    cparams.causal_attn = value;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    cparams.warmup = value;
}

void llama_context::set_adapter_lora(
            llama_adapter_lora * adapter,
            float scale) {
    LLAMA_LOG_DEBUG("%s: adapter = %p, scale = %f\n", __func__, (void *) adapter, scale);

    loras[adapter] = scale;
}

bool llama_context::rm_adapter_lora(
            llama_adapter_lora * adapter) {
    LLAMA_LOG_DEBUG("%s: adapter = %p\n", __func__, (void *) adapter);

    auto pos = loras.find(adapter);
    if (pos != loras.end()) {
        loras.erase(pos);
        return true;
    }

    return false;
}

void llama_context::clear_adapter_lora() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    loras.clear();
}

bool llama_context::apply_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    return cvec.apply(model, data, len, n_embd, il_start, il_end);
}

int llama_context::encode(llama_batch & inp_batch) {
    if (inp_batch.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    // temporary allocate memory for the input batch if needed
    // note: during encode, we always pass the full sequence starting from pos = 0
    llama_batch_allocr batch_allocr(inp_batch, inp_batch.pos ? -1 : 0);

    const llama_batch & batch = batch_allocr.batch;
    const int32_t n_tokens = batch.n_tokens;

    const auto & hparams = model.hparams;

    GGML_ASSERT((!batch.token && batch.embd) || (batch.token && !batch.embd)); // NOLINT

    if (batch.token) {
        for (int32_t i = 0; i < n_tokens; ++i) {
            if (batch.token[i] < 0 || (uint32_t) batch.token[i] >= model.vocab.n_tokens()) {
                LLAMA_LOG_ERROR("%s: invalid token[%d] = %d\n", __func__, i, batch.token[i]);
                return -1;
            }
        }
    }

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= (uint32_t) n_tokens && "encoder requires n_ubatch >= n_tokens");

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    embd_seq.clear();

    n_queued_tokens += n_tokens;

    const int64_t n_embd = hparams.n_embd;

    llama_sbatch sbatch = llama_sbatch(batch, n_embd, /* simple_split */ true, /* logits_all */ true);

    const llama_ubatch ubatch = sbatch.split_simple(n_tokens);

    // reserve output buffer
    if (output_reserve(n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (int32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    n_outputs = n_tokens;

    //batch_manager->prepare(ubatch);

    ggml_backend_sched_reset(sched.get());
    ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    auto * gf = graph_init();
    auto res = graph_build(ctx_compute.get(), gf, ubatch, LLM_GRAPH_TYPE_ENCODER);

    ggml_backend_sched_alloc_graph(sched.get(), gf);

    res->set_inputs(&ubatch);

    cparams.causal_attn = causal_attn_org;

    const auto compute_status = graph_compute(gf, n_tokens > 1);
    switch (compute_status) {
        case GGML_STATUS_SUCCESS:
            break;
        case GGML_STATUS_ABORTED:
            return 2;
        case GGML_STATUS_ALLOC_FAILED:
            return -2;
        case GGML_STATUS_FAILED:
        default:
            return -3;
    }

    auto * t_embd = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();

    // extract embeddings
    if (t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd != nullptr);

                    GGML_ASSERT(n_tokens*n_embd <= (int64_t) embd_size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd, 0, n_tokens*n_embd*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;
                    embd_seq_out.clear();

                    GGML_ASSERT(!ubatch.equal_seqs); // TODO: handle equal splits

                    for (int32_t i = 0; i < n_tokens; i++) {
                        const llama_seq_id seq_id = ubatch.seq_id[i][0];
                        if (embd_seq_out.find(seq_id) != embd_seq_out.end()) {
                            continue;
                        }
                        embd_seq_out[seq_id].resize(n_embd);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd*seq_id)*sizeof(float), n_embd*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - a single float per sequence
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs; ++s) {
                        const llama_seq_id seq_id = ubatch.seq_id[s][0];
                        if (embd_seq_out.find(seq_id) != embd_seq_out.end()) {
                            continue;
                        }
                        embd_seq_out[seq_id].resize(1);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (seq_id)*sizeof(float), sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(sched.get());

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd, ggml_nbytes(t_embd));

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (int32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();
            for (int s = 0; s < ubatch.n_seq_id[i]; s++) {
                llama_seq_id seq_id = ubatch.seq_id[i][s];
                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

/**
 * @brief 把用户给的 batch 规范化 -> 拆成一个或多个 ubatch（物理计算块） -> 为每个 ubatch 构图并在 GGML 后端执行 -> 从后端异步拷回 logits/embedding -> 维护与提交 KV 内存 -> 必要时安排碎片整理
 * 
 * @param inp_batch 
 * @return int 
 */
int llama_context::decode(llama_batch & inp_batch) {
    if (!memory) {
        LLAMA_LOG_WARN("%s: cannot decode batches with this context (use llama_encode() instead)\n", __func__);
        return encode(inp_batch);
    }

    if (inp_batch.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    if (!inp_batch.pos) {
        if (inp_batch.seq_id) {
            LLAMA_LOG_ERROR("%s: pos == NULL, but seq_id != NULL\n", __func__);
            return -1;
        }
    }

    llama_kv_cache * kv_self = static_cast<llama_kv_cache *>(memory.get());

    // temporary allocate memory for the input batch if needed
    // 用 llama_batch_allocr 生成一个规范化的 batch 视图：如果用户没有提供 pos，它会基于 KV 中当前序列的最大位置 seq_pos_max 续接位置，也会为后续按 ubatch 切分准备各种索引映射与一致性检查
    llama_batch_allocr batch_allocr(inp_batch, inp_batch.pos ? -1 : kv_self->seq_pos_max(0) + 1);  // 构造一个批次分配器。如果调用方已经提供了 batch.pos ，就传 -1 表示“别替我填位置”。 否则用 KV 缓存里 序列 0 的最大已用位置 + 1 作为起始位置，自动补齐本次 token 的pos。当该序列还没有写入过 KV 时， seq_pos_max 返回 -1 ，于是起点会是 0

    const llama_batch & batch = batch_allocr.batch;

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int32_t n_vocab = vocab.n_tokens(); // 词表大小

    const int64_t n_tokens_all = batch.n_tokens; // 本次要处理的 token 数
    const int64_t n_embd       = hparams.n_embd; // 模型的 embedding 维度

    llama_kv_cache_guard kv_guard(kv_self); // 典型的 RALL 守卫：如果后续批处理失败，它就会回滚 KV；成功则提交。

    GGML_ASSERT((!batch.token && batch.embd) || (batch.token && !batch.embd)); // NOLINT 要么喂 token，要么喂外部 embedding，不能同时两边插队。

    if (batch.token) {
        for (int64_t i = 0; i < n_tokens_all; ++i) {
            if (batch.token[i] < 0 || (uint32_t) batch.token[i] >= model.vocab.n_tokens()) {
                LLAMA_LOG_ERROR("%s: invalid token[%" PRId64 "] = %d\n", __func__, i, batch.token[i]);
                throw std::runtime_error("invalid token");
            }
        }
    }

    // 批大小约束， n_batch 是逻辑批大小（应用侧一次希望输出/保留多少个位置的结果）
    GGML_ASSERT(n_tokens_all <= cparams.n_batch); // 本次要处理的 token 数 不能超过 n_batch。在 llama.cpp 里 batch-size 是应用层缓冲
    // n_ubatch 是物理计算块大小（受设备/后端能力限制的每次最大并行 token 数）
    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    // this indicates we are doing pooled embedding, so we ignore batch.logits and output all tokens
    const bool embd_pooled = cparams.embeddings && cparams.pooling_type != LLAMA_POOLING_TYPE_NONE;

    embd_seq.clear();

    // 这块是统计本批需要从模型取回多少个输出的核心：
    /**
         * @brief 普通解码（非 pooled embedding）且 batch.logits 非空：遍历标志数组，只有 batch.logits[i] != 0 的那些位置会产出输出（logits / embedding）。llama_batch.logits 的设计语义正是“哪些 token 位置要回传输出”；这些位置的结果会被按出现顺序连续存放。

pooled embedding 模式：把 n_outputs_all 直接设为 n_tokens_all，也就是对每个 token 位置都先收集（供池化用），因此忽略 batch.logits 的选择性回传逻辑。注意：对外可见的结果在 pooling≠NONE 时通常是“每个序列一个池化后的向量”（而不是每个 token 一个），但在内部计数/收集阶段会先按 token 全量准备，再在图里做池化与归一化。

既没有 batch.logits，也不是 pooled embedding：默认只保留“最后一个 token”的输出（典型的自回归解码场景），这是 llama.h 对“logits == NULL 仅返回最后一个 token 的 logits”这条 API 约定在实现层的体现。

batch.logits[i] 是一个输出掩码：1 表示“请给我第 i 个 token 的输出”，0 表示忽略。没有提供时，默认只给最后一个；提供了就按位挑选，并把这些位置的结果连续返回
         * 
         */
    int64_t n_outputs_all = 0;

    // count outputs
    if (batch.logits && !embd_pooled) {
        for (uint32_t i = 0; i < n_tokens_all; ++i) {
            n_outputs_all += batch.logits[i] != 0;   // 遍历标志数组，只有 batch.logits[i] != 0 的那些位置会产生输出
        }
    } else if (embd_pooled) {
        n_outputs_all = n_tokens_all;
    } else {
        // keep last output only
        n_outputs_all = 1; // 默认只返回 最后一个 token 的输出 （典型的自回归解码场景）
    }

    // 这里把用户 batch 转成内部“顺序批”（sbatch），并指明是否“对所有 token 取日志”。
    llama_sbatch sbatch = kv_self->sbatch_init(batch, /* logits_all */ n_outputs_all == n_tokens_all);

    // reserve output buffer 接着预留 host 侧的 logits / embeddings 输出缓冲，以便从后端拉数据。
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %" PRId64 " outputs\n", __func__, n_outputs_all);
        return -2;
    };

    // handle any pending defrags/shifts
    // 处理任何挂起的 KV “移位/整理”请求（如上下文滑动、碎片整理），以保证后续找 KV 槽位时成功率更高。近期对 KV 更新/碎片整理的机制有过多次重构与修复。
    // KV一有分吹草动，就用“满仓 KV + 最大批次”构一张保守的大图，立刻把后端的内存/显存预热到位，从而让后续推理一路丝滑
    kv_self_update();

    int64_t n_outputs_prev = 0;

    while (sbatch.n_tokens > 0) {
        // 主循环：按 ubatch 执行，从后端取回结果
        llama_ubatch ubatch = kv_self->ubatch_next(sbatch, cparams.n_ubatch, embd_pooled);

        // count the outputs in this u_batch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                GGML_ASSERT(ubatch.output);
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        // find KV slot
        // 为 ubatch 找 KV 槽位
        // 若 KV 碎片化严重，可能一时找不到可用槽位，需要触发（或放宽阈值以自动触发）defrag；相关的 issue/PR 也在不断改进这部分的鲁棒性
        // ubatch 是带有批处理相关的信息
        if (!kv_self->find_slot(ubatch)) {
            return 1; // 找不到 KV 槽
        }

        // 调度器的重置和回调设置
        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data); // 为调度器设置一个评估回调函数 cb_eval，以及用户数据 cb_eval_user_data。这个回调函数可能在推理过程中被触发，用于执行一些自定义的操作。

        // 这里调用解码器图的构建，把图交给多后端调度器  ggml_backend_sched 来分配/执行；失败时按状态码分类返回。项目讨论里对调度器如何切分/分配，何时 get/set_tensor 都有说明
        auto * gf = graph_init();
        auto res = graph_build(ctx_compute.get(), gf, ubatch, LLM_GRAPH_TYPE_DECODER);

        // LLAMA_LOG_INFO("graph build time: %.3f ms (%d nodes, %d leafs)\n", (ggml_time_us() - t_start_us)/1000.0, gf->n_nodes, gf->n_leafs);

        // 调度器分配计算资源
        ggml_backend_sched_alloc_graph(sched.get(), gf); // 由调度器负责分配计算资源，如内存和计算设备（CPU/GPU）。这一步确保了计算图中的每个节点都被正确地分配到适当的计算设备上

        // 将输入数据 ubatch 设置到图 gf 中，意味着图中的每个节点都会获得相应的输入数据，准备进行计算
        res->set_inputs(&ubatch);

        // 计算状态检查和错误处理
        const auto compute_status = graph_compute(gf, ubatch.n_tokens > 1); // 执行计算图，开始实际的推理计算
        if (compute_status != GGML_STATUS_SUCCESS) {
            switch (compute_status) {
                case GGML_STATUS_ABORTED:
                    return 2;
                case GGML_STATUS_ALLOC_FAILED:
                    return -2;
                case GGML_STATUS_FAILED:
                default:
                    return -3;
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        // 这两行代码根据 cparams.embedding 的值来决定是获取logits或者是embdding
        auto * t_logits = cparams.embeddings ? nullptr         : res->get_logits();
        auto * t_embd   = cparams.embeddings ? res->get_embd() : nullptr;

        // 如果有 pooled embedding ，切换 embedding
        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // extract logits
        // 这段代码的主要功能是处理 logits 和 embedding 数据的提取，并将计算结果异步传回到主机端。它会根据不同的条件选择数据类型，并通过调度器和后端进行异步数据获取。
        if (t_logits && n_outputs > 0) {
            // 异步拉回 logits/embedding ，先通过调度器拿到结果 tensor 所在后端句柄，再用 *_get_async() 把数据搬回 host 缓冲。维护者解释过：不同后端存储布局可能不同，set_tensor/get_tensor 层会处理必要的交换，且通常只在取得计算结果或跨后端拷贝时调用。
            // 获取后端句柄
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits != nullptr);

            // 计算 logits_out 的指针位置。假设 logits 是一个连续的内存块，logits_out 是结果存储的起始位置。通过 n_outputs_prev*n_vocab 计算偏移量，确保每个批次的结果不会覆盖之前的结果。
            float * logits_out = logits + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits_size);
                // 异步获取 logits 数据
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }

        // extract embeddings
        if (t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            // 支持多种 embedding 池化
            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd != nullptr);
                        float * embd_out = embd + n_outputs_prev*n_embd;

                        if (n_outputs) {
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd <= (int64_t) embd_size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        for (uint32_t s = 0; s < ubatch.n_seqs; ++s) {
                            const llama_seq_id seq_id = ubatch.seq_id[s][0];
                            if (embd_seq_out.find(seq_id) != embd_seq_out.end()) {
                                continue;
                            }
                            embd_seq_out[seq_id].resize(n_embd);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd*seq_id)*sizeof(float), n_embd*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - a single float per sequence
                        auto & embd_seq_out = embd_seq;

                        for (uint32_t s = 0; s < ubatch.n_seqs; ++s) {
                            const llama_seq_id seq_id = ubatch.seq_id[s][0];
                            if (embd_seq_out.find(seq_id) != embd_seq_out.end()) {
                                continue;
                            }
                            embd_seq_out[seq_id].resize(1);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (seq_id)*sizeof(float), sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        n_outputs_prev += n_outputs;
    }

    // 提交 KV，更换批，安排碎片整理
    // finalize the batch processing
    // 提交 KV 改动
    kv_guard.commit();

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all; // 供 llama_get_logits_ith 使用

    // set output mappings
    // 这段代码的主要功能是处理映射，确保输出的顺序与用户传入 batch 顺序一致，尤其是在需要保留序列顺序的情况下，如递归模型（Recurrent Models）或并行处理时
    {
        bool sorted_output = true; 

        auto & out_ids = sbatch.out_ids; // 这是模型在处理批次时输出的 token id 的列表。代码首先检查 out_ids 是否是按照递增顺序排列的

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs_all);

        for (int64_t i = 0; i < n_outputs_all; ++i) {
            int64_t out_id = out_ids[i]; 
            output_ids[out_id] = i; // 这一步将每个输出的 ID 映射到它在当前批次中的位置（即 i）
            if (out_id != i) {
                sorted_output = false; // sorted_output 这是一个标志位，检查输出 ID 是否与其索引一致。如果发现某个 out_id 与其索引不一致，标志位 sorted_output 会被设置为 false，表示输出顺序被打乱
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        // 若输出顺序被打乱，则重新排序
        if (!sorted_output) {
            const uint32_t n_vocab = model.vocab.n_tokens();
            const uint32_t n_embd  = model.hparams.n_embd;

            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            // 使用选择排序重新排序
            // 对于 out_ids 数组，它会遍历所有元素，找到最小的元素并交换位置，从而确保 out_ids 是有序的
            for (int32_t i = 0; i < n_outputs - 1; ++i) {
                int32_t j_min = i;
                for (int32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) { continue; }
                std::swap(out_ids[i], out_ids[j_min]);
                if (logits_size > 0) {
                    for (uint32_t k = 0; k < n_vocab; k++) {
                        std::swap(logits[i*n_vocab + k], logits[j_min*n_vocab + k]);
                    }
                }
                if (embd_size > 0) {
                    for (uint32_t k = 0; k < n_embd; k++) {
                        std::swap(embd[i*n_embd + k], embd[j_min*n_embd + k]);
                    }
                }
            }
            std::fill(output_ids.begin(), output_ids.end(), -1);
            for (int32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    // decide if we need to defrag the kv cache
    // 按阈值安排 KV defrag
    if (cparams.defrag_thold > 0.0f) {
        // 这不会立刻阻塞式整理，而是把整理请求安排进日程，由后续 kv_self_update() 等环节择机执行。 KV 碎片整理相关议题在项目里很多：原因是长上下文，并发/复用下很容易产生“洞”；近期也在迭代更好的整理策略与 bug 修复
        kv_self->defrag_sched(cparams.defrag_thold);
    }

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(sched.get()); // 调度器 reset，为下一轮做准备

    return 0;
}

//
// output
//

int32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch = cparams.n_batch;
    const auto n_vocab = vocab.n_tokens();
    const auto n_embd  = hparams.n_embd;

    // TODO: use a per-batch flag for logits presence instead
    bool has_logits = !cparams.embeddings;
    bool has_embd   =  cparams.embeddings && (cparams.pooling_type == LLAMA_POOLING_TYPE_NONE);

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }

    logits_size = has_logits ? n_vocab*n_outputs_max : 0;
    embd_size   = has_embd   ?  n_embd*n_outputs_max : 0;

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size  = (logits_size + embd_size) * sizeof(float);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_INFO("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            buf_output = nullptr;
            logits = nullptr;
            embd = nullptr;
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    logits = has_logits ? output_base               : nullptr;
    embd   = has_embd   ? output_base + logits_size : nullptr;

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs     = 0;
    this->n_outputs_max = n_outputs_max;

    return n_outputs_max;
}

//
// graph
//

int32_t llama_context::graph_max_nodes() const {
    return std::max<int32_t>(65536, 5*model.n_tensors());
}

ggml_cgraph * llama_context::graph_init() {
    ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta.size(),
        /*.mem_buffer =*/ buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ctx_compute.reset(ggml_init(params));

    return ggml_new_graph_custom(ctx_compute.get(), graph_max_nodes(), false);
}

llm_graph_result_ptr llama_context::graph_build(
            ggml_context * ctx,
             ggml_cgraph * gf,
      const llama_ubatch & ubatch,
            llm_graph_type gtype) {
    return model.build_graph(
            {
                /*.ctx         =*/ ctx,
                /*.arch        =*/ model.arch,
                /*.hparams     =*/ model.hparams,
                /*.cparams     =*/ cparams,
                /*.ubatch      =*/ ubatch,
                /*.sched       =*/ sched.get(),
                /*.backend_cpu =*/ backend_cpu,
                /*.cvec        =*/ &cvec,
                /*.loras       =*/ &loras,
                /*.memory      =*/ memory.get(),
                /*.cross       =*/ &cross,
                /*.n_outputs   =*/ n_outputs,
                /*.cb          =*/ graph_get_cb(),
            }, gf, gtype);
}

/**
 * @brief 为一次图(gf)计算 配置并行度与线程池，然后把图提交给多后端调度器异步执行
 * 
 * @param gf 
 * @param batched 
 * @return ggml_status 
 */
ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    // 根据 batched 选择并行配置，为什么要区分：批处理时算子工作量更大，适合开更过线程。单 token 时工作量小，线程多了反而会受同步/调度开销影响
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;  // 要给 CPU 后端设置的线程池对象（ggml自己的线程池），批处理和非批处理各用一个， 减少和其他任务的抢占和过度订阅

    if (backend_cpu != nullptr) {  // 给 CPU 后端注入线程池
        // 通过设备注册表 reg 动态查找（dlsym 风格）CPU 后端的 ggml_backend_cpu_set_threadpool 符号，然后把选好的线程池 tp 设置给 backend_cp。
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        // 之所以用“查符号”的方式，是为了避免在编译期强依赖某个具体后端头文件；统一通过后端注册表按名字拿函数指针，便于可插拔的后端体系（CPU、CUDA、Metal、Vulkan…）。
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        set_threadpool_fn(backend_cpu, tp);
    }

    // set the number of threads for all the backends
    // 给所有后端设置算子级线程数
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    // 异步提交图到调度器执行
    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        if (!cparams.offload_kqv) {
            if (strcmp(name, "kqv_merged_cont") == 0) {
                // all nodes between the KV store and the attention output are run on the CPU
                ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend_cpu);
            }
        }

        // norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.params.n_gpu_layers > (int) model.hparams.n_layer;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && strcmp(name, "norm") == 0) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy() = default;

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(const ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    size_t size_written = 0;
};

class llama_io_write_buffer : public llama_io_write_i {
public:
    llama_io_write_buffer(
            uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(const ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        ggml_backend_tensor_get(tensor, ptr, offset, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;
};

class llama_io_read_buffer : public llama_io_read_i {
public:
    llama_io_read_buffer(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    const uint8_t * read(size_t size) override {
        const uint8_t * base_ptr = ptr;
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        ptr += size;
        size_read += size;
        buf_size -= size;
        return base_ptr;
    }

    void read_to(void * dst, size_t size) override {
        memcpy(dst, read(size), size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(const ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read_to(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    const uint8_t * read(size_t size) override {
        temp_buffer.resize(size);
        read_to(temp_buffer.data(), size);
        return temp_buffer.data();
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io;
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_buffer io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_buffer io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_size(llama_seq_id seq_id) {
    llama_io_write_dummy io;
    try {
        return state_seq_write_data(io, seq_id);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size) {
    llama_io_write_buffer io(dst, size);
    try {
        return state_seq_write_data(io, seq_id);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size) {
    llama_io_read_buffer io(src, size);
    try {
        return state_seq_read_data(io, seq_id);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    // write output ids
    {
        LLAMA_LOG_DEBUG("%s: - writing output ids\n", __func__);

        const auto n_outputs    = this->n_outputs;
        const auto & output_ids = this->output_ids;

        std::vector<int32_t> w_output_pos;

        GGML_ASSERT(n_outputs <= n_outputs_max);

        w_output_pos.resize(n_outputs);

        // build a more compact representation of the output ids
        for (size_t i = 0; i < n_batch(); ++i) {
            // map an output id to a position in the batch
            int32_t pos = output_ids[i];
            if (pos >= 0) {
                GGML_ASSERT(pos < n_outputs);
                w_output_pos[pos] = i;
            }
        }

        io.write(&n_outputs, sizeof(n_outputs));

        if (n_outputs) {
            io.write(w_output_pos.data(), n_outputs * sizeof(int32_t));
        }
    }

    // write logits
    {
        LLAMA_LOG_DEBUG("%s: - writing logits\n", __func__);

        const uint64_t logits_size = std::min((uint64_t) this->logits_size, (uint64_t) n_outputs * model.vocab.n_tokens());

        io.write(&logits_size, sizeof(logits_size));

        if (logits_size) {
            io.write(logits, logits_size * sizeof(float));
        }
    }

    // write embeddings
    {
        LLAMA_LOG_DEBUG("%s: - writing embeddings\n", __func__);

        const uint64_t embd_size = std::min((uint64_t) this->embd_size, (uint64_t) n_outputs * model.hparams.n_embd);

        io.write(&embd_size, sizeof(embd_size));

        if (embd_size) {
            io.write(embd, embd_size * sizeof(float));
        }
    }

    llama_kv_cache * kv_self = static_cast<llama_kv_cache *>(memory.get());

    if (kv_self != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing KV self\n", __func__);
        kv_self->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    // read output ids
    {
        LLAMA_LOG_DEBUG("%s: - reading output ids\n", __func__);

        auto n_outputs = this->n_outputs;
        io.read_to(&n_outputs, sizeof(n_outputs));

        if (n_outputs > output_reserve(n_outputs)) {
            throw std::runtime_error("could not reserve outputs");
        }

        std::vector<int32_t> output_pos;

        if (n_outputs) {
            output_pos.resize(n_outputs);
            io.read_to(output_pos.data(), n_outputs * sizeof(int32_t));

            for (int32_t i = 0; i < (int32_t) output_pos.size(); ++i) {
                int32_t id = output_pos[i];
                if ((uint32_t) id >= n_batch()) {
                    throw std::runtime_error(format("invalid output id, %d does not fit in batch size of %u", id, n_batch()));
                }
                this->output_ids[id] = i;
            }

            this->n_outputs = n_outputs;
        }
    }

    // read logits
    {
        LLAMA_LOG_DEBUG("%s: - reading logits\n", __func__);

        uint64_t logits_size;
        io.read_to(&logits_size, sizeof(logits_size));

        if (this->logits_size < logits_size) {
            throw std::runtime_error("logits buffer too small");
        }

        if (logits_size) {
            io.read_to(this->logits, logits_size * sizeof(float));
        }
    }

    // read embeddings
    {
        LLAMA_LOG_DEBUG("%s: - reading embeddings\n", __func__);

        uint64_t embd_size;
        io.read_to(&embd_size, sizeof(embd_size));

        if (this->embd_size < embd_size) {
            throw std::runtime_error("embeddings buffer too small");
        }

        if (embd_size) {
            io.read_to(this->embd, embd_size * sizeof(float));
        }
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading KV self\n", __func__);

        llama_kv_cache * kv_self = static_cast<llama_kv_cache *>(memory.get());

        kv_self->state_read(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id) {
    GGML_UNUSED(seq_id);

    if (memory) {
        llama_kv_cache * kv_self = static_cast<llama_kv_cache *>(memory.get());

        kv_self->state_write(io, seq_id);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id) {
    GGML_UNUSED(seq_id);

    if (memory) {
        llama_kv_cache * kv_self = static_cast<llama_kv_cache *>(memory.get());

        kv_self->state_read(io, seq_id);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;

    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    llama_kv_cache * kv_self = static_cast<llama_kv_cache *>(memory.get());

    kv_self->clear();
    llama_kv_cache_guard kv_guard(kv_self);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        const auto n_tokens_all = batch.n_tokens;

        n_queued_tokens += n_tokens_all;

        // this indicates we are doing pooled embedding, so we ignore batch.logits and output all tokens
        const bool embd_pooled = cparams.embeddings && cparams.pooling_type != LLAMA_POOLING_TYPE_NONE;

        embd_seq.clear();

        int64_t n_outputs_all = n_tokens_all;

        llama_sbatch sbatch = kv_self->sbatch_init(batch, /*logits_all =*/ true);

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %" PRId64 " outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        for (uint32_t pos_batch = 0; pos_batch < n_batch; pos_batch += n_ubatch) {
            llama_ubatch ubatch = kv_self->ubatch_next(sbatch, cparams.n_ubatch, embd_pooled);

            n_outputs = ubatch.n_tokens;

            // TODO: not sure if this is needed
            if (!kv_self->find_slot(ubatch)) {
                LLAMA_LOG_WARN("%s: failed to find KV cache slot for ubatch of size %d\n", __func__, ubatch.n_tokens);

                GGML_ABORT("TODO: handle this error");
            }

            auto * gf = graph_init();
            auto res = graph_build(ctx_compute.get(), gf, ubatch, LLM_GRAPH_TYPE_DEFAULT);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);
            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);
        }
    }

    kv_guard.commit();
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ 1.0f,
        /*.yarn_beta_fast              =*/ 32.0f,
        /*.yarn_beta_slow              =*/ 1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.flash_attn                  =*/ false,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
    };

    return result;
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn = false;
    }

    if (ggml_is_quantized(params.type_v) && !params.flash_attn) {
        LLAMA_LOG_ERROR("%s: V cache quantization requires flash_attn\n", __func__);
        return nullptr;
    }

    try {
        auto * ctx = new llama_context(*model, params);
        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

llama_kv_cache * llama_get_kv_self(llama_context * ctx) {
    return ctx->get_kv_self();
}

void llama_kv_self_update(llama_context * ctx) {
    ctx->kv_self_update();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_logits_ith(i);
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

// llama adapter API

int32_t llama_set_adapter_lora(
            llama_context * ctx,
            llama_adapter_lora * adapter,
            float scale) {
    ctx->set_adapter_lora(adapter, scale);

    return 0;
}

int32_t llama_rm_adapter_lora(
            llama_context * ctx,
            llama_adapter_lora * adapter) {
    bool res = ctx->rm_adapter_lora(adapter);

    return res ? 0 : -1;
}

void llama_clear_adapter_lora(llama_context * ctx) {
    ctx->clear_adapter_lora();
}

int32_t llama_apply_adapter_cvec(
        llama_context * ctx,
                 const float * data,
                      size_t   len,
                     int32_t   n_embd,
                     int32_t   il_start,
                     int32_t   il_end) {
    bool res = ctx->apply_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// kv cache
//

// deprecated
int32_t llama_kv_self_n_tokens(const llama_context * ctx) {
    const auto * kv = ctx->get_kv_self();
    if (!kv) {
        return 0;
    }

    int32_t res = 0;

    for (uint32_t s = 0; s < ctx->get_cparams().n_seq_max; s++) {
        const llama_pos p0 = kv->seq_pos_min(s);
        const llama_pos p1 = kv->seq_pos_max(s);

        if (p0 >= 0) {
            res += (p1 - p0) + 1;
        }
    }

    return res;
}

// deprecated
// note: this is the same as above - will be removed anyway, so it's ok
int32_t llama_kv_self_used_cells(const llama_context * ctx) {
    const auto * kv = ctx->get_kv_self();
    if (!kv) {
        return 0;
    }

    int32_t res = 0;

    for (uint32_t s = 0; s < ctx->get_cparams().n_seq_max; s++) {
        const llama_pos p0 = kv->seq_pos_min(s);
        const llama_pos p1 = kv->seq_pos_max(s);

        if (p0 >= 0) {
            res += (p1 - p0) + 1;
        }
    }

    return res;
}

void llama_kv_self_clear(llama_context * ctx) {
    auto * kv = ctx->get_kv_self();
    if (!kv) {
        return;
    }

    kv->clear();
}

bool llama_kv_self_seq_rm(
        llama_context * ctx,
         llama_seq_id   seq_id,
            llama_pos   p0,
            llama_pos   p1) {
    auto * kv = ctx->get_kv_self();
    if (!kv) {
        return true;
    }

    return kv->seq_rm(seq_id, p0, p1);
}

void llama_kv_self_seq_cp(
        llama_context * ctx,
         llama_seq_id   seq_id_src,
         llama_seq_id   seq_id_dst,
            llama_pos   p0,
            llama_pos   p1) {
    auto * kv = ctx->get_kv_self();
    if (!kv) {
        return;
    }

    kv->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_self_seq_keep(llama_context * ctx, llama_seq_id seq_id) {
    auto * kv = ctx->get_kv_self();
    if (!kv) {
        return;
    }

    kv->seq_keep(seq_id);
}

void llama_kv_self_seq_add(
        llama_context * ctx,
         llama_seq_id   seq_id,
            llama_pos   p0,
            llama_pos   p1,
            llama_pos   delta) {
    auto * kv = ctx->get_kv_self();
    if (!kv) {
        return;
    }

    kv->seq_add(seq_id, p0, p1, delta);
}

void llama_kv_self_seq_div(
        llama_context * ctx,
         llama_seq_id   seq_id,
            llama_pos   p0,
            llama_pos   p1,
                  int   d) {
    auto * kv = ctx->get_kv_self();
    if (!kv) {
        return;
    }

    kv->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_kv_self_seq_pos_min(llama_context * ctx, llama_seq_id seq_id) {
    const auto * kv = ctx->get_kv_self();
    if (!kv) {
        return -1;
    }

    return kv->seq_pos_min(seq_id);
}

llama_pos llama_kv_self_seq_pos_max(llama_context * ctx, llama_seq_id seq_id) {
    const auto * kv = ctx->get_kv_self();
    if (!kv) {
        return -1;
    }

    return kv->seq_pos_max(seq_id);
}

void llama_kv_self_defrag(llama_context * ctx) {
    auto * kv = ctx->get_kv_self(); // 定义了一个帮忙函数，作用域只在当前翻译上下文里动手。从上下文里拿自注意力用的 KV 缓存对象。KV 缓存里存的是过往 token 的 key/value，用来加速后续解码
    if (!kv) {
        return;
    }

    // force defrag
    kv->defrag_sched(-1.0f); // 调用缓存的安排碎片整理接口。这里传 -1.0f，表示“强制整理”，不管当前碎片化程度如何。实际的整理通常在下一轮图执行/更新时发生，而不是立刻把整个程序卡住
}

bool llama_kv_self_can_shift(const llama_context * ctx) {
    const auto * kv = ctx->get_kv_self();
    if (!kv) {
        return false;
    }

    return kv->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return ctx->state_seq_get_size(seq_id);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size);
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    int ret = ctx->decode(batch); // 调用底层 ctx->decode　处理一个 batch （一批 token，可包含多条并行序列） 。
    // 返回 1 ，代表“找不到 KVCache 的空槽位”， kV 缓存满 或者 碎片化严重，无法为这批 token 分配位置

    // defrag and try again
    // TODO: distinguish return code when we are sure that even after defrag there is no space available
    // 
    if (ret == 1) {
        // 如果 KVCache 键值缓存不够用就整理 defrag ，然后再是一次 decode，按返回码记录日志并把结果返回
        llama_kv_self_defrag(ctx);
        ret = ctx->decode(batch);

        if (ret == 1) {
            LLAMA_LOG_WARN("%s: failed to find KV cache slot for batch of size %d\n", __func__, batch.n_tokens);

            return ret;
        }
    }

    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}
