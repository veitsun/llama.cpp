#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [prompt]\n", argv[0]);
    printf("\n");
}

int main(int argc, char ** argv) {
    // path to the model gguf file
    std::string model_path;
    // prompt to generate text from
    std::string prompt = "Hello my name is";
    // number of layers to offload to the GPU
    int ngl = 99;
    // number of tokens to predict
    int n_predict = 32;

    // parse command line arguments

    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) {
                    try {
                        n_predict = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    try {
                        ngl = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                // prompt starts here
                break;
            }
        }
        if (model_path.empty()) {
            print_usage(argc, argv);
            return 1;
        }
        if (i < argc) {
            prompt = argv[i++];
            for (; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
        }
    }

    // load dynamic backends

    ggml_backend_load_all(); // 只是加载后端插件/动态库，不涉及模型权重数据

    // initialize the model

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    // 这里返回的 model 是一个 llama_model 指针，包含了模型权重，结构等静态信息。
    // 读取并构建 llama_model
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    // tokenize the prompt

    // find the number of tokens in the prompt 这个 prompt 里一共有多少 token
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    // allocate space for the tokens and tokenize the prompt 分配能存放下prompt中所有 token 的空间
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    // initialize the context

    llama_context_params ctx_params = llama_context_default_params();
    // n_ctx is the context size
    ctx_params.n_ctx = n_prompt + n_predict - 1;
    // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    ctx_params.n_batch = n_prompt;
    // enable performance counters
    ctx_params.no_perf = false;

    // llama_init_from_model 里创建对应后端（CUDA）的 buffer，并把需要 offload 到 GPU 的权重从 CPU 侧( mmap 视图或 RAM)上传到显存
    // 如果 n_gpu_layers = 0 ，就不会发生权重上传到 GPU 的过程（全部留在 CPU ，主存侧）
    llama_context * ctx = llama_init_from_model(model, ctx_params); //  基于模型创建 llama_context （计算所需的缓冲区，后端等），它们在头文件里对外声明，可以据此在 src 里跟进实现

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // initialize the sampler
    // 什么是采样。采样就是把模型前向计算得到的下一个 token 的概率分布变成下一个具体 token 的方法。

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    // 把采样做成一条采样链，若干变换器 -> 终端采样器负责最后选 token。官方接口与实现明确写着：链通常应以 greedy/dist/mirostat结尾，并在选出 token 后 accept() 更新状态
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // print the prompt token-by-token

    for (auto id : prompt_tokens) {
        char buf[128];
        // 将 给定的 llama_token 转换成文本片段，并将其存储到一个字符缓冲区中。从词汇表获取 llama_token 对应的文本片段，并将其复制到预先分配的缓冲区
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        std::string s(buf, n);
        printf("%s", s.c_str());
    }

    // prepare a batch for the prompt

    // 相当于初始化了一个包含了所有 tokens 的批次（batch），它创建并返回了一个 llama_batch 结构体，包含了 prompt_tokens 中的所有 token
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    // main loop

    const auto t_main_start = ggml_time_us();
    int n_decode = 0;
    llama_token new_token_id;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        // evaluate the current batch with the transformer model
        // llama_decode() 则是在已就绪的权重上做计算，不再主动从磁盘读取权重（但如果前面用了 mmap 的话， 操作系统就可能在首次访问某些尚未触达的张量页时继续，按页读入--这属于操作系统的懒加载行为，而不是代码里显示的读文件逻辑）
        if (llama_decode(ctx, batch)) { 
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }

        n_pos += batch.n_tokens;

        // sample the next token 基于当前模型的输出，采样生成下一个 token
        {
            new_token_id = llama_sampler_sample(smpl, ctx, -1); // 用于从模型的输出中采样下一个 token。smpl是采样器的上下文， ctx 是模型的上下文， -1 从模型的 logits 中进行采样

            // is it an end of generation? 检查是否生成结束标记 EOG（End Of Gerneration）
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            // 将 token 转换为 字符串 并输出
            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string s(buf, n);
            printf("%s", s.c_str());
            fflush(stdout);

            // prepare the next batch with the sampled token
            // 使用采样得到的 new_token_id 创建一个新的 批次，将这个新的 token 作为当前批次的一部分。 llama_batch_get_one 返回一个新的批次，其中 tokens 数组包含当前的 token new_token_id ，并且 n_tokens 为 1
            batch = llama_batch_get_one(&new_token_id, 1);

            n_decode += 1;
        }
    }

    printf("\n");

    const auto t_main_end = ggml_time_us();

    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
