#include <cstdio>
#include <cstring>
#include "arg.h"
#include "common.h"
#include "console.h"
#include "log.h"


static common_params *g_params = nullptr;

struct callback_data {
  std::vector<uint8_t> data;
};

static void print_usage(int argc, char **argv) {
	(void)argc;

	LOG("\nexample usage:\n");
  	LOG("\n  text generation:     %s -m your_model.gguf -p \"I believe the ""meaning of life is\" -n 128\n", argv[0]);
  	LOG("\n  chat (conversation): %s -m your_model.gguf -p \"You are a helpful ""assistant\" -cnv\n", argv[0]);
  	LOG("\n");
	
}

// （该文件的底层 api）
static void my_ggml_print_tensor_csv(ggml_tensor *t) {
	// 以逗号分隔的格式输出张量的所有数值
  std::vector<uint8_t> tmp;
  ggml_type type = t->type;
  const int64_t *ne = t->ne;
  const size_t *nb = t->nb;
  int64_t n = 1;
  const bool is_host = ggml_backend_buffer_is_host(t->buffer);
  if (!is_host) {
    auto n_bytes = ggml_nbytes(t);
    tmp.resize(n_bytes);
    // 取第4个元素，一个 int32_t 占4个bytes，因此offset为0，size为4*4=16
    ggml_backend_tensor_get(t, tmp.data(), 0, n_bytes);
  }
  uint8_t *data = is_host ? (uint8_t *)t->data : tmp.data();
  GGML_ASSERT(n > 0);
  float sum = 0;
  for (int64_t i3 = 0; i3 < ne[3]; i3++) {
    for (int64_t i2 = 0; i2 < ne[2]; i2++) {
      if (i2 == n && ne[2] > 2 * n) {
        i2 = ne[2] - n;
      }
      for (int64_t i1 = 0; i1 < ne[1]; i1++) {
        if (i1 == n && ne[1] > 2 * n) {
          i1 = ne[1] - n;
        }
        for (int64_t i0 = 0; i0 < ne[0]; i0++) {
          if (i0 == n && ne[0] > 2 * n) {
            i0 = ne[0] - n;
          }
          size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
          float v;
          if (type == GGML_TYPE_F16) {
            v = ggml_fp16_to_fp32(*(ggml_fp16_t *)&data[i]);
          } else if (type == GGML_TYPE_F32) {
            v = *(float *)&data[i];
          } else if (type == GGML_TYPE_I32) {
            v = (float)*(int32_t *)&data[i];
          } else if (type == GGML_TYPE_I16) {
            v = (float)*(int16_t *)&data[i];
          } else if (type == GGML_TYPE_I8) {
            v = (float)*(int8_t *)&data[i];
          } else {
            GGML_ABORT("fatal error");
          }
          printf("%.4f, ", v);
          // sum += v;
          // if (i0 < ne[0] - 1) printf(", ");
        }
      }
    }
  }
}

// (该文件的底层 api )
static std::string ggml_ne_string(const ggml_tensor *t) {
	// 该函数的主要作用是将 GGML 张量的维度信息转换为一个格式化的字符串表示。它遍历张量的 ne 数组，将每个维度的大小连接成一个逗号分隔的字符串
  std::string str;
  for (int i = 0; i < GGML_MAX_DIMS; ++i) {
    str += std::to_string(t->ne[i]); // 将每个维度的元素数量转换成字符串格式
    if (i + 1 < GGML_MAX_DIMS) {
      str += ", "; // 在每个维度值后添加逗号和空格，除了最后一个维度
    }
  }
  return str;
}

// （给main使用的中间层 api）
static bool ggml_debug_csv(struct ggml_tensor *t, bool ask, void *user_data) {
  auto *cb_data = (callback_data *)user_data;

  const struct ggml_tensor *src0 = t->src[0];
  const struct ggml_tensor *src1 = t->src[1];

  if (ask) {
    return true;  // Always retrieve data
  }

  // if (strncmp(t->name, "kq-", 3) == 0 or strncmp(t->name, "kq_", 3) == 0)
  {
    printf("%s\n", t->name);      // name 输出张量的名称
    my_ggml_print_tensor_csv(t);  // data 以逗号分隔的格式输出张量的所有数值
    printf("%s, %s, %s, %s\n", ggml_ne_string(t).c_str(), ggml_op_desc(t),
           src0 ? src0->name : "", src1 ? src1->name : "");
    //         shape,  op,    src0,  src1, 元数据信息（CSV格式）
  }
  return true;
}

static bool ggml_is_pred_callback(struct ggml_tensor *t, bool ask,
                                  void *user_data) {
  // 检查 name 是否以 pred- 开头
  if (ask) {
    return strncmp(t->name, "pred-", 5) == 0 ? true : false;
  }
  return false;
}

int main(int argc, char **argv) {
	// printf("Hello, Specee!\n");

	common_params params;
	g_params = &params;
	if(!common_params_parse(argc, argv, params, llama_example::LLAMA_EXAMPLE_MAIN, print_usage)) {
		return 1;
	}

	common_init();


	bool random_exit = false;
	if(random_exit) {
		printf("Random exit!\n");
		return 0;
	}

	if(params.verbosity) {
		params.cb_eval = ggml_debug_csv;
	} else {
		params.cb_eval = ggml_is_pred_callback;
	}
	/**
		 * @brief  这里注意一下，我把老代码里的 params.sparams 换成了 params.sampling
		 * 
		 */
	auto &sparams = params.sampling; 

	// save choice to use color for later
	// (note for later: this is a slightly awkward choice)
	console::init(params.simple_io, params.use_color);
	atexit([]() { console::cleanup(); });

	return 0;
}