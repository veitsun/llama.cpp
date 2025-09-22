#include <cstdio>
#include "arg.h"
#include "common.h"
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

static bool ggml_debug_csv(struct ggml_tensor *t, bool ask, void *user_data) {
  auto *cb_data = (callback_data *)user_data;

  const struct ggml_tensor *src0 = t->src[0];
  const struct ggml_tensor *src1 = t->src[1];

//   if (ask) {
//     return true;  // Always retrieve data
//   }

//   // if (strncmp(t->name, "kq-", 3) == 0 or strncmp(t->name, "kq_", 3) == 0)
//   {
//     printf("%s\n", t->name);      // name
//     my_ggml_print_tensor_csv(t);  // data
//     printf("%s, %s, %s, %s\n", ggml_ne_string(t).c_str(), ggml_op_desc(t),
//            src0 ? src0->name : "", src1 ? src1->name : "");
//     //         shape,  op,    src0,  src1,
//   }
  return true;
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
	}

	return 0;
}