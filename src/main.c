#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "ggml_model.h"
#include "inference.h"
#include "ternary_conversion.h"
#include "tokenizer.h"
#include "utils.h"
#include "log.h"

#define MAX_PROMPT_LENGTH 1024
#define MAX_TOKENS_GENERATE 100
#define BUFFER_SIZE 4096
#define CONTEXT_LEN 2048
#define TEMPERATURE 1.0f;

/**
 * @brief Print usage/help message to stdout.
 *
 * Prints a detailed usage string describing required and optional CLI
 * arguments and example usage.
 *
 * @param program_name Typically argv[0]; may be NULL in some callers.
 */
static void print_help(const char* program_name) {
    printf("\n");
    printf("================================================================================\n");
    printf("                        Sapphire Inference Engine\n");
    printf("================================================================================\n");
    printf("Usage: %s [options]\n\n", program_name);
    printf("Required Arguments:\n");
    printf("  -m, --model <name>        Model name (e.g., gemma3-270m-it)\n\n");
    printf("Optional Arguments:\n");
    printf("  -c, --context <length>    Context length (default: 2048)\n");
    printf("  -t, --temp <value>        Temperature for sampling (default: 1.0)\n");
    printf("  -n, --max-tokens <num>    Maximum tokens to generate (default: 100)\n");
    printf("  -p, --prompt <string>     Run a single prompt non-interactively and exit (echoes prompt)\n");
    printf("  --convert-ternary         Run ternary conversion mode instead of inference\n");
    printf("  --output <path>           Output file (single-layer) or output directory (full-model)\n");
    printf("  --layer <name>            Optional single-layer conversion filter\n");
    printf("  --calibration-corpus <p>  Optional text corpus file for tokenized STE calibration\n");
    printf("  --calibration-samples <n> Calibration sample count per tensor (default: 4)\n");
    printf("  --validation-corpus <p>   Optional held-out text corpus file for conversion checkpoints\n");
    printf("  --validation-samples <n>  Held-out prompt count for checkpoint evaluation\n");
    printf("  --validate-every <n>      Run checkpoint validation every N converted tensors\n");
    printf("  --kl-weight <value>       Optional KL distillation weight (default: 0.05)\n");
    printf("  --save-state <path>       Save session state to .sapphire file before exit\n");
    printf("  --load-state <path>       Load session state from .sapphire file at startup\n");
    printf("  -h, --help                Show this help message\n");
    printf("\nModel Directory Structure (required files):\n");
    printf("  model.safetensors (or model.gguf / model.bin)\n");
    printf("  tokenizer.json\n");
    printf("  tokenizer_config.json\n");
    printf("  (optional) special_tokens_map.json\n");
    printf("\nInteractive Commands:\n");
    printf("  /exit                     Exit the program\n");
    printf("  /clear                    Clear conversation history\n");
    printf("  /save <path>              Save session state to file\n");
    printf("  /load <path>              Load session state from file\n");
    printf("  /state                    Show session state visibility\n");
    printf("  /info                     Show model information\n");
    printf("  /help                     Show command help\n");
    printf("\nExample:\n");
    printf("  %s -m gemma3-270m-it -c 4096 -t 0.7 -n 200\n", program_name);
    printf("  %s -m gemma-3-27b-it --convert-ternary --output ./out/model-ternary\n", program_name);
    printf("  %s -m gemma-3-270m-it --convert-ternary --layer model.layers.0.self_attn.q_proj.weight --output ./out/layer0-qproj.safetensors\n", program_name);
    printf("\n");
}

typedef struct {
    const char* model_name;
    int context_len;
    float temperature;
    int max_tokens;
    const char* prompt_arg;
    int convert_ternary;
    const char* output_path;
    const char* layer_name;
    const char* calibration_corpus_path;
    int calibration_sample_limit;
    const char* validation_corpus_path;
    int validation_sample_limit;
    int validate_every_n;
    float kl_weight;
    const char* save_state_path;
    const char* load_state_path;
} cli_args_t;

static void cli_args_init(cli_args_t *args) {
    if (!args) return;
    args->model_name = NULL;
    args->context_len = CONTEXT_LEN;
    args->temperature = TEMPERATURE;
    args->max_tokens = MAX_TOKENS_GENERATE;
    args->prompt_arg = NULL;
    args->convert_ternary = 0;
    args->output_path = NULL;
    args->layer_name = NULL;
    args->calibration_corpus_path = NULL;
    args->calibration_sample_limit = 4;
    args->validation_corpus_path = NULL;
    args->validation_sample_limit = 4;
    args->validate_every_n = 0;
    args->kl_weight = 0.05f;
    args->save_state_path = NULL;
    args->load_state_path = NULL;
}

static int validate_cli_args(const cli_args_t *args) {
    if (!args) return -1;

    if (!args->model_name) {
        LOG_ERROR("ERROR: Model name required. Use -m or --model flag.");
        return -1;
    }

    if (args->convert_ternary) {
        if (!args->output_path) {
            LOG_ERROR("ERROR: --output is required with --convert-ternary.");
            return -1;
        }
        if (args->prompt_arg) {
            LOG_ERROR("ERROR: --prompt cannot be combined with --convert-ternary.");
            return -1;
        }
        if (args->save_state_path || args->load_state_path) {
            LOG_ERROR("ERROR: session state flags are not valid in --convert-ternary mode.");
            return -1;
        }
        if (args->calibration_sample_limit <= 0) {
            LOG_ERROR("ERROR: --calibration-samples must be > 0.");
            return -1;
        }
        if (args->kl_weight < 0.0f) {
            LOG_ERROR("ERROR: --kl-weight must be >= 0.");
            return -1;
        }
        if (args->validation_sample_limit < 0) {
            LOG_ERROR("ERROR: --validation-samples must be >= 0.");
            return -1;
        }
        if (args->validate_every_n < 0) {
            LOG_ERROR("ERROR: --validate-every must be >= 0.");
            return -1;
        }
        if (args->layer_name && args->validate_every_n > 0) {
            LOG_ERROR("ERROR: --validate-every is only supported for full-model ternary conversion.");
            return -1;
        }
    }

    return 0;
}

static int run_ternary_conversion_mode(const cli_args_t *args) {
    ternary_conversion_config_t config;

    if (!args) {
        LOG_ERROR("ternary conversion mode: args is NULL");
        return -1;
    }

    config.model_name = args->model_name;
    config.output_path = args->output_path;
    config.layer_name = args->layer_name;
    config.calibration_corpus_path = args->calibration_corpus_path;
    config.validation_corpus_path = args->validation_corpus_path;
    config.context_len = args->context_len;
    config.calibration_sample_limit = args->calibration_sample_limit;
    config.validation_sample_limit = args->validation_sample_limit;
    config.validate_every_n = args->validate_every_n;
    config.kl_weight = args->kl_weight;
    return transformer_run_ternary_conversion(&config);
}

static void print_session_state(const inference_context_t *ctx,
                                const char *last_state_op,
                                const char *last_state_path) {
    const char *backend_name = (ctx->session && ctx->session->backend && ctx->session->backend->name)
        ? ctx->session->backend->name
        : "unknown";
    int persistence_supported = (ctx->session && ctx->session->backend &&
        (ctx->session->backend->type == SAPPHIRE_BACKEND_TYPE_CPU ||
         ctx->session->backend->type == SAPPHIRE_BACKEND_TYPE_VULKAN));
    int kv_seq_len = -1;
    if (ctx->session && ctx->session->kv_cache) {
        kv_seq_len = kv_cache_get_seq_len(ctx->session->kv_cache);
    }

    printf("\nSession State:\n");
    printf("  Backend: %s\n", backend_name);
    printf("  Persistence: %s\n", persistence_supported ? "supported" : "not supported for current backend");
    printf("  Conversation tokens: %d\n", ctx->conversation_len);
    if (kv_seq_len >= 0) {
        printf("  KV sequence length: %d\n", kv_seq_len);
    } else {
        printf("  KV sequence length: unavailable\n");
    }
    if (last_state_op && last_state_path[0] != '\0') {
        printf("  Last state file (%s): %s\n", last_state_op, last_state_path);
    } else {
        printf("  Last state file: none in this session\n");
    }
}

static const char *skip_ws(const char *s) {
    while (*s == ' ' || *s == '\t') s++;
    return s;
}

static int handle_slash_command(inference_context_t *ctx,
                                const char *prompt,
                                char *last_state_path,
                                size_t last_state_path_size,
                                const char **last_state_op,
                                int *should_exit) {
    if (!ctx || !prompt || prompt[0] != '/') return 0;

    if (strcmp(prompt, "/exit") == 0 || strcmp(prompt, "/quit") == 0) {
        printf("Exiting Sapphire inference engine. Goodbye!\n");
        *should_exit = 1;
        return 1;
    }

    if (strcmp(prompt, "/clear") == 0) {
        printf("Conversation history cleared\n");
        if (ctx->session) {
            destroy_inference_session(ctx->session);
            ctx->session = inference_session_create(ctx->spec, ctx->context_len);
        }
        ctx->conversation_len = 0;
        return 1;
    }

    if (strncmp(prompt, "/save", 5) == 0) {
        const char *path = skip_ws(prompt + 5);
        if (*path == '\0') {
            printf("Usage: /save <path>\n");
        } else if (inference_context_save_state(ctx, path) == 0) {
            printf("Session state saved to: %s\n", path);
            snprintf(last_state_path, last_state_path_size, "%s", path);
            *last_state_op = "saved";
        } else {
            printf("Failed to save session state to: %s\n", path);
        }
        return 1;
    }

    if (strncmp(prompt, "/load", 5) == 0) {
        const char *path = skip_ws(prompt + 5);
        if (*path == '\0') {
            printf("Usage: /load <path>\n");
        } else if (inference_context_load_state(ctx, path) == 0) {
            printf("Session state loaded from: %s\n", path);
            snprintf(last_state_path, last_state_path_size, "%s", path);
            *last_state_op = "loaded";
        } else {
            printf("Failed to load session state from: %s\n", path);
        }
        return 1;
    }

    if (strcmp(prompt, "/state") == 0) {
        print_session_state(ctx, *last_state_op, last_state_path);
        return 1;
    }

    if (strcmp(prompt, "/info") == 0) {
        printf("\nModel Information:\n");
        printf("\nInference Settings:\n");
        printf("  Temperature: %.2f\n", ctx->temperature);
        printf("  Max tokens: %d\n", ctx->max_tokens);
        printf("  Context length: %d\n", ctx->context_len);
        return 1;
    }

    if (strcmp(prompt, "/help") == 0) {
        printf("\nAvailable Commands:\n");
        printf("  /exit          - Exit the program\n");
        printf("  /clear         - Clear conversation history\n");
        printf("  /save <path>   - Save session state\n");
        printf("  /load <path>   - Load session state\n");
        printf("  /state         - Show session state visibility\n");
        printf("  /info          - Show model configuration\n");
        printf("  /help          - Show this help message\n");
        printf("\nJust type your prompt to generate responses.\n");
        return 1;
    }

    printf("Unknown command: %s\n", prompt);
    printf("Type '/help' for available commands.\n");
    return 1;
}

static int parse_cli_args(int argc, const char * const argv[], cli_args_t *args) {
    typedef enum {
        CLI_OPT_UNKNOWN = 0,
        CLI_OPT_MODEL,
        CLI_OPT_CONTEXT,
        CLI_OPT_TEMP,
        CLI_OPT_MAX_TOKENS,
        CLI_OPT_PROMPT,
        CLI_OPT_CONVERT_TERNARY,
        CLI_OPT_OUTPUT,
        CLI_OPT_LAYER,
        CLI_OPT_CALIBRATION_CORPUS,
        CLI_OPT_CALIBRATION_SAMPLES,
        CLI_OPT_VALIDATION_CORPUS,
        CLI_OPT_VALIDATION_SAMPLES,
        CLI_OPT_VALIDATE_EVERY,
        CLI_OPT_KL_WEIGHT,
        CLI_OPT_SAVE_STATE,
        CLI_OPT_LOAD_STATE
    } cli_option_t;

    if (!args) return -1;

    for (int i = 1; i < argc; i++) {
        const char *value = NULL;
        cli_option_t option = CLI_OPT_UNKNOWN;

        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) option = CLI_OPT_MODEL;
        else if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--context") == 0) option = CLI_OPT_CONTEXT;
        else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--temp") == 0) option = CLI_OPT_TEMP;
        else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--max-tokens") == 0) option = CLI_OPT_MAX_TOKENS;
        else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) option = CLI_OPT_PROMPT;
        else if (strcmp(argv[i], "--convert-ternary") == 0) option = CLI_OPT_CONVERT_TERNARY;
        else if (strcmp(argv[i], "--output") == 0) option = CLI_OPT_OUTPUT;
        else if (strcmp(argv[i], "--layer") == 0) option = CLI_OPT_LAYER;
        else if (strcmp(argv[i], "--calibration-corpus") == 0) option = CLI_OPT_CALIBRATION_CORPUS;
        else if (strcmp(argv[i], "--calibration-samples") == 0) option = CLI_OPT_CALIBRATION_SAMPLES;
        else if (strcmp(argv[i], "--validation-corpus") == 0) option = CLI_OPT_VALIDATION_CORPUS;
        else if (strcmp(argv[i], "--validation-samples") == 0) option = CLI_OPT_VALIDATION_SAMPLES;
        else if (strcmp(argv[i], "--validate-every") == 0) option = CLI_OPT_VALIDATE_EVERY;
        else if (strcmp(argv[i], "--kl-weight") == 0) option = CLI_OPT_KL_WEIGHT;
        else if (strcmp(argv[i], "--save-state") == 0) option = CLI_OPT_SAVE_STATE;
        else if (strcmp(argv[i], "--load-state") == 0) option = CLI_OPT_LOAD_STATE;

        if (option == CLI_OPT_CONVERT_TERNARY) {
            args->convert_ternary = 1;
            continue;
        }
        if (option == CLI_OPT_UNKNOWN || i + 1 >= argc) {
            continue;
        }

        value = argv[++i];
        switch (option) {
            case CLI_OPT_MODEL: args->model_name = value; break;
            case CLI_OPT_CONTEXT: args->context_len = atoi(value); break;
            case CLI_OPT_TEMP: args->temperature = atof(value); break;
            case CLI_OPT_MAX_TOKENS: args->max_tokens = atoi(value); break;
            case CLI_OPT_PROMPT: args->prompt_arg = value; break;
            case CLI_OPT_OUTPUT: args->output_path = value; break;
            case CLI_OPT_LAYER: args->layer_name = value; break;
            case CLI_OPT_CALIBRATION_CORPUS: args->calibration_corpus_path = value; break;
            case CLI_OPT_CALIBRATION_SAMPLES: args->calibration_sample_limit = atoi(value); break;
            case CLI_OPT_VALIDATION_CORPUS: args->validation_corpus_path = value; break;
            case CLI_OPT_VALIDATION_SAMPLES: args->validation_sample_limit = atoi(value); break;
            case CLI_OPT_VALIDATE_EVERY: args->validate_every_n = atoi(value); break;
            case CLI_OPT_KL_WEIGHT: args->kl_weight = (float)atof(value); break;
            case CLI_OPT_SAVE_STATE: args->save_state_path = value; break;
            case CLI_OPT_LOAD_STATE: args->load_state_path = value; break;
            default: break;
        }
    }

    return 0;
}

/**
 * @brief Run the interactive REPL loop for prompts and commands.
 *
 * Reads lines from stdin, handles slash-commands ("/exit", "/clear", "/info",
 * and "/help"), calls `perform_inference()` for non-command prompts, and may
 * recreate `ctx->session` on `/clear`.
 *
 * @param ctx Non-NULL inference context used for session state and inference.
 * @return 0 on normal exit; -1 if `ctx` is NULL or an immediate error occurs.
 */
static int interactive_loop(inference_context_t* ctx) {
    if (!ctx) return -1;

    char prompt[MAX_PROMPT_LENGTH];
    char output[BUFFER_SIZE];
    char last_state_path[BUFFER_SIZE];
    const char* last_state_op = NULL;
    last_state_path[0] = '\0';

    printf("\n");
    printf("================================================================================\n");
    printf("                         SAPPHIRE INFERENCE ENGINE                             \n");
    printf("================================================================================\n");
    printf("\nModel loaded and ready for inference.\n");
    printf("Type '/help' for commands, '/exit' to quit.\n");
    printf("Type 'quit' or 'exit' to end the program.\n");
    printf("\n");

    while (1) {
        // Print prompt
        printf("\n[Sapphire] > ");
        fflush(stdout);

        // Read user input
        if (!fgets(prompt, sizeof(prompt), stdin)) {
            break;
        }

        // Remove trailing newline
        size_t len = strlen(prompt);
        if (len > 0 && prompt[len - 1] == '\n') {
            prompt[len - 1] = '\0';
        }

        // Skip empty lines
        if (strlen(prompt) == 0) {
            continue;
        }

        if (prompt[0] == '/') {
            int should_exit = 0;
            (void)handle_slash_command(ctx,
                                       prompt,
                                       last_state_path,
                                       sizeof(last_state_path),
                                       &last_state_op,
                                       &should_exit);
            if (should_exit) break;
        } else {
            // Perform inference
            printf("\n[Generating response...]\n");

            int result = perform_inference(ctx, prompt, output, sizeof(output));

            if (result == 0) {
                double elapsed = ctx->last_inference_time;
                const char *backend_name = (ctx->session && ctx->session->backend) ? ctx->session->backend->name : "unknown";
                printf("\n[Response]\n%s\n", output);
                printf("\n[Generation time: %.3f seconds] (backend=%s)\n", elapsed, backend_name);
            } else {
                printf("Inference failed\n");
            }
        }
    }

    return 0;
}

/**
 * @brief Run a single non-interactive inference and print the result.
 *
 * Convenience wrapper that runs `perform_inference()` for `prompt`, prints the
 * response to the log (via `LOG_INFO`) and returns the inference result code.
 *
 * Ownership semantics: this function DOES NOT free or destroy `ctx`; the
 * caller retains ownership and is responsible for cleaning up the context
 * (e.g., by calling `destroy_inference_context(ctx)`).
 *
 * @param ctx         Non-NULL inference context to use. Caller retains ownership.
 * @param prompt      Null-terminated prompt string to generate from.
 * @param output_size Size in bytes of the output buffer to be used by
 *                    `perform_inference()`; must be >= 1. If greater than
 *                    `BUFFER_SIZE`, a heap buffer will be allocated.
 * @return 0 on success, non-zero on failure (invalid args, allocation failure,
 *         or inference error).
 */
int one_shot_inference(inference_context_t* ctx, const char* prompt, int output_size) {
    if (!ctx || !prompt || output_size <= 0) return -1;

    char stack_buf[BUFFER_SIZE];
    char *heap_buf = NULL;
    char *output = NULL;
    int use_heap = 0;

    if (output_size <= BUFFER_SIZE) {
        output = stack_buf;
    } else {
        heap_buf = (char *)malloc((size_t)output_size);
        if (!heap_buf) {
            LOG_ERROR("One-shot inference: failed to allocate output buffer of size %d", output_size);
            return -1;
        }
        use_heap = 1;
        output = heap_buf;
    }

    /* Defensive: ensure last byte is NUL so logging is safe even if
     * perform_inference() does not NUL-terminate on errors. */
    output[output_size - 1] = '\0';

    LOG_INFO("Running prompt (non-interactive): '%s'", prompt);

    if (ctx->session) {
        inference_session_reset(ctx->session);
    }
    ctx->conversation_len = 0;

    int rc = perform_inference(ctx, prompt, output, output_size);
    if (rc == 0) {
        LOG_INFO("\n[Response]\n%s\n", output);
        LOG_INFO("[Inference time: %.3f seconds] (backend=%s)", ctx->last_inference_time,
                 (ctx->session && ctx->session->backend) ? ctx->session->backend->name : "unknown");
    } else {
        LOG_ERROR("One-shot inference failed");
    }

    if (use_heap) free(heap_buf);
    return rc;
}

/**
 * @brief Entry point for the Sapphire inference engine.
 *
 * Parses command-line arguments, creates an inference context, then runs
 * either a single non-interactive prompt (via -p) or the interactive REPL.
 * Cleans up resources and returns an exit status.
 *
 * @return 0 on success; non-zero on failure (missing arguments, context creation
 *         failure, or runtime errors).
 */
int main(int argc, char* argv[]) {
    // Check for help first
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_help(argc > 0 ? argv[0] : "sapphire");
            return 0;
        }
    }

    cli_args_t args;
    cli_args_init(&args);
    if (parse_cli_args(argc, (const char * const *)argv, &args) != 0) return 1;

    if (validate_cli_args(&args) != 0) {
        print_help(argv[0]);
        return 1;
    }

    log_set_level_from_env("SAPPHIRE_LOG_LEVEL");

    printf("================================================================================\n");
    printf("                      SAPPHIRE INFERENCE ENGINE (v1.0)\n");
    printf("================================================================================\n");

    if (args.convert_ternary) {
        int rc = run_ternary_conversion_mode(&args);
        printf("\n================================================================================\n");
        printf("                    Sapphire Inference Engine Closed\n");
        printf("================================================================================\n");
        return rc;
    }

    // Create inference context with tokenizer
    inference_context_t* ctx = create_inference_context(args.temperature, args.max_tokens, args.context_len, args.model_name);
    if (!ctx) {
        LOG_ERROR("Failed to create inference context. Exiting.");
        return 1;
    }

    if (args.load_state_path) {
        if (inference_context_load_state(ctx, args.load_state_path) != 0) {
            LOG_WARN("Failed to load session state from %s", args.load_state_path);
        } else {
            LOG_INFO("Loaded session state from %s", args.load_state_path);
        }
    }

    int result = 0;
    // If prompt_arg provided, run a single non-interactive inference and exit
    if (args.prompt_arg) {
        result = one_shot_inference(ctx, args.prompt_arg, BUFFER_SIZE);
    } else {
        // Enter interactive loop
        result = interactive_loop(ctx);
    }

    if (args.save_state_path) {
        if (inference_context_save_state(ctx, args.save_state_path) != 0) {
            LOG_WARN("Failed to save session state to %s", args.save_state_path);
        } else {
            LOG_INFO("Saved session state to %s", args.save_state_path);
        }
    }

    // Cleanup
    destroy_inference_context(ctx);

    printf("\n================================================================================\n");
    printf("                    Sapphire Inference Engine Closed\n");
    printf("================================================================================\n");

    return result;
}
