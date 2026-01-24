#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "ggml_model.h"
#include "inference.h"
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
    printf("  -h, --help                Show this help message\n");
    printf("\nModel Directory Structure (required files):\n");
    printf("  model.safetensors (or model.gguf / model.bin)\n");
    printf("  tokenizer.json\n");
    printf("  tokenizer_config.json\n");
    printf("  (optional) special_tokens_map.json\n");
    printf("\nInteractive Commands:\n");
    printf("  /exit                     Exit the program\n");
    printf("  /clear                    Clear conversation history\n");
    printf("  /info                     Show model information\n");
    printf("  /help                     Show command help\n");
    printf("\nExample:\n");
    printf("  %s -m gemma3-270m-it -c 4096 -t 0.7 -n 200\n", program_name);
    printf("\n");
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

        // Handle commands
        if (prompt[0] == '/') {
            if (strcmp(prompt, "/exit") == 0 || strcmp(prompt, "/quit") == 0) {
                printf("Exiting Sapphire inference engine. Goodbye!\n");
                break;
            } else if (strcmp(prompt, "/clear") == 0) {
                printf("Conversation history cleared\n");
                // Reset session
                if (ctx->session) {
                    destroy_inference_session(ctx->session);
                    ctx->session = inference_session_create(ctx->spec, ctx->context_len);
                }
            } else if (strcmp(prompt, "/info") == 0) {
                printf("\nModel Information:\n");
                printf("\nInference Settings:\n");
                printf("  Temperature: %.2f\n", ctx->temperature);
                printf("  Max tokens: %d\n", ctx->max_tokens);
                printf("  Context length: %d\n", ctx->context_len);
            } else if (strcmp(prompt, "/help") == 0) {
                printf("\nAvailable Commands:\n");
                printf("  /exit          - Exit the program\n");
                printf("  /clear         - Clear conversation history\n");
                printf("  /info          - Show model configuration\n");
                printf("  /help          - Show this help message\n");
                printf("\nJust type your prompt to generate responses.\n");
            } else {
                printf("Unknown command: %s\n", prompt);
                printf("Type '/help' for available commands.\n");
            }
        } else {
            // Perform inference
            printf("\n[Generating response...]\n");

            clock_t start = clock();
            int result = perform_inference(ctx, prompt, output, sizeof(output));
            clock_t end = clock();

            if (result == 0) {
                double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
                printf("\n[Response]\n%s\n", output);
                printf("\n[Generation time: %.3f seconds]\n", elapsed);
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

    LOG_INFO("\nRunning prompt (non-interactive): '%s'\n", prompt);
    LOG_INFO("\n[Running one-shot inference for prompt: '%s']\n", prompt);

    int rc = perform_inference(ctx, prompt, output, output_size);
    if (rc == 0) {
        LOG_INFO("\n[Response]\n%s\n", output);
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

    // Parse arguments
    const char* model_name = NULL;
    int context_len = CONTEXT_LEN;
    float temperature = TEMPERATURE;
    int max_tokens = MAX_TOKENS_GENERATE;
    const char* prompt_arg = NULL;  // Non-interactive prompt (via -p)

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            model_name = argv[++i];
        } else if ((strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--context") == 0) && i + 1 < argc) {
            context_len = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--temp") == 0) && i + 1 < argc) {
            temperature = atof(argv[++i]);
        } else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--max-tokens") == 0) && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) && i + 1 < argc) {
            prompt_arg = argv[++i];
        }
    }

    // Validate that model name was provided
    if (!model_name) {
        LOG_ERROR("ERROR: Model name required. Use -m or --model flag.\n");
        print_help(argv[0]);
        return 1;
    }

    log_set_level_from_env("SAPPHIRE_LOG_LEVEL");

    printf("================================================================================\n");
    printf("                      SAPPHIRE INFERENCE ENGINE (v1.0)\n");
    printf("================================================================================\n");

    // Create inference context with tokenizer
    inference_context_t* ctx = create_inference_context(temperature, max_tokens, context_len, model_name);
    if (!ctx) {
        LOG_ERROR("Failed to create inference context. Exiting.");
        return 1;
    }

    int result = 0;
    // If prompt_arg provided, run a single non-interactive inference and exit
    if (prompt_arg) {
        result = one_shot_inference(ctx, prompt_arg, BUFFER_SIZE);
    } else {
        // Enter interactive loop
        result = interactive_loop(ctx);
    }

    // Cleanup
    destroy_inference_context(ctx);

    printf("\n================================================================================\n");
    printf("                    Sapphire Inference Engine Closed\n");
    printf("================================================================================\n");

    return result;
}
