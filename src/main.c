#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <time.h>
#include "ggml_model.h"
#include "inference.h"
#include "rope.h"
#include "positional_encoding.h"
#include "attention.h"
#include "attention_strategy.h"
#include "utils.h"
#include "safetensors_reader.h"
#include "tensor_mapper.h"
#include "tensor.h"

#define MAX_PROMPT_LENGTH 1024
#define MAX_TOKENS_GENERATE 100
#define BUFFER_SIZE 4096

typedef enum {
    MODEL_FORMAT_UNKNOWN = 0,
    MODEL_FORMAT_GGML = 1,
    MODEL_FORMAT_SAFETENSORS = 2
} model_format_t;

typedef struct {
    llm_model_t *model;
    inference_session_t *session;
    int max_tokens;
    float temperature;
    int context_len;
    model_format_t format;
    void *format_specific;  // For Safetensors: safetensors_file_t*
} inference_context_t;


/**
 * @brief Detect model file format by file extension and header.
 *
 * @param path Path to model file
 * @return Detected format (GGML, SAFETENSORS, or UNKNOWN)
 */
static model_format_t detect_model_format(const char *path) {
    if (!path) return MODEL_FORMAT_UNKNOWN;
    
    // Check file extension
    const char *ext = strrchr(path, '.');
    if (!ext) return MODEL_FORMAT_UNKNOWN;
    
    ext++;  // Skip the dot
    
    // Safetensors format
    if (strcmp(ext, "safetensors") == 0) {
        return MODEL_FORMAT_SAFETENSORS;
    }
    
    // GGML format (gguf, ggml)
    if (strcmp(ext, "gguf") == 0 || strcmp(ext, "ggml") == 0 || strcmp(ext, "bin") == 0) {
        return MODEL_FORMAT_GGML;
    }
    
    // Try to detect by file header
    FILE *f = fopen(path, "rb");
    if (!f) return MODEL_FORMAT_UNKNOWN;
    
    uint32_t magic;
    if (fread(&magic, sizeof(uint32_t), 1, f) == 1) {
        fclose(f);
        
        // GGML/GGUF magic number is 0x67676d6c ("ggml")
        if (magic == 0x67676d6c) {
            return MODEL_FORMAT_GGML;
        }
    }
    
    fclose(f);
    return MODEL_FORMAT_UNKNOWN;
}

/**
 * @brief Load a model from Safetensors file
 * 
 * Uses sapphire_load_safetensors() to:
 * 1. Extract model config from tensor metadata
 * 2. Map HuggingFace tensor names to Sapphire structure
 * 3. Load weights into llm_model_t structure (zero-copy, mmapped)
 * 4. Support BF16 and F16 formats
 */
static llm_model_t* load_model_safetensors(const char *model_path, model_config_t *config) {
    if (!model_path || !config) return NULL;
    
    char error_msg[512] = {0};
    llm_model_t *model = malloc(sizeof(llm_model_t));
    if (!model) {
        fprintf(stderr, "ERROR: Failed to allocate model structure\n");
        return NULL;
    }
    memset(model, 0, sizeof(llm_model_t));
    
    // Use tensor_mapper to load Safetensors file
    int rc = sapphire_load_safetensors(model_path, model, error_msg, sizeof(error_msg));
    if (rc != SAPPHIRE_OK) {
        fprintf(stderr, "ERROR: Failed to load Safetensors model: %s\n", error_msg);
        free(model);
        return NULL;
    }
    
    // Copy loaded model config to output config parameter
    *config = model->config;
    
    printf("✓ Successfully loaded Safetensors model\n");
    printf("  - Layers: %d\n", model->config.num_layers);
    printf("  - Hidden size: %d\n", model->config.d_model);
    printf("  - Vocab size: %d\n", model->config.vocab_size);
    printf("  - Num heads: %d\n", model->config.num_heads);
    printf("  - Head dim: %d\n", model->config.d_k);
    
    return model;
}

/**
 * @brief Load a model from GGML/GGUF file using new tensor_mapper API
 */
static llm_model_t* load_model_ggml(const char *model_path, model_config_t *config) {
    if (!model_path || !config) return NULL;

    char error_msg[512] = {0};
    llm_model_t *model = malloc(sizeof(llm_model_t));
    if (!model) {
        fprintf(stderr, "ERROR: Failed to allocate model structure\n");
        return NULL;
    }
    memset(model, 0, sizeof(llm_model_t));

    // Use new GGML loader via tensor_mapper
    int rc = sapphire_load_ggml(model_path, model, error_msg, sizeof(error_msg));
    if (rc != SAPPHIRE_OK) {
        fprintf(stderr, "ERROR: Failed to load GGML model: %s\n", error_msg);
        free(model);
        return NULL;
    }

    // Copy loaded model config to output config parameter
    *config = model->config;

    printf("✓ Successfully loaded GGML model\n");
    printf("  - Layers: %d\n", model->config.num_layers);
    printf("  - Hidden size: %d\n", model->config.d_model);
    printf("  - Vocab size: %d\n", model->config.vocab_size);
    printf("  - Num heads: %d\n", model->config.num_heads);
    printf("  - Head dim: %d\n", model->config.d_k);

    return model;
}

/**
 * @brief Load a model from file (auto-detect format)
 */
static llm_model_t* load_model(const char *model_path) {
    if (!model_path) {
        fprintf(stderr, "ERROR: Model path is NULL\n");
        return NULL;
    }
    
    printf("Loading model from: %s\n", model_path);
    
    // Check if file exists
    if (access(model_path, F_OK) == -1) {
        fprintf(stderr, "ERROR: Model file not found: %s\n", model_path);
        return NULL;
    }
    
    // Detect file format
    model_format_t format = detect_model_format(model_path);
    const char *format_name = "UNKNOWN";
    
    switch (format) {
        case MODEL_FORMAT_GGML:
            format_name = "GGML/GGUF";
            break;
        case MODEL_FORMAT_SAFETENSORS:
            format_name = "Safetensors";
            break;
        default:
            format_name = "UNKNOWN";
            break;
    }
    
    printf("✓ Detected format: %s\n", format_name);
    
    // Create default config for Gemma 3
    // In a real implementation, this would be read from file or config
    model_config_t config = {
        .vocab_size = 256000,      // Gemma 3 vocabulary
        .d_model = 2048,           // Model dimension
        .num_heads = 16,           // Number of attention heads
        .d_k = 128,                // Dimension per head (2048 / 16)
        .num_layers = 18,          // Number of transformer layers
        .max_context_len = 8192,   // Context length
        .rope_base = 500000.0f     // RoPE base for Gemma 3
    };
    
    // Load model using the appropriate loader
    llm_model_t *model = NULL;
    
    if (format == MODEL_FORMAT_GGML) {
        printf("Loading GGML format model...\n");
        model = load_model_ggml(model_path, &config);
    } else if (format == MODEL_FORMAT_SAFETENSORS) {
        printf("Loading Safetensors format model...\n");
        model = load_model_safetensors(model_path, &config);
    } else {
        fprintf(stderr, "ERROR: Unknown or unsupported model format\n");
        return NULL;
    }
    
    if (!model) {
        fprintf(stderr, "ERROR: Failed to load model from %s\n", model_path);
        return NULL;
    }
    
    printf("✓ Model loaded successfully\n");
    printf("  - Vocabulary size: %d\n", model->config.vocab_size);
    printf("  - Model dimension: %d\n", model->config.d_model);
    printf("  - Attention heads: %d\n", model->config.num_heads);
    printf("  - Dimension per head: %d\n", model->config.d_k);
    printf("  - Number of layers: %d\n", model->config.num_layers);
    printf("  - Max context: %d\n", model->config.max_context_len);
    printf("  - RoPE base: %.1f\n", model->config.rope_base);
    
    return model;
}

/**
 * @brief Initialize inference context
 */
static inference_context_t* create_inference_context(llm_model_t *model, int context_len) {
    if (!model) return NULL;
    
    inference_context_t *ctx = (inference_context_t *)malloc(sizeof(inference_context_t));
    if (!ctx) {
        fprintf(stderr, "ERROR: Failed to allocate inference context\n");
        return NULL;
    }
    
    ctx->model = model;
    ctx->context_len = context_len;
    ctx->max_tokens = MAX_TOKENS_GENERATE;
    ctx->temperature = 1.0f;
    ctx->format = MODEL_FORMAT_GGML;  // Default format (track actual format if needed)
    ctx->format_specific = NULL;
    
    // Create inference session
    ctx->session = inference_session_create(model, context_len);
    if (!ctx->session) {
        fprintf(stderr, "ERROR: Failed to create inference session\n");
        free(ctx);
        return NULL;
    }
    
    printf("✓ Inference session created (context length: %d)\n", context_len);
    return ctx;
}

/**
 * @brief Destroy inference context
 */
static void destroy_inference_context(inference_context_t *ctx) {
    if (!ctx) return;
    
    if (ctx->session) {
        inference_session_destroy(ctx->session);
    }
    
    // Clean up format-specific resources
    if (ctx->format == MODEL_FORMAT_SAFETENSORS && ctx->format_specific) {
        safetensors_file_t *st = (safetensors_file_t*)ctx->format_specific;
        safetensors_close(st);
    }
    
    free(ctx);
}

/**
 * @brief Print a single token ID as a character or placeholder.
 * This is the streaming version that prints immediately instead of buffering.
 */
static void print_token(int token_id) {
    // Skip BOS token (2)
    if (token_id == 2) {
        return;
    }
    
    // EOS token (107)
    if (token_id == 107) {
        return;
    }
    
    // Handle our custom ASCII range [256, 382] (printable ASCII 32-126)
    if (token_id >= 256 && token_id <= 382) {
        unsigned char c = (unsigned char)(32 + (token_id - 256));
        printf("%c", c);
        fflush(stdout);  // Streaming effect
        return;
    }
    
    // If the model generates a token outside our ASCII range, show it as a debug token
    // This helps diagnose whether the model is actually generating in the right range
    printf("[T%d]", token_id);
    fflush(stdout);
}

/**
 * @brief Decode a single token ID back to a character or string.
 * Inverse of the tokenize_prompt function (buffered version).
 */
static char* decode_token(int token_id) {
    static char buf[20];
    memset(buf, 0, sizeof(buf));
    
    // Skip BOS token (2)
    if (token_id == 2) {
        return "";
    }
    
    // EOS token (107)
    if (token_id == 107) {
        return "";
    }
    
    // Handle our custom ASCII range [256, 382] (printable ASCII 32-126)
    if (token_id >= 256 && token_id <= 382) {
        unsigned char c = (unsigned char)(32 + (token_id - 256));
        buf[0] = c;
        buf[1] = '\0';
        return buf;
    }
    
    // For tokens outside our range, return a placeholder
    snprintf(buf, sizeof(buf), "[T%d]", token_id);
    return buf;
}

/**
 * @brief Simple character-level tokenizer (MVP for Gemma 3)
 * Maps characters to token IDs. For production, use sentencepiece.
 */
static int tokenize_prompt(const char *prompt, int *token_ids, int max_tokens) {
    if (!prompt || !token_ids) return -1;
    
    int len = strlen(prompt);
    int token_count = 0;
    
    // Add BOS token (Gemma uses 2)
    if (token_count < max_tokens) {
        token_ids[token_count++] = 2;
    }
    
    // Character-level tokenization (simplified - real implementation uses BPE)
    // Map printable ASCII (32-126) to tokens [256, 382]
    for (int i = 0; i < len && token_count < max_tokens; i++) {
        unsigned char c = (unsigned char)prompt[i];
        if (c >= 32 && c <= 126) {
            // Map printable ASCII to token range [256, 382]
            token_ids[token_count++] = 256 + (c - 32);
        }
    }
    
    return token_count;
}

/**
 * @brief Greedy sampler: select token with highest logit
 */
static int greedy_sample(float *logits, int vocab_size) {
    if (!logits || vocab_size <= 0) return -1;
    
    int best_token = 0;
    float best_logit = logits[0];
    
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best_logit) {
            best_logit = logits[i];
            best_token = i;
        }
    }
    
    return best_token;
}

/**
 * @brief Temperature sampling with softmax
 */
static int temperature_sample(float *logits, int vocab_size, float temperature) {
    if (!logits || vocab_size <= 0 || temperature <= 0) return greedy_sample(logits, vocab_size);
    
    // Find max logit for numerical stability
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    // Compute softmax with temperature
    float *probs = malloc(vocab_size * sizeof(float));
    float sum = 0.0f;
    
    for (int i = 0; i < vocab_size; i++) {
        float exp_val = expf((logits[i] - max_logit) / temperature);
        probs[i] = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    for (int i = 0; i < vocab_size; i++) {
        probs[i] /= sum;
    }
    
    // Sample using cumulative distribution
    float rand_val = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    int sampled_token = vocab_size - 1;
    
    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if (rand_val <= cumsum) {
            sampled_token = i;
            break;
        }
    }
    
    free(probs);
    return sampled_token;
}

/**
 * @brief Simple detokenizer using the decode_token function
 */
static char* detokenize_tokens(const int *token_ids, int count) {
    if (!token_ids || count <= 0) {
        char *empty = malloc(1);
        empty[0] = '\0';
        return empty;
    }
    
    // Allocate enough space for all tokens (max 4 bytes per token + null terminator)
    char *output = malloc(count * 4 + 1);
    int out_idx = 0;
    
    for (int i = 0; i < count; i++) {
        const char *decoded = decode_token(token_ids[i]);
        if (decoded && *decoded != '\0') {
            int len = strlen(decoded);
            memcpy(output + out_idx, decoded, len);
            out_idx += len;
        }
    }
    
    output[out_idx] = '\0';
    return output;
}

/**
 * @brief Perform full inference with greedy/temperature sampling
 * 
 * Complete generation loop:
 * 1. Tokenize prompt
 * 2. For each token position up to max_tokens:
 *    - Forward pass through model layers
 *    - Get logits from LM head
 *    - Sample next token
 *    - Add to sequence
 *    - Check for EOS
 * 3. Detokenize output
 */
static int perform_inference(inference_context_t *ctx, const char *prompt, char *output, int output_size) {
    if (!ctx || !prompt || !output) return -1;
    
    llm_model_t *model = ctx->model;
    inference_session_t *session = ctx->session;
    
    printf("Generating response...\n");
    fflush(stdout);
    
    // 1. Tokenize prompt
    int *tokens = malloc((ctx->max_tokens + 1) * sizeof(int));
    int prompt_len = tokenize_prompt(prompt, tokens, ctx->max_tokens);
    if (prompt_len <= 0) {
        fprintf(stderr, "ERROR: Failed to tokenize prompt\n");
        free(tokens);
        return -1;
    }
    
    // Reset session for new sequence
    inference_session_reset(session);
    
    // 2. Process prompt tokens through model
    float *logits = malloc(model->config.vocab_size * sizeof(float));
    int token_count = prompt_len;
    int total_tokens = prompt_len;
    
    // Debug: show first few tokens
    printf("DEBUG: Tokenized prompt to %d tokens\n", prompt_len);
    
    // 3. Generation loop
    clock_t start_time = clock();
    int generated_count = 0;
    
    while (total_tokens < ctx->max_tokens) {
        // Get last token
        int current_token = tokens[total_tokens - 1];
        
        // Clear logits buffer before inference
        memset(logits, 0, model->config.vocab_size * sizeof(float));
        
        // Forward pass
        inference_forward(session, current_token, total_tokens - 1, logits);
        
        // Sample next token using temperature
        int next_token = temperature_sample(logits, model->config.vocab_size, 1.0f);
        
        if (next_token < 0 || next_token >= model->config.vocab_size) {
            fprintf(stderr, "ERROR: Invalid token sampled: %d (vocab_size=%d)\n", next_token, model->config.vocab_size);
            break;
        }
        
        // Debug: print sampled token
        if (generated_count < 5) {  // Only print first 5 tokens for debug
            fprintf(stderr, "DEBUG: Generated token %d ", next_token);
            if (next_token >= 256 && next_token <= 382) {
                fprintf(stderr, "(ASCII: '%c')\n", (unsigned char)(32 + (next_token - 256)));
            } else {
                fprintf(stderr, "(out of ASCII range)\n");
            }
        }
        
        // Check for EOS (token 107 for Gemma)
        if (next_token == 107) {
            break;
        }
        
        // Append to sequence
        if (total_tokens < ctx->max_tokens) {
            tokens[total_tokens++] = next_token;
            generated_count++;
        } else {
            break;
        }
    }
    
    clock_t end_time = clock();
    double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    // 4. Detokenize and return
    char *response = detokenize_tokens(tokens + prompt_len, total_tokens - prompt_len);
    snprintf(output, output_size, "%s", response);
    
    printf("DEBUG: Generated %d tokens\n", generated_count);
    printf("[Generation time: %.3f seconds]\n", elapsed);
    
    free(logits);
    free(tokens);
    free(response);
    
    return 0;
}

/**
 * @brief Print help message
 */
static void print_help(const char *program_name) {
    printf("\n");
    printf("================================================================================\n");
    printf("                        Sapphire Inference Engine\n");
    printf("                   Hybrid Daemon LLM Architecture (Phase 6)\n");
    printf("================================================================================\n");
    printf("\nUsage: %s <model_path> [options]\n\n", program_name);
    printf("Arguments:\n");
    printf("  <model_path>              Path to GGML format model file (required)\n");
    printf("  -c, --context <length>    Context length (default: 2048)\n");
    printf("  -t, --temp <value>        Temperature for sampling (default: 1.0)\n");
    printf("  -n, --max-tokens <num>    Maximum tokens to generate (default: 100)\n");
    printf("  -h, --help                Show this help message\n");
    printf("\nInteractive Commands:\n");
    printf("  /exit                     Exit the program\n");
    printf("  /clear                    Clear conversation history\n");
    printf("  /info                     Show model information\n");
    printf("  /help                     Show command help\n");
    printf("\nExample:\n");
    printf("  %s ./models/gemma-3-2b.gguf -c 4096 -t 0.7 -n 200\n", program_name);
    printf("\n");
}

/**
 * @brief Interactive prompt loop
 */
static int interactive_loop(inference_context_t *ctx) {
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
                printf("✓ Conversation history cleared\n");
                // Reset session
                if (ctx->session) {
                    inference_session_destroy(ctx->session);
                    ctx->session = inference_session_create(ctx->model, ctx->context_len);
                }
            } else if (strcmp(prompt, "/info") == 0) {
                printf("\nModel Information:\n");
                printf("  Vocabulary: %d tokens\n", ctx->model->config.vocab_size);
                printf("  Dimension: %d\n", ctx->model->config.d_model);
                printf("  Heads: %d\n", ctx->model->config.num_heads);
                printf("  Head dimension: %d\n", ctx->model->config.d_k);
                printf("  Layers: %d\n", ctx->model->config.num_layers);
                printf("  Max context: %d\n", ctx->model->config.max_context_len);
                printf("  RoPE base: %.1f\n", ctx->model->config.rope_base);
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
                fprintf(stderr, "ERROR: Inference failed\n");
            }
        }
    }
    
    return 0;
}

/**
 * @brief Main function - Interactive Sapphire Inference Engine
 */
int main(int argc, char *argv[]) {
    // Check for help first
    if (argc < 2 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        print_help(argc > 0 ? argv[0] : "sapphire");
        return argc < 2 ? 1 : 0;
    }
    
    // Parse arguments
    const char *model_path = argv[1];
    int context_len = 2048;
    float temperature = 1.0f;
    int max_tokens = MAX_TOKENS_GENERATE;
    
    for (int i = 2; i < argc; i++) {
        if ((strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--context") == 0) && i + 1 < argc) {
            context_len = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--temp") == 0) && i + 1 < argc) {
            temperature = atof(argv[++i]);
        } else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--max-tokens") == 0) && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_help(argv[0]);
            return 0;
        }
    }
    
    printf("================================================================================\n");
    printf("                      SAPPHIRE INFERENCE ENGINE (v1.0)\n");
    printf("            Hybrid Daemon LLM Framework with Streaming Inference\n");
    printf("================================================================================\n");
    
    // Load model
    llm_model_t *model = load_model(model_path);
    if (!model) {
        fprintf(stderr, "\nFailed to load model. Exiting.\n");
        return 1;
    }
    
    // Create inference context
    inference_context_t *ctx = create_inference_context(model, context_len);
    if (!ctx) {
        fprintf(stderr, "\nFailed to create inference context. Exiting.\n");
        llm_model_destroy(model);
        return 1;
    }
    
    // Update settings
    ctx->temperature = temperature;
    ctx->max_tokens = max_tokens;
    
    printf("\nInference Settings:\n");
    printf("  Temperature: %.2f\n", ctx->temperature);
    printf("  Max tokens: %d\n", ctx->max_tokens);
    printf("  Context length: %d\n", ctx->context_len);
    
    // Enter interactive loop
    int result = interactive_loop(ctx);
    
    // Cleanup
    destroy_inference_context(ctx);
    llm_model_destroy(model);
    
    printf("\n================================================================================\n");
    printf("                    Sapphire Inference Engine Closed\n");
    printf("================================================================================\n");
    
    return result;
}
