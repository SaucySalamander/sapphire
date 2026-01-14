#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tokenizer.h"
#include "utils.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path>\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    printf("DEBUG: Loading tokenizer from %s\n", model_path);
    sapphire_tokenizer_t *tokenizer = tokenizer_load(model_path);
    if (!tokenizer) {
        fprintf(stderr, "Failed to load tokenizer\n");
        return 1;
    }

    const char *prompt = "<start_of_turn>user\nThe capital of France is\n<end_of_turn>\n<start_of_turn>model\n";
    printf("DEBUG: Tokenizing prompt:\n%s\n", prompt);

    int tokens[1024];
    int count = tokenize(tokenizer, prompt, tokens, 1024);

    printf("DEBUG: Token count: %d\n", count);
    printf("Tokens: [");
    for (int i = 0; i < count; i++) {
        printf("%d", tokens[i]);
        if (i < count - 1) printf(", ");
    }
    printf("]\n");
    
    // Check specific token at index 4 (0-based)
    // Expected: 2, 105, 2364, 107, 818 (The)
    if (count > 4) {
        if (tokens[4] == 818) {
             printf("SUCCESS: Token[4] is 818 ('The')\n");
        } else if (tokens[4] == 669) {
             printf("FAILURE: Token[4] is 669 (' The') - Leading space issue persists.\n");
        } else {
             printf("FAILURE: Token[4] is %d (Unknown)\n", tokens[4]);
        }
    }

    return 0;
}
