/*
 * @file gemma3_27b_map.c
 * @brief Tensor name mapping table for Gemma 3 27B IT.
 *
 * Maps HuggingFace safetensors tensor names (language_model.* prefix) to
 * Sapphire internal structure keys. 62 layers × 13 tensors + 3 = 809 entries.
 */

#include "llm_model.h"
#include "model_spec.h"
#include <stddef.h>

/*
 * Expand all 13 entries for layer N. Uses preprocessor stringification so
 * adjacent string literals are concatenated by the compiler:
 *   "language_model.model.layers." #N ".q_proj.weight"
 * with N=10 → "language_model.model.layers." "10" ".q_proj.weight"
 *           → "language_model.model.layers.10.q_proj.weight"
 */
#define G27B_LAYER(N)                                                                              \
    {"language_model.model.layers." #N ".input_layernorm.weight",                                 \
     "blk." #N, "norm_attn_weight"},                                                              \
    {"language_model.model.layers." #N ".self_attn.q_proj.weight",                               \
     "blk." #N, "q_proj_weight"},                                                                 \
    {"language_model.model.layers." #N ".self_attn.q_norm.weight",                               \
     "blk." #N, "q_norm_weight"},                                                                 \
    {"language_model.model.layers." #N ".self_attn.k_proj.weight",                               \
     "blk." #N, "k_proj_weight"},                                                                 \
    {"language_model.model.layers." #N ".self_attn.k_norm.weight",                               \
     "blk." #N, "k_norm_weight"},                                                                 \
    {"language_model.model.layers." #N ".self_attn.v_proj.weight",                               \
     "blk." #N, "v_proj_weight"},                                                                 \
    {"language_model.model.layers." #N ".self_attn.o_proj.weight",                               \
     "blk." #N, "out_proj_weight"},                                                               \
    {"language_model.model.layers." #N ".post_attention_layernorm.weight",                        \
     "blk." #N, "norm_attn_post_weight"},                                                         \
    {"language_model.model.layers." #N ".pre_feedforward_layernorm.weight",                       \
     "blk." #N, "norm_ffn_weight"},                                                               \
    {"language_model.model.layers." #N ".post_feedforward_layernorm.weight",                      \
     "blk." #N, "norm_ffn_post_weight"},                                                          \
    {"language_model.model.layers." #N ".mlp.gate_proj.weight",                                   \
     "blk." #N, "gate_proj_weight"},                                                              \
    {"language_model.model.layers." #N ".mlp.up_proj.weight",                                     \
     "blk." #N, "up_proj_weight"},                                                                \
    {"language_model.model.layers." #N ".mlp.down_proj.weight",                                   \
     "blk." #N, "down_proj_weight"}

const tensor_map_entry_t GEMMA3_27B_TENSOR_MAP[] = {
    /* Embedding */
    {"language_model.model.embed_tokens.weight", "embedding", "embedding_weight"},

    /* Layers 0-9 */
    G27B_LAYER(0),  G27B_LAYER(1),  G27B_LAYER(2),  G27B_LAYER(3),  G27B_LAYER(4),
    G27B_LAYER(5),  G27B_LAYER(6),  G27B_LAYER(7),  G27B_LAYER(8),  G27B_LAYER(9),

    /* Layers 10-19 */
    G27B_LAYER(10), G27B_LAYER(11), G27B_LAYER(12), G27B_LAYER(13), G27B_LAYER(14),
    G27B_LAYER(15), G27B_LAYER(16), G27B_LAYER(17), G27B_LAYER(18), G27B_LAYER(19),

    /* Layers 20-29 */
    G27B_LAYER(20), G27B_LAYER(21), G27B_LAYER(22), G27B_LAYER(23), G27B_LAYER(24),
    G27B_LAYER(25), G27B_LAYER(26), G27B_LAYER(27), G27B_LAYER(28), G27B_LAYER(29),

    /* Layers 30-39 */
    G27B_LAYER(30), G27B_LAYER(31), G27B_LAYER(32), G27B_LAYER(33), G27B_LAYER(34),
    G27B_LAYER(35), G27B_LAYER(36), G27B_LAYER(37), G27B_LAYER(38), G27B_LAYER(39),

    /* Layers 40-49 */
    G27B_LAYER(40), G27B_LAYER(41), G27B_LAYER(42), G27B_LAYER(43), G27B_LAYER(44),
    G27B_LAYER(45), G27B_LAYER(46), G27B_LAYER(47), G27B_LAYER(48), G27B_LAYER(49),

    /* Layers 50-59 */
    G27B_LAYER(50), G27B_LAYER(51), G27B_LAYER(52), G27B_LAYER(53), G27B_LAYER(54),
    G27B_LAYER(55), G27B_LAYER(56), G27B_LAYER(57), G27B_LAYER(58), G27B_LAYER(59),

    /* Layers 60-61 (final partial group — both local) */
    G27B_LAYER(60), G27B_LAYER(61),

    /* Final norm and LM head */
    {"language_model.model.norm.weight", "final", "norm_final_weight"},
    {"language_model.lm_head.weight",    "final", "lm_head_weight"},

    /* Sentinel */
    {NULL, NULL, NULL}
};

size_t get_gemma3_27b_tensor_map_size(void) {
    return (sizeof(GEMMA3_27B_TENSOR_MAP) / sizeof(GEMMA3_27B_TENSOR_MAP[0]) - 1u);
}

#undef G27B_LAYER
