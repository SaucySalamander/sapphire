/*
 * @file gemma3_1b_map.c
 * @brief Tensor name mapping table for Gemma 3 1B IT.
 */

#include "gemma3_1b_spec.h"

#define G1B_LAYER(N)                                                                               \
    {"model.layers." #N ".input_layernorm.weight", "blk." #N, "norm_attn_weight"},             \
    {"model.layers." #N ".self_attn.q_proj.weight", "blk." #N, "q_proj_weight"},               \
    {"model.layers." #N ".self_attn.q_norm.weight", "blk." #N, "q_norm_weight"},               \
    {"model.layers." #N ".self_attn.k_proj.weight", "blk." #N, "k_proj_weight"},               \
    {"model.layers." #N ".self_attn.k_norm.weight", "blk." #N, "k_norm_weight"},               \
    {"model.layers." #N ".self_attn.v_proj.weight", "blk." #N, "v_proj_weight"},               \
    {"model.layers." #N ".self_attn.o_proj.weight", "blk." #N, "out_proj_weight"},             \
    {"model.layers." #N ".post_attention_layernorm.weight", "blk." #N, "norm_attn_post_weight"}, \
    {"model.layers." #N ".pre_feedforward_layernorm.weight", "blk." #N, "norm_ffn_weight"},    \
    {"model.layers." #N ".post_feedforward_layernorm.weight", "blk." #N, "norm_ffn_post_weight"}, \
    {"model.layers." #N ".mlp.gate_proj.weight", "blk." #N, "gate_proj_weight"},               \
    {"model.layers." #N ".mlp.up_proj.weight", "blk." #N, "up_proj_weight"},                   \
    {"model.layers." #N ".mlp.down_proj.weight", "blk." #N, "down_proj_weight"}

const tensor_map_entry_t GEMMA3_1B_TENSOR_MAP[] = {
    {"model.embed_tokens.weight", "embedding", "embedding_weight"},

    G1B_LAYER(0),  G1B_LAYER(1),  G1B_LAYER(2),  G1B_LAYER(3),  G1B_LAYER(4),
    G1B_LAYER(5),  G1B_LAYER(6),  G1B_LAYER(7),  G1B_LAYER(8),  G1B_LAYER(9),
    G1B_LAYER(10), G1B_LAYER(11), G1B_LAYER(12), G1B_LAYER(13), G1B_LAYER(14),
    G1B_LAYER(15), G1B_LAYER(16), G1B_LAYER(17), G1B_LAYER(18), G1B_LAYER(19),
    G1B_LAYER(20), G1B_LAYER(21), G1B_LAYER(22), G1B_LAYER(23), G1B_LAYER(24),
    G1B_LAYER(25),

    {"model.norm.weight", "final", "norm_final_weight"},
    {"lm_head.weight", "final", "lm_head_weight"},

    {NULL, NULL, NULL}
};

size_t get_gemma3_1b_tensor_map_size(void) {
    return (sizeof(GEMMA3_1B_TENSOR_MAP) / sizeof(GEMMA3_1B_TENSOR_MAP[0]) - 1u);
}

#undef G1B_LAYER