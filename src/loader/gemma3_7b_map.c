/*
 * @file gemma3_7b_map.c
 * @brief Tensor name mapping table for Gemma 3 7B IT.
 *
 * Uses the same Gemma 3 HF tensor naming convention as the existing 4B/27B
 * checkpoints: language_model.model.layers.N.*
 */

#include "gemma3_7b_spec.h"

#define G7B_LAYER(N)                                                                               \
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

const tensor_map_entry_t GEMMA3_7B_TENSOR_MAP[] = {
    {"language_model.model.embed_tokens.weight", "embedding", "embedding_weight"},

    G7B_LAYER(0),  G7B_LAYER(1),  G7B_LAYER(2),  G7B_LAYER(3),  G7B_LAYER(4),
    G7B_LAYER(5),  G7B_LAYER(6),  G7B_LAYER(7),  G7B_LAYER(8),  G7B_LAYER(9),
    G7B_LAYER(10), G7B_LAYER(11), G7B_LAYER(12), G7B_LAYER(13), G7B_LAYER(14),
    G7B_LAYER(15), G7B_LAYER(16), G7B_LAYER(17), G7B_LAYER(18), G7B_LAYER(19),
    G7B_LAYER(20), G7B_LAYER(21), G7B_LAYER(22), G7B_LAYER(23), G7B_LAYER(24),
    G7B_LAYER(25), G7B_LAYER(26), G7B_LAYER(27),

    /* Final norm and LM head must stay within the trimmed 7B runtime prefix. */
    {"language_model.model.norm.weight", "final", "norm_final_weight"},
    {"language_model.lm_head.weight",    "final", "lm_head_weight"},

    G7B_LAYER(28), G7B_LAYER(29),
    G7B_LAYER(30), G7B_LAYER(31), G7B_LAYER(32), G7B_LAYER(33), G7B_LAYER(34),
    G7B_LAYER(35), G7B_LAYER(36), G7B_LAYER(37), G7B_LAYER(38), G7B_LAYER(39),
    G7B_LAYER(40), G7B_LAYER(41), G7B_LAYER(42), G7B_LAYER(43), G7B_LAYER(44),
    G7B_LAYER(45), G7B_LAYER(46), G7B_LAYER(47), G7B_LAYER(48), G7B_LAYER(49),
    G7B_LAYER(50), G7B_LAYER(51), G7B_LAYER(52), G7B_LAYER(53), G7B_LAYER(54),
    G7B_LAYER(55), G7B_LAYER(56), G7B_LAYER(57), G7B_LAYER(58), G7B_LAYER(59),
    G7B_LAYER(60), G7B_LAYER(61), G7B_LAYER(62), G7B_LAYER(63),

    {NULL, NULL, NULL}
};

size_t get_gemma3_7b_tensor_map_size(void)
{
    return (sizeof(GEMMA3_7B_TENSOR_MAP) / sizeof(GEMMA3_7B_TENSOR_MAP[0]) - 1u);
}

#undef G7B_LAYER