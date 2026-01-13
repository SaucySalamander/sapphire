/**
 * @file gemma3_270m_map.h
 * @brief Tensor name mapping table for Gemma 3 270M model.
 *
 * Maps Hugging Face Safetensors tensor names to Sapphire internal structure pointers.
 * This is a header-only translation layer that avoids cluttering loader logic.
 *
 * Gemma 3 270M Hybrid Architecture:
 * - Vocabulary: 262,144 tokens
 * - Embedding Dimension (d_model): 640
 * - Attention/Transformer Hidden Dimension (d_inner): 1024 (expansion of embedding)
 * - Attention Heads: 4 (Query)
 * - Attention Head Dimension: 256 (4 heads * 256 = 1024)
 * - Number of KV Heads: 1 (GQA with 4:1 reduction)
 * - Number of Layers: 18
 * - MLP Hidden Dimension (d_ff): 2048
 * - Total Parameters: ~270M
 *
 * Architecture Details:
 * The embedding layer outputs 640-dimensional vectors, which are then expanded to
 * 1024-dimensional hidden states for the transformer attention and FFN operations.
 * This "hybrid" design allows using smaller embeddings (efficient storage) while
 * maintaining computational capacity in the transformer core.
 *
 * HF Tensor Name Structure:
 * - Embeddings: model.embed_tokens.weight [262144, 640]
 * - Per-Layer (i=0..17):
 *   * Attention norm: model.layers.{i}.input_layernorm.weight [640]
 *   * Attention Q/K/V: model.layers.{i}.self_attn.{q,k,v}_proj.weight [1024, 640]
 *   * Attention O: model.layers.{i}.self_attn.o_proj.weight [640, 1024]
 *   * FFN norm: model.layers.{i}.post_attention_layernorm.weight [640]
 *   * FFN gate/up/down: model.layers.{i}.mlp.{gate,up,down}_proj.weight [2048, 640] / [640, 2048]
 * - Final norm: model.norm.weight [640]
 * - LM Head: lm_head.weight [262144, 640] (tied to embeddings in Gemma)
 *
 * Sapphire Internal Structure (llm_model_t):
 * - config.d_model = 640 (embedding dimension)
 * - config.d_inner = 1024 (attention hidden dimension)
 * - Embedding: embedding_weight [262144, 640]
 * - Per-Layer (i=0..17, model->layers[i]):
 *   * Attention norm: norm_attn_weight [640]
 *   * Attention: q_proj_weight [1024, 640], k_proj_weight [1024, 640],
 *               v_proj_weight [1024, 640], out_proj_weight [640, 1024]
 *   * FFN norm: norm_ffn_weight [640] (Mapped from post_attention_layernorm)
 *   * FFN: gate_proj_weight [2048, 640], up_proj_weight [2048, 640],
 *          down_proj_weight [640, 2048]
 * - Final norm: norm_final_weight [640]
 * - LM Head: lm_head_weight [262144, 640]
 *
 * @note This file defines no functions; it is a pure data file with
 *       mapping constants for use by tensor_mapper.c.
 */

#ifndef GEMMA3_270M_MAP_H
#define GEMMA3_270M_MAP_H

#include <stdint.h>
#include "model_spec.h"

/* Reuse the generic tensor_map_entry_t defined in model_spec.h */
static const tensor_map_entry_t GEMMA3_270M_TENSOR_MAP[] = {
    // Embedding layer
    {"model.embed_tokens.weight", "embedding", "embedding_weight"},

    // Layer 0
    {"model.layers.0.input_layernorm.weight", "blk.0", "norm_attn_weight"},
    {"model.layers.0.self_attn.q_proj.weight", "blk.0", "q_proj_weight"},
    {"model.layers.0.self_attn.q_norm.weight", "blk.0", "q_norm_weight"},
    {"model.layers.0.self_attn.k_proj.weight", "blk.0", "k_proj_weight"},
    {"model.layers.0.self_attn.k_norm.weight", "blk.0", "k_norm_weight"},
    {"model.layers.0.self_attn.v_proj.weight", "blk.0", "v_proj_weight"},
    {"model.layers.0.self_attn.o_proj.weight", "blk.0", "out_proj_weight"},
    {"model.layers.0.post_attention_layernorm.weight", "blk.0", "norm_attn_post_weight"},
    {"model.layers.0.pre_feedforward_layernorm.weight", "blk.0", "norm_ffn_weight"},
    {"model.layers.0.post_feedforward_layernorm.weight", "blk.0", "norm_ffn_post_weight"},
    {"model.layers.0.mlp.gate_proj.weight", "blk.0", "gate_proj_weight"},
    {"model.layers.0.mlp.up_proj.weight", "blk.0", "up_proj_weight"},
    {"model.layers.0.mlp.down_proj.weight", "blk.0", "down_proj_weight"},

    // Layer 1
    {"model.layers.1.input_layernorm.weight", "blk.1", "norm_attn_weight"},
    {"model.layers.1.self_attn.q_proj.weight", "blk.1", "q_proj_weight"},
    {"model.layers.1.self_attn.q_norm.weight", "blk.1", "q_norm_weight"},
    {"model.layers.1.self_attn.k_proj.weight", "blk.1", "k_proj_weight"},
    {"model.layers.1.self_attn.k_norm.weight", "blk.1", "k_norm_weight"},
    {"model.layers.1.self_attn.v_proj.weight", "blk.1", "v_proj_weight"},
    {"model.layers.1.self_attn.o_proj.weight", "blk.1", "out_proj_weight"},
    {"model.layers.1.post_attention_layernorm.weight", "blk.1", "norm_attn_post_weight"},
    {"model.layers.1.pre_feedforward_layernorm.weight", "blk.1", "norm_ffn_weight"},
    {"model.layers.1.post_feedforward_layernorm.weight", "blk.1", "norm_ffn_post_weight"},
    {"model.layers.1.mlp.gate_proj.weight", "blk.1", "gate_proj_weight"},
    {"model.layers.1.mlp.up_proj.weight", "blk.1", "up_proj_weight"},
    {"model.layers.1.mlp.down_proj.weight", "blk.1", "down_proj_weight"},

    // Layer 2
    {"model.layers.2.input_layernorm.weight", "blk.2", "norm_attn_weight"},
    {"model.layers.2.self_attn.q_proj.weight", "blk.2", "q_proj_weight"},
    {"model.layers.2.self_attn.q_norm.weight", "blk.2", "q_norm_weight"},
    {"model.layers.2.self_attn.k_proj.weight", "blk.2", "k_proj_weight"},
    {"model.layers.2.self_attn.k_norm.weight", "blk.2", "k_norm_weight"},
    {"model.layers.2.self_attn.v_proj.weight", "blk.2", "v_proj_weight"},
    {"model.layers.2.self_attn.o_proj.weight", "blk.2", "out_proj_weight"},
    {"model.layers.2.post_attention_layernorm.weight", "blk.2", "norm_attn_post_weight"},
    {"model.layers.2.pre_feedforward_layernorm.weight", "blk.2", "norm_ffn_weight"},
    {"model.layers.2.post_feedforward_layernorm.weight", "blk.2", "norm_ffn_post_weight"},
    {"model.layers.2.mlp.gate_proj.weight", "blk.2", "gate_proj_weight"},
    {"model.layers.2.mlp.up_proj.weight", "blk.2", "up_proj_weight"},
    {"model.layers.2.mlp.down_proj.weight", "blk.2", "down_proj_weight"},

    // Layer 3
    {"model.layers.3.input_layernorm.weight", "blk.3", "norm_attn_weight"},
    {"model.layers.3.self_attn.q_proj.weight", "blk.3", "q_proj_weight"},
    {"model.layers.3.self_attn.q_norm.weight", "blk.3", "q_norm_weight"},
    {"model.layers.3.self_attn.k_proj.weight", "blk.3", "k_proj_weight"},
    {"model.layers.3.self_attn.k_norm.weight", "blk.3", "k_norm_weight"},
    {"model.layers.3.self_attn.v_proj.weight", "blk.3", "v_proj_weight"},
    {"model.layers.3.self_attn.o_proj.weight", "blk.3", "out_proj_weight"},
    {"model.layers.3.post_attention_layernorm.weight", "blk.3", "norm_attn_post_weight"},
    {"model.layers.3.pre_feedforward_layernorm.weight", "blk.3", "norm_ffn_weight"},
    {"model.layers.3.post_feedforward_layernorm.weight", "blk.3", "norm_ffn_post_weight"},
    {"model.layers.3.mlp.gate_proj.weight", "blk.3", "gate_proj_weight"},
    {"model.layers.3.mlp.up_proj.weight", "blk.3", "up_proj_weight"},
    {"model.layers.3.mlp.down_proj.weight", "blk.3", "down_proj_weight"},

    // Layer 4
    {"model.layers.4.input_layernorm.weight", "blk.4", "norm_attn_weight"},
    {"model.layers.4.self_attn.q_proj.weight", "blk.4", "q_proj_weight"},
    {"model.layers.4.self_attn.q_norm.weight", "blk.4", "q_norm_weight"},
    {"model.layers.4.self_attn.k_proj.weight", "blk.4", "k_proj_weight"},
    {"model.layers.4.self_attn.k_norm.weight", "blk.4", "k_norm_weight"},
    {"model.layers.4.self_attn.v_proj.weight", "blk.4", "v_proj_weight"},
    {"model.layers.4.self_attn.o_proj.weight", "blk.4", "out_proj_weight"},
    {"model.layers.4.post_attention_layernorm.weight", "blk.4", "norm_attn_post_weight"},
    {"model.layers.4.pre_feedforward_layernorm.weight", "blk.4", "norm_ffn_weight"},
    {"model.layers.4.post_feedforward_layernorm.weight", "blk.4", "norm_ffn_post_weight"},
    {"model.layers.4.mlp.gate_proj.weight", "blk.4", "gate_proj_weight"},
    {"model.layers.4.mlp.up_proj.weight", "blk.4", "up_proj_weight"},
    {"model.layers.4.mlp.down_proj.weight", "blk.4", "down_proj_weight"},

    // Layer 5
    {"model.layers.5.input_layernorm.weight", "blk.5", "norm_attn_weight"},
    {"model.layers.5.self_attn.q_proj.weight", "blk.5", "q_proj_weight"},
    {"model.layers.5.self_attn.q_norm.weight", "blk.5", "q_norm_weight"},
    {"model.layers.5.self_attn.k_proj.weight", "blk.5", "k_proj_weight"},
    {"model.layers.5.self_attn.k_norm.weight", "blk.5", "k_norm_weight"},
    {"model.layers.5.self_attn.v_proj.weight", "blk.5", "v_proj_weight"},
    {"model.layers.5.self_attn.o_proj.weight", "blk.5", "out_proj_weight"},
    {"model.layers.5.post_attention_layernorm.weight", "blk.5", "norm_attn_post_weight"},
    {"model.layers.5.pre_feedforward_layernorm.weight", "blk.5", "norm_ffn_weight"},
    {"model.layers.5.post_feedforward_layernorm.weight", "blk.5", "norm_ffn_post_weight"},
    {"model.layers.5.mlp.gate_proj.weight", "blk.5", "gate_proj_weight"},
    {"model.layers.5.mlp.up_proj.weight", "blk.5", "up_proj_weight"},
    {"model.layers.5.mlp.down_proj.weight", "blk.5", "down_proj_weight"},

    // Layer 6
    {"model.layers.6.input_layernorm.weight", "blk.6", "norm_attn_weight"},
    {"model.layers.6.self_attn.q_proj.weight", "blk.6", "q_proj_weight"},
    {"model.layers.6.self_attn.q_norm.weight", "blk.6", "q_norm_weight"},
    {"model.layers.6.self_attn.k_proj.weight", "blk.6", "k_proj_weight"},
    {"model.layers.6.self_attn.k_norm.weight", "blk.6", "k_norm_weight"},
    {"model.layers.6.self_attn.v_proj.weight", "blk.6", "v_proj_weight"},
    {"model.layers.6.self_attn.o_proj.weight", "blk.6", "out_proj_weight"},
    {"model.layers.6.post_attention_layernorm.weight", "blk.6", "norm_attn_post_weight"},
    {"model.layers.6.pre_feedforward_layernorm.weight", "blk.6", "norm_ffn_weight"},
    {"model.layers.6.post_feedforward_layernorm.weight", "blk.6", "norm_ffn_post_weight"},
    {"model.layers.6.mlp.gate_proj.weight", "blk.6", "gate_proj_weight"},
    {"model.layers.6.mlp.up_proj.weight", "blk.6", "up_proj_weight"},
    {"model.layers.6.mlp.down_proj.weight", "blk.6", "down_proj_weight"},

    // Layer 7
    {"model.layers.7.input_layernorm.weight", "blk.7", "norm_attn_weight"},
    {"model.layers.7.self_attn.q_proj.weight", "blk.7", "q_proj_weight"},
    {"model.layers.7.self_attn.q_norm.weight", "blk.7", "q_norm_weight"},
    {"model.layers.7.self_attn.k_proj.weight", "blk.7", "k_proj_weight"},
    {"model.layers.7.self_attn.k_norm.weight", "blk.7", "k_norm_weight"},
    {"model.layers.7.self_attn.v_proj.weight", "blk.7", "v_proj_weight"},
    {"model.layers.7.self_attn.o_proj.weight", "blk.7", "out_proj_weight"},
    {"model.layers.7.post_attention_layernorm.weight", "blk.7", "norm_attn_post_weight"},
    {"model.layers.7.pre_feedforward_layernorm.weight", "blk.7", "norm_ffn_weight"},
    {"model.layers.7.post_feedforward_layernorm.weight", "blk.7", "norm_ffn_post_weight"},
    {"model.layers.7.mlp.gate_proj.weight", "blk.7", "gate_proj_weight"},
    {"model.layers.7.mlp.up_proj.weight", "blk.7", "up_proj_weight"},
    {"model.layers.7.mlp.down_proj.weight", "blk.7", "down_proj_weight"},

    // Layer 8
    {"model.layers.8.input_layernorm.weight", "blk.8", "norm_attn_weight"},
    {"model.layers.8.self_attn.q_proj.weight", "blk.8", "q_proj_weight"},
    {"model.layers.8.self_attn.q_norm.weight", "blk.8", "q_norm_weight"},
    {"model.layers.8.self_attn.k_proj.weight", "blk.8", "k_proj_weight"},
    {"model.layers.8.self_attn.k_norm.weight", "blk.8", "k_norm_weight"},
    {"model.layers.8.self_attn.v_proj.weight", "blk.8", "v_proj_weight"},
    {"model.layers.8.self_attn.o_proj.weight", "blk.8", "out_proj_weight"},
    {"model.layers.8.post_attention_layernorm.weight", "blk.8", "norm_attn_post_weight"},
    {"model.layers.8.pre_feedforward_layernorm.weight", "blk.8", "norm_ffn_weight"},
    {"model.layers.8.post_feedforward_layernorm.weight", "blk.8", "norm_ffn_post_weight"},
    {"model.layers.8.mlp.gate_proj.weight", "blk.8", "gate_proj_weight"},
    {"model.layers.8.mlp.up_proj.weight", "blk.8", "up_proj_weight"},
    {"model.layers.8.mlp.down_proj.weight", "blk.8", "down_proj_weight"},

    // Layer 9
    {"model.layers.9.input_layernorm.weight", "blk.9", "norm_attn_weight"},
    {"model.layers.9.self_attn.q_proj.weight", "blk.9", "q_proj_weight"},
    {"model.layers.9.self_attn.q_norm.weight", "blk.9", "q_norm_weight"},
    {"model.layers.9.self_attn.k_proj.weight", "blk.9", "k_proj_weight"},
    {"model.layers.9.self_attn.k_norm.weight", "blk.9", "k_norm_weight"},
    {"model.layers.9.self_attn.v_proj.weight", "blk.9", "v_proj_weight"},
    {"model.layers.9.self_attn.o_proj.weight", "blk.9", "out_proj_weight"},
    {"model.layers.9.post_attention_layernorm.weight", "blk.9", "norm_attn_post_weight"},
    {"model.layers.9.pre_feedforward_layernorm.weight", "blk.9", "norm_ffn_weight"},
    {"model.layers.9.post_feedforward_layernorm.weight", "blk.9", "norm_ffn_post_weight"},
    {"model.layers.9.mlp.gate_proj.weight", "blk.9", "gate_proj_weight"},
    {"model.layers.9.mlp.up_proj.weight", "blk.9", "up_proj_weight"},
    {"model.layers.9.mlp.down_proj.weight", "blk.9", "down_proj_weight"},

    // Layer 10
    {"model.layers.10.input_layernorm.weight", "blk.10", "norm_attn_weight"},
    {"model.layers.10.self_attn.q_proj.weight", "blk.10", "q_proj_weight"},
    {"model.layers.10.self_attn.q_norm.weight", "blk.10", "q_norm_weight"},
    {"model.layers.10.self_attn.k_proj.weight", "blk.10", "k_proj_weight"},
    {"model.layers.10.self_attn.k_norm.weight", "blk.10", "k_norm_weight"},
    {"model.layers.10.self_attn.v_proj.weight", "blk.10", "v_proj_weight"},
    {"model.layers.10.self_attn.o_proj.weight", "blk.10", "out_proj_weight"},
    {"model.layers.10.post_attention_layernorm.weight", "blk.10", "norm_attn_post_weight"},
    {"model.layers.10.pre_feedforward_layernorm.weight", "blk.10", "norm_ffn_weight"},
    {"model.layers.10.post_feedforward_layernorm.weight", "blk.10", "norm_ffn_post_weight"},
    {"model.layers.10.mlp.gate_proj.weight", "blk.10", "gate_proj_weight"},
    {"model.layers.10.mlp.up_proj.weight", "blk.10", "up_proj_weight"},
    {"model.layers.10.mlp.down_proj.weight", "blk.10", "down_proj_weight"},

    // Layer 11
    {"model.layers.11.input_layernorm.weight", "blk.11", "norm_attn_weight"},
    {"model.layers.11.self_attn.q_proj.weight", "blk.11", "q_proj_weight"},
    {"model.layers.11.self_attn.q_norm.weight", "blk.11", "q_norm_weight"},
    {"model.layers.11.self_attn.k_proj.weight", "blk.11", "k_proj_weight"},
    {"model.layers.11.self_attn.k_norm.weight", "blk.11", "k_norm_weight"},
    {"model.layers.11.self_attn.v_proj.weight", "blk.11", "v_proj_weight"},
    {"model.layers.11.self_attn.o_proj.weight", "blk.11", "out_proj_weight"},
    {"model.layers.11.post_attention_layernorm.weight", "blk.11", "norm_attn_post_weight"},
    {"model.layers.11.pre_feedforward_layernorm.weight", "blk.11", "norm_ffn_weight"},
    {"model.layers.11.post_feedforward_layernorm.weight", "blk.11", "norm_ffn_post_weight"},
    {"model.layers.11.mlp.gate_proj.weight", "blk.11", "gate_proj_weight"},
    {"model.layers.11.mlp.up_proj.weight", "blk.11", "up_proj_weight"},
    {"model.layers.11.mlp.down_proj.weight", "blk.11", "down_proj_weight"},

    // Layer 12
    {"model.layers.12.input_layernorm.weight", "blk.12", "norm_attn_weight"},
    {"model.layers.12.self_attn.q_proj.weight", "blk.12", "q_proj_weight"},
    {"model.layers.12.self_attn.q_norm.weight", "blk.12", "q_norm_weight"},
    {"model.layers.12.self_attn.k_proj.weight", "blk.12", "k_proj_weight"},
    {"model.layers.12.self_attn.k_norm.weight", "blk.12", "k_norm_weight"},
    {"model.layers.12.self_attn.v_proj.weight", "blk.12", "v_proj_weight"},
    {"model.layers.12.self_attn.o_proj.weight", "blk.12", "out_proj_weight"},
    {"model.layers.12.post_attention_layernorm.weight", "blk.12", "norm_attn_post_weight"},
    {"model.layers.12.pre_feedforward_layernorm.weight", "blk.12", "norm_ffn_weight"},
    {"model.layers.12.post_feedforward_layernorm.weight", "blk.12", "norm_ffn_post_weight"},
    {"model.layers.12.mlp.gate_proj.weight", "blk.12", "gate_proj_weight"},
    {"model.layers.12.mlp.up_proj.weight", "blk.12", "up_proj_weight"},
    {"model.layers.12.mlp.down_proj.weight", "blk.12", "down_proj_weight"},

    // Layer 13
    {"model.layers.13.input_layernorm.weight", "blk.13", "norm_attn_weight"},
    {"model.layers.13.self_attn.q_proj.weight", "blk.13", "q_proj_weight"},
    {"model.layers.13.self_attn.q_norm.weight", "blk.13", "q_norm_weight"},
    {"model.layers.13.self_attn.k_proj.weight", "blk.13", "k_proj_weight"},
    {"model.layers.13.self_attn.k_norm.weight", "blk.13", "k_norm_weight"},
    {"model.layers.13.self_attn.v_proj.weight", "blk.13", "v_proj_weight"},
    {"model.layers.13.self_attn.o_proj.weight", "blk.13", "out_proj_weight"},
    {"model.layers.13.post_attention_layernorm.weight", "blk.13", "norm_attn_post_weight"},
    {"model.layers.13.pre_feedforward_layernorm.weight", "blk.13", "norm_ffn_weight"},
    {"model.layers.13.post_feedforward_layernorm.weight", "blk.13", "norm_ffn_post_weight"},
    {"model.layers.13.mlp.gate_proj.weight", "blk.13", "gate_proj_weight"},
    {"model.layers.13.mlp.up_proj.weight", "blk.13", "up_proj_weight"},
    {"model.layers.13.mlp.down_proj.weight", "blk.13", "down_proj_weight"},

    // Layer 14
    {"model.layers.14.input_layernorm.weight", "blk.14", "norm_attn_weight"},
    {"model.layers.14.self_attn.q_proj.weight", "blk.14", "q_proj_weight"},
    {"model.layers.14.self_attn.q_norm.weight", "blk.14", "q_norm_weight"},
    {"model.layers.14.self_attn.k_proj.weight", "blk.14", "k_proj_weight"},
    {"model.layers.14.self_attn.k_norm.weight", "blk.14", "k_norm_weight"},
    {"model.layers.14.self_attn.v_proj.weight", "blk.14", "v_proj_weight"},
    {"model.layers.14.self_attn.o_proj.weight", "blk.14", "out_proj_weight"},
    {"model.layers.14.post_attention_layernorm.weight", "blk.14", "norm_attn_post_weight"},
    {"model.layers.14.pre_feedforward_layernorm.weight", "blk.14", "norm_ffn_weight"},
    {"model.layers.14.post_feedforward_layernorm.weight", "blk.14", "norm_ffn_post_weight"},
    {"model.layers.14.mlp.gate_proj.weight", "blk.14", "gate_proj_weight"},
    {"model.layers.14.mlp.up_proj.weight", "blk.14", "up_proj_weight"},
    {"model.layers.14.mlp.down_proj.weight", "blk.14", "down_proj_weight"},

    // Layer 15
    {"model.layers.15.input_layernorm.weight", "blk.15", "norm_attn_weight"},
    {"model.layers.15.self_attn.q_proj.weight", "blk.15", "q_proj_weight"},
    {"model.layers.15.self_attn.q_norm.weight", "blk.15", "q_norm_weight"},
    {"model.layers.15.self_attn.k_proj.weight", "blk.15", "k_proj_weight"},
    {"model.layers.15.self_attn.k_norm.weight", "blk.15", "k_norm_weight"},
    {"model.layers.15.self_attn.v_proj.weight", "blk.15", "v_proj_weight"},
    {"model.layers.15.self_attn.o_proj.weight", "blk.15", "out_proj_weight"},
    {"model.layers.15.post_attention_layernorm.weight", "blk.15", "norm_attn_post_weight"},
    {"model.layers.15.pre_feedforward_layernorm.weight", "blk.15", "norm_ffn_weight"},
    {"model.layers.15.post_feedforward_layernorm.weight", "blk.15", "norm_ffn_post_weight"},
    {"model.layers.15.mlp.gate_proj.weight", "blk.15", "gate_proj_weight"},
    {"model.layers.15.mlp.up_proj.weight", "blk.15", "up_proj_weight"},
    {"model.layers.15.mlp.down_proj.weight", "blk.15", "down_proj_weight"},

    // Layer 16
    {"model.layers.16.input_layernorm.weight", "blk.16", "norm_attn_weight"},
    {"model.layers.16.self_attn.q_proj.weight", "blk.16", "q_proj_weight"},
    {"model.layers.16.self_attn.q_norm.weight", "blk.16", "q_norm_weight"},
    {"model.layers.16.self_attn.k_proj.weight", "blk.16", "k_proj_weight"},
    {"model.layers.16.self_attn.k_norm.weight", "blk.16", "k_norm_weight"},
    {"model.layers.16.self_attn.v_proj.weight", "blk.16", "v_proj_weight"},
    {"model.layers.16.self_attn.o_proj.weight", "blk.16", "out_proj_weight"},
    {"model.layers.16.post_attention_layernorm.weight", "blk.16", "norm_attn_post_weight"},
    {"model.layers.16.pre_feedforward_layernorm.weight", "blk.16", "norm_ffn_weight"},
    {"model.layers.16.post_feedforward_layernorm.weight", "blk.16", "norm_ffn_post_weight"},
    {"model.layers.16.mlp.gate_proj.weight", "blk.16", "gate_proj_weight"},
    {"model.layers.16.mlp.up_proj.weight", "blk.16", "up_proj_weight"},
    {"model.layers.16.mlp.down_proj.weight", "blk.16", "down_proj_weight"},

    // Layer 17
    {"model.layers.17.input_layernorm.weight", "blk.17", "norm_attn_weight"},
    {"model.layers.17.self_attn.q_proj.weight", "blk.17", "q_proj_weight"},
    {"model.layers.17.self_attn.q_norm.weight", "blk.17", "q_norm_weight"},
    {"model.layers.17.self_attn.k_proj.weight", "blk.17", "k_proj_weight"},
    {"model.layers.17.self_attn.k_norm.weight", "blk.17", "k_norm_weight"},
    {"model.layers.17.self_attn.v_proj.weight", "blk.17", "v_proj_weight"},
    {"model.layers.17.self_attn.o_proj.weight", "blk.17", "out_proj_weight"},
    {"model.layers.17.post_attention_layernorm.weight", "blk.17", "norm_attn_post_weight"},
    {"model.layers.17.pre_feedforward_layernorm.weight", "blk.17", "norm_ffn_weight"},
    {"model.layers.17.post_feedforward_layernorm.weight", "blk.17", "norm_ffn_post_weight"},
    {"model.layers.17.mlp.gate_proj.weight", "blk.17", "gate_proj_weight"},
    {"model.layers.17.mlp.up_proj.weight", "blk.17", "up_proj_weight"},
    {"model.layers.17.mlp.down_proj.weight", "blk.17", "down_proj_weight"},

    // Final norm and LM head
    {"model.norm.weight", "final", "norm_final_weight"},
    {"lm_head.weight", "final", "lm_head_weight"},

    // Sentinel (marks end of array)
    {NULL, NULL, NULL}
};

/**
 * @brief Number of entries in the Gemma 3 270M mapping table (excluding sentinel).
 */
#define GEMMA3_270M_TENSOR_MAP_SIZE (sizeof(GEMMA3_270M_TENSOR_MAP) / sizeof(GEMMA3_270M_TENSOR_MAP[0]) - 1)

#endif // GEMMA3_270M_MAP_H
