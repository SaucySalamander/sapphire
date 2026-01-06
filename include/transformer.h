#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Hyperparameters
typedef struct {
    int dimensions;
    int hidden_dimensions;
    int num_layers;
    int num_heads;
    int num_keyvalue_heads;
    int vocab_size;
    int sequence_length;
    float learning_rate;
} TransformerConfig;

// Trained model
typedef struct {
    float* token_embedding_table;

    float* wq;
    float* wk;
    float* wv;
    float* wo;

    float* rms_att_weight;
    float* rms_ffn_weight;
} TransformerWeights;

// Gradients: Error signals
typedef struct {
    float* dwq;
    float* dwk;
    float* dwv;
    float* dwo;

    float* d_rms_att;
    float* d_rms_ffn;
} TransformerGradients;

// Short Term memory
typedef struct {
    float* x;
    float* xb;
    float* q;
    float* k;
    float* v;
    float* logits;

    float* post_norm_x;
} RunState;

#endif // TRANSFORMER_H