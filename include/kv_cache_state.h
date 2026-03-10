#ifndef KV_CACHE_STATE_H
#define KV_CACHE_STATE_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t header_size;
    uint32_t num_layers;
    uint32_t num_kv_heads;
    uint32_t max_seq_len;
    uint32_t head_dim;
    uint32_t current_seq_len;
    uint32_t page_tokens;
    uint32_t page_float_count;
    uint64_t page_count;
    uint64_t reserved0;
    uint64_t reserved1;
} kv_state_file_header_t;

typedef struct {
    uint32_t layer;
    uint32_t page_idx;
    uint32_t token_start;
    uint32_t token_count;
    uint32_t tier;
    uint32_t flags;
} kv_state_page_record_t;

typedef struct kv_state_writer_t kv_state_writer_t;
typedef struct kv_state_reader_t kv_state_reader_t;

int kv_state_writer_open(const char *path,
                         const kv_state_file_header_t *header,
                         kv_state_writer_t **out_writer);
int kv_state_writer_write_page(kv_state_writer_t *writer,
                               const kv_state_page_record_t *record,
                               const float *k_page,
                               const float *v_page,
                               size_t page_float_count);
int kv_state_writer_close(kv_state_writer_t *writer);

int kv_state_reader_open(const char *path,
                         kv_state_file_header_t *out_header,
                         kv_state_reader_t **out_reader);
int kv_state_reader_read_page(kv_state_reader_t *reader,
                              kv_state_page_record_t *out_record,
                              float *out_k_page,
                              float *out_v_page,
                              size_t page_float_count);
int kv_state_reader_close(kv_state_reader_t *reader);

int kv_state_append_transcript(const char *path,
                               const int *tokens,
                               uint32_t token_count);
int kv_state_read_transcript(const char *path,
                             int *out_tokens,
                             uint32_t max_tokens,
                             uint32_t *out_token_count);

#endif
