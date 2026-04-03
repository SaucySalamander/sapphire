#ifndef VK_KV_PAGED_H
#define VK_KV_PAGED_H

#include <stdint.h>

#include "kv_paged.h"

typedef struct vk_kv_pager_t vk_kv_pager_t;

#define VK_KV_TRANSFER_OP_EVICT   1u
#define VK_KV_TRANSFER_OP_PROMOTE 2u

typedef struct {
    uint8_t type;
    uint32_t layer;
    uint32_t token_start;
    uint32_t token_count;
    uint32_t table_slot;
} vk_kv_transfer_op_t;

typedef struct {
    uint8_t enabled;
    uint32_t page_tokens;
    uint32_t max_pages;
} vk_kv_pager_config_t;

typedef struct {
    uint64_t page_hits;
    uint64_t page_misses;
    uint64_t page_binds;
    uint64_t page_evictions;
    uint64_t write_touches;
    uint64_t read_touches;
} vk_kv_pager_stats_t;

int vk_kv_pager_config_from_env(vk_kv_pager_config_t *cfg,
                                int num_layers,
                                int max_seq_len);

vk_kv_pager_t *vk_kv_pager_create(const vk_kv_pager_config_t *cfg,
                                  int num_layers,
                                  int max_seq_len,
                                  unsigned long long layer_types_mask,
                                  int sliding_window);

void vk_kv_pager_destroy(vk_kv_pager_t *pager);

void vk_kv_pager_reset(vk_kv_pager_t *pager);

int vk_kv_pager_touch_read_range(vk_kv_pager_t *pager,
                                 int layer,
                                 int start_pos,
                                 int end_pos);

int vk_kv_pager_touch_write_range(vk_kv_pager_t *pager,
                                  int layer,
                                  int start_pos,
                                  int end_pos);

int vk_kv_pager_get_stats(const vk_kv_pager_t *pager,
                          vk_kv_pager_stats_t *out_stats);

uint32_t vk_kv_pager_page_tokens(const vk_kv_pager_t *pager);
uint32_t vk_kv_pager_table_slots(const vk_kv_pager_t *pager);

uint32_t vk_kv_pager_drain_transfer_ops(vk_kv_pager_t *pager,
                                        vk_kv_transfer_op_t *out_ops,
                                        uint32_t max_ops);

#endif