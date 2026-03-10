#ifndef KV_PAGED_H
#define KV_PAGED_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    KV_PAGE_TIER_VRAM = 0,
    KV_PAGE_TIER_RAM = 1,
    KV_PAGE_TIER_DISK = 2
} kv_page_tier_t;

typedef struct kv_page_t {
    uint32_t page_id;
    uint32_t layer_id;
    uint32_t head_group_id;
    uint64_t token_start;
    uint32_t token_count;
    uint8_t quant_bits;
    uint8_t dirty;
    kv_page_tier_t tier;
    uint32_t pin_count;
    uint64_t last_access_tick;
    void *k_data;
    void *v_data;
    uint64_t disk_offset_k;
    uint64_t disk_offset_v;
    int32_t next_free;
    int32_t prev_lru;
    int32_t next_lru;
} kv_page_t;

typedef struct {
    uint32_t page_tokens;
    uint32_t max_pages;
    uint32_t vram_budget_mb;
    uint32_t ram_budget_mb;
    uint32_t evict_watermark_pct;
    uint32_t low_watermark_pct;
    uint32_t prefetch_pages;
    uint8_t int8_enable;
} kv_pager_config_t;

typedef struct kv_pager_t kv_pager_t;

typedef struct {
    uint32_t total_pages;
    uint32_t free_pages;
    uint32_t used_pages;
    uint32_t pinned_pages;
    uint64_t alloc_count;
    uint64_t free_count;
    uint64_t evict_count;
    uint64_t touch_count;
} kv_pager_stats_t;

int kv_pager_config_from_env(kv_pager_config_t *cfg);
kv_pager_t *kv_pager_create(const kv_pager_config_t *cfg);
void kv_pager_destroy(kv_pager_t *pager);
void kv_pager_reset(kv_pager_t *pager);
kv_page_t *kv_pager_alloc_page(kv_pager_t *pager);
int kv_pager_free_page(kv_pager_t *pager, kv_page_t *page);
int kv_pager_touch_page(kv_pager_t *pager, kv_page_t *page);
kv_page_t *kv_pager_evict_candidate(kv_pager_t *pager);
int kv_pager_pin_page(kv_page_t *page);
int kv_pager_unpin_page(kv_page_t *page);
int kv_pager_get_stats(const kv_pager_t *pager, kv_pager_stats_t *out_stats);
uint32_t kv_pager_page_tokens(const kv_pager_t *pager);

#endif
