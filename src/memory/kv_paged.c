#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "../include/kv_paged.h"
#include "../include/log.h"

struct kv_pager_t {
    kv_pager_config_t cfg;
    kv_page_t *pages;
    uint32_t num_pages;
    int32_t free_head;
    int32_t lru_head;
    int32_t lru_tail;
    uint32_t free_count;
    uint32_t used_count;
    uint32_t pinned_count;
    uint64_t tick;
    uint64_t alloc_count;
    uint64_t free_count_total;
    uint64_t evict_count;
    uint64_t touch_count;
};

static uint32_t read_env_u32(const char *name, uint32_t fallback, uint32_t min_v, uint32_t max_v) {
    const char *v = getenv(name);
    if (!v || !v[0]) return fallback;
    errno = 0;
    char *end = NULL;
    unsigned long parsed = strtoul(v, &end, 10);
    if (errno != 0 || end == v || (end && *end != '\0')) return fallback;
    if (parsed < min_v) return min_v;
    if (parsed > max_v) return max_v;
    return (uint32_t)parsed;
}

int kv_pager_config_from_env(kv_pager_config_t *cfg) {
    if (!cfg) return -1;
    memset(cfg, 0, sizeof(*cfg));
    cfg->page_tokens = read_env_u32("SAPPHIRE_KV_PAGE_TOKENS", 64u, 8u, 4096u);
    cfg->max_pages = read_env_u32("SAPPHIRE_KV_MAX_PAGES", 4096u, 1u, 2000000u);
    cfg->vram_budget_mb = read_env_u32("SAPPHIRE_KV_VRAM_BUDGET_MB", 2048u, 64u, 262144u);
    cfg->ram_budget_mb = read_env_u32("SAPPHIRE_KV_RAM_BUDGET_MB", 8192u, 64u, 1048576u);
    cfg->evict_watermark_pct = read_env_u32("SAPPHIRE_KV_EVICT_WATERMARK", 90u, 50u, 99u);
    cfg->low_watermark_pct = read_env_u32("SAPPHIRE_KV_LOW_WATERMARK", 80u, 30u, 98u);
    cfg->prefetch_pages = read_env_u32("SAPPHIRE_KV_PREFETCH_PAGES", 1u, 0u, 64u);
    cfg->int8_enable = (uint8_t)read_env_u32("SAPPHIRE_KV_INT8_ENABLE", 1u, 0u, 1u);
    if (cfg->low_watermark_pct >= cfg->evict_watermark_pct) {
        cfg->low_watermark_pct = (cfg->evict_watermark_pct > 5u) ? (cfg->evict_watermark_pct - 5u) : cfg->evict_watermark_pct;
    }
    return 0;
}

static void lru_detach(kv_pager_t *pager, int32_t idx) {
    kv_page_t *p = &pager->pages[idx];
    int32_t prev = p->prev_lru;
    int32_t next = p->next_lru;
    if (prev >= 0) pager->pages[prev].next_lru = next;
    if (next >= 0) pager->pages[next].prev_lru = prev;
    if (pager->lru_head == idx) pager->lru_head = next;
    if (pager->lru_tail == idx) pager->lru_tail = prev;
    p->prev_lru = -1;
    p->next_lru = -1;
}

static void lru_insert_head(kv_pager_t *pager, int32_t idx) {
    kv_page_t *p = &pager->pages[idx];
    p->prev_lru = -1;
    p->next_lru = pager->lru_head;
    if (pager->lru_head >= 0) pager->pages[pager->lru_head].prev_lru = idx;
    pager->lru_head = idx;
    if (pager->lru_tail < 0) pager->lru_tail = idx;
}

kv_pager_t *kv_pager_create(const kv_pager_config_t *cfg) {
    if (!cfg || cfg->max_pages == 0u || cfg->page_tokens == 0u) return NULL;
    kv_pager_t *pager = (kv_pager_t *)calloc(1, sizeof(kv_pager_t));
    if (!pager) return NULL;
    pager->cfg = *cfg;
    pager->num_pages = cfg->max_pages;
    pager->free_head = -1;
    pager->lru_head = -1;
    pager->lru_tail = -1;
    pager->pages = (kv_page_t *)calloc((size_t)pager->num_pages, sizeof(kv_page_t));
    if (!pager->pages) {
        free(pager);
        return NULL;
    }

    for (int32_t i = (int32_t)pager->num_pages - 1; i >= 0; --i) {
        kv_page_t *page = &pager->pages[i];
        page->page_id = (uint32_t)i;
        page->layer_id = UINT32_MAX;
        page->head_group_id = UINT32_MAX;
        page->token_start = UINT64_MAX;
        page->token_count = 0u;
        page->quant_bits = 16u;
        page->dirty = 0u;
        page->tier = KV_PAGE_TIER_RAM;
        page->pin_count = 0u;
        page->last_access_tick = 0u;
        page->k_data = NULL;
        page->v_data = NULL;
        page->disk_offset_k = 0u;
        page->disk_offset_v = 0u;
        page->prev_lru = -1;
        page->next_lru = -1;
        page->next_free = pager->free_head;
        pager->free_head = i;
    }

    pager->free_count = pager->num_pages;
    LOG_INFO("KV pager initialized: pages=%u page_tokens=%u vram=%uMB ram=%uMB",
             pager->num_pages,
             pager->cfg.page_tokens,
             pager->cfg.vram_budget_mb,
             pager->cfg.ram_budget_mb);
    return pager;
}

void kv_pager_destroy(kv_pager_t *pager) {
    if (!pager) return;
    free(pager->pages);
    free(pager);
}

void kv_pager_reset(kv_pager_t *pager) {
    if (!pager || !pager->pages) return;
    pager->free_head = -1;
    pager->lru_head = -1;
    pager->lru_tail = -1;
    pager->free_count = pager->num_pages;
    pager->used_count = 0u;
    pager->pinned_count = 0u;

    for (int32_t i = (int32_t)pager->num_pages - 1; i >= 0; --i) {
        kv_page_t *page = &pager->pages[i];
        page->layer_id = UINT32_MAX;
        page->head_group_id = UINT32_MAX;
        page->token_start = UINT64_MAX;
        page->token_count = 0u;
        page->dirty = 0u;
        page->tier = KV_PAGE_TIER_RAM;
        page->pin_count = 0u;
        page->last_access_tick = 0u;
        page->k_data = NULL;
        page->v_data = NULL;
        page->disk_offset_k = 0u;
        page->disk_offset_v = 0u;
        page->prev_lru = -1;
        page->next_lru = -1;
        page->next_free = pager->free_head;
        pager->free_head = i;
    }
}

kv_page_t *kv_pager_alloc_page(kv_pager_t *pager) {
    if (!pager || pager->free_head < 0) return NULL;
    int32_t idx = pager->free_head;
    kv_page_t *page = &pager->pages[idx];
    pager->free_head = page->next_free;
    page->next_free = -1;
    page->pin_count = 0u;
    page->dirty = 0u;
    page->last_access_tick = ++pager->tick;
    lru_insert_head(pager, idx);
    pager->free_count--;
    pager->used_count++;
    pager->alloc_count++;
    return page;
}

int kv_pager_free_page(kv_pager_t *pager, kv_page_t *page) {
    if (!pager || !page) return -1;
    uint32_t idx = page->page_id;
    if (idx >= pager->num_pages) return -1;
    if (page->pin_count > 0u) return -1;

    if (page->prev_lru >= 0 || page->next_lru >= 0 || pager->lru_head == (int32_t)idx) {
        lru_detach(pager, (int32_t)idx);
    }

    page->layer_id = UINT32_MAX;
    page->head_group_id = UINT32_MAX;
    page->token_start = UINT64_MAX;
    page->token_count = 0u;
    page->dirty = 0u;
    page->k_data = NULL;
    page->v_data = NULL;
    page->disk_offset_k = 0u;
    page->disk_offset_v = 0u;
    page->next_free = pager->free_head;
    pager->free_head = (int32_t)idx;

    pager->free_count++;
    if (pager->used_count > 0u) pager->used_count--;
    pager->free_count_total++;
    return 0;
}

int kv_pager_touch_page(kv_pager_t *pager, kv_page_t *page) {
    if (!pager || !page) return -1;
    uint32_t idx = page->page_id;
    if (idx >= pager->num_pages) return -1;
    page->last_access_tick = ++pager->tick;
    if (pager->lru_head != (int32_t)idx) {
        if (page->prev_lru >= 0 || page->next_lru >= 0) {
            lru_detach(pager, (int32_t)idx);
        }
        lru_insert_head(pager, (int32_t)idx);
    }
    pager->touch_count++;
    return 0;
}

kv_page_t *kv_pager_evict_candidate(kv_pager_t *pager) {
    if (!pager) return NULL;
    int32_t cur = pager->lru_tail;
    while (cur >= 0) {
        kv_page_t *page = &pager->pages[cur];
        if (page->pin_count == 0u) {
            pager->evict_count++;
            return page;
        }
        cur = page->prev_lru;
    }
    return NULL;
}

int kv_pager_pin_page(kv_page_t *page) {
    if (!page) return -1;
    page->pin_count++;
    return 0;
}

int kv_pager_unpin_page(kv_page_t *page) {
    if (!page || page->pin_count == 0u) return -1;
    page->pin_count--;
    return 0;
}

int kv_pager_get_stats(const kv_pager_t *pager, kv_pager_stats_t *out_stats) {
    if (!pager || !out_stats) return -1;
    uint32_t pinned = 0u;
    for (uint32_t i = 0; i < pager->num_pages; ++i) {
        if (pager->pages[i].pin_count > 0u) pinned++;
    }
    out_stats->total_pages = pager->num_pages;
    out_stats->free_pages = pager->free_count;
    out_stats->used_pages = pager->used_count;
    out_stats->pinned_pages = pinned;
    out_stats->alloc_count = pager->alloc_count;
    out_stats->free_count = pager->free_count_total;
    out_stats->evict_count = pager->evict_count;
    out_stats->touch_count = pager->touch_count;
    return 0;
}

uint32_t kv_pager_page_tokens(const kv_pager_t *pager) {
    if (!pager) return 0u;
    return pager->cfg.page_tokens;
}
