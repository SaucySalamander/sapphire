#include "../../include/vk_kv_paged.h"

#include <stdlib.h>
#include <string.h>

#include "../../include/log.h"

struct vk_kv_pager_t {
    kv_pager_t *pager;
    kv_page_t **page_table;
    uint32_t pages_per_layer;
    uint32_t page_tokens;
    uint32_t page_table_slots;
    int num_layers;
    int max_seq_len;
    int sliding_window;
    int latest_seq_pos_exclusive;
    unsigned long long layer_types_mask;
    uint64_t page_hits;
    uint64_t page_misses;
    uint64_t page_binds;
    uint64_t page_evictions;
    uint64_t write_touches;
    uint64_t read_touches;
    uint8_t *shadow_valid;
    uint8_t *pending_promote;
    uint8_t *pending_evict;
    vk_kv_transfer_op_t *pending_ops;
    uint32_t pending_count;
    uint32_t pending_capacity;
};

typedef enum {
    VK_KV_PAGE_CLASS_LOCAL_EXPIRED = 0,
    VK_KV_PAGE_CLASS_GLOBAL = 1,
    VK_KV_PAGE_CLASS_LOCAL_ACTIVE = 2
} vk_kv_page_class_t;

static int pager_layer_is_global(const vk_kv_pager_t *p, uint32_t layer) {
    if (!p || layer >= (uint32_t)p->num_layers) return 0;
    if (p->layer_types_mask != 0ULL) {
        return (((p->layer_types_mask >> layer) & 1ULL) != 0ULL) ? 1 : 0;
    }
    return ((((int)layer + 1) % 6) == 0) ? 1 : 0;
}

static int pager_local_window_tokens(const vk_kv_pager_t *p) {
    if (!p) return 1024;
    return (p->sliding_window > 0) ? p->sliding_window : 1024;
}

static int pager_local_window_start(const vk_kv_pager_t *p) {
    int window_tokens = pager_local_window_tokens(p);
    if (!p || p->latest_seq_pos_exclusive <= window_tokens) return 0;
    return p->latest_seq_pos_exclusive - window_tokens;
}

static vk_kv_page_class_t classify_page(const vk_kv_pager_t *p, const kv_page_t *page) {
    if (!p || !page) return VK_KV_PAGE_CLASS_GLOBAL;
    if (page->layer_id == UINT32_MAX || pager_layer_is_global(p, page->layer_id)) {
        return VK_KV_PAGE_CLASS_GLOBAL;
    }

    int active_start = pager_local_window_start(p);
    int active_end = p->latest_seq_pos_exclusive;
    int page_start = (page->token_start == UINT64_MAX) ? 0 : (int)page->token_start;
    int page_end = page_start + (int)page->token_count;
    if (page_end > active_start && page_start < active_end) {
        return VK_KV_PAGE_CLASS_LOCAL_ACTIVE;
    }
    return VK_KV_PAGE_CLASS_LOCAL_EXPIRED;
}

static void pin_non_class_pages(vk_kv_pager_t *p, vk_kv_page_class_t keep_class) {
    if (!p || !p->page_table) return;
    for (uint32_t slot = 0; slot < p->page_table_slots; ++slot) {
        kv_page_t *page = p->page_table[slot];
        if (!page) continue;
        if (classify_page(p, page) != keep_class) {
            (void)kv_pager_pin_page(page);
        }
    }
}

static void unpin_non_class_pages(vk_kv_pager_t *p, vk_kv_page_class_t keep_class) {
    if (!p || !p->page_table) return;
    for (uint32_t slot = 0; slot < p->page_table_slots; ++slot) {
        kv_page_t *page = p->page_table[slot];
        if (!page) continue;
        if (classify_page(p, page) != keep_class) {
            (void)kv_pager_unpin_page(page);
        }
    }
}

static kv_page_t *select_evict_candidate_for_class(vk_kv_pager_t *p,
                                                   vk_kv_page_class_t target_class) {
    kv_page_t *victim = NULL;
    if (!p || !p->pager) return NULL;

    pin_non_class_pages(p, target_class);
    victim = kv_pager_evict_candidate(p->pager);
    unpin_non_class_pages(p, target_class);

    if (!victim) return NULL;
    return (classify_page(p, victim) == target_class) ? victim : NULL;
}

static kv_page_t *select_priority_evict_candidate(vk_kv_pager_t *p) {
    kv_page_t *victim = select_evict_candidate_for_class(p, VK_KV_PAGE_CLASS_LOCAL_EXPIRED);
    if (victim) return victim;

    victim = select_evict_candidate_for_class(p, VK_KV_PAGE_CLASS_GLOBAL);
    if (victim) return victim;

    return select_evict_candidate_for_class(p, VK_KV_PAGE_CLASS_LOCAL_ACTIVE);
}

static void clear_pending_for_slot(vk_kv_pager_t *p, uint32_t slot) {
    if (!p || slot >= p->page_table_slots) return;
    p->pending_promote[slot] = 0u;
    p->pending_evict[slot] = 0u;
}

static int enqueue_transfer_op(vk_kv_pager_t *p,
                               uint8_t type,
                               uint32_t slot,
                               uint32_t layer,
                               uint32_t token_start,
                               uint32_t token_count) {
    if (!p || slot >= p->page_table_slots || !p->pending_ops) return -1;
    if (type == VK_KV_TRANSFER_OP_PROMOTE && p->pending_promote[slot]) return 0;
    if (type == VK_KV_TRANSFER_OP_EVICT && p->pending_evict[slot]) return 0;

    if (p->pending_count >= p->pending_capacity) {
        LOG_WARN("VK KV pager transfer queue full; dropping op type=%u slot=%u", (unsigned)type, slot);
        return -1;
    }

    vk_kv_transfer_op_t *op = &p->pending_ops[p->pending_count++];
    op->type = type;
    op->layer = layer;
    op->token_start = token_start;
    op->token_count = token_count;
    op->table_slot = slot;

    if (type == VK_KV_TRANSFER_OP_PROMOTE) p->pending_promote[slot] = 1u;
    if (type == VK_KV_TRANSFER_OP_EVICT) p->pending_evict[slot] = 1u;
    return 0;
}

static uint32_t read_env_u32(const char *name, uint32_t fallback) {
    const char *v = getenv(name);
    if (!v || v[0] == '\0') return fallback;
    char *end = NULL;
    unsigned long parsed = strtoul(v, &end, 10);
    if (end == v || *end != '\0') return fallback;
    if (parsed > 0xFFFFFFFFul) return fallback;
    return (uint32_t)parsed;
}

static uint32_t compute_default_max_pages(int num_layers, int max_seq_len, uint32_t page_tokens) {
    if (num_layers <= 0 || max_seq_len <= 0 || page_tokens == 0u) return 0u;
    uint32_t pages_per_layer = (uint32_t)(((uint32_t)max_seq_len + page_tokens - 1u) / page_tokens);
    return pages_per_layer * (uint32_t)num_layers;
}

static int get_slot_for_pos(const vk_kv_pager_t *p, int layer, int pos, uint32_t *out_slot) {
    if (!p || !out_slot || !p->page_table || p->page_tokens == 0u || p->pages_per_layer == 0u) return -1;
    if (layer < 0 || layer >= p->num_layers) return -1;
    if (pos < 0 || pos >= p->max_seq_len) return -1;
    uint32_t page_idx = (uint32_t)pos / p->page_tokens;
    if (page_idx >= p->pages_per_layer) return -1;
    *out_slot = (uint32_t)layer * p->pages_per_layer + page_idx;
    return 0;
}

static void clear_page_owner(vk_kv_pager_t *p, const kv_page_t *page) {
    if (!p || !p->page_table || !page || p->page_tokens == 0u || p->pages_per_layer == 0u) return;
    if (page->layer_id == UINT32_MAX || page->token_start == UINT64_MAX) return;
    if (page->layer_id >= (uint32_t)p->num_layers) return;
    uint32_t page_idx = (uint32_t)(page->token_start / p->page_tokens);
    if (page_idx >= p->pages_per_layer) return;
    uint32_t slot = page->layer_id * p->pages_per_layer + page_idx;
    if (slot < p->page_table_slots && p->page_table[slot] == page) {
        p->page_table[slot] = NULL;
    }
}

static kv_page_t *bind_page_for_pos(vk_kv_pager_t *p, int layer, int pos) {
    uint32_t slot = 0u;
    if (get_slot_for_pos(p, layer, pos, &slot) != 0) return NULL;

    kv_page_t *page = p->page_table[slot];
    if (page) {
        p->page_hits++;
        kv_pager_touch_page(p->pager, page);
        return page;
    }

    p->page_misses++;
    page = kv_pager_alloc_page(p->pager);
    if (!page) {
        kv_page_t *victim = select_priority_evict_candidate(p);
        if (victim) {
            uint32_t victim_slot = 0u;
            if (get_slot_for_pos(p, (int)victim->layer_id, (int)victim->token_start, &victim_slot) == 0) {
                (void)enqueue_transfer_op(p,
                                          VK_KV_TRANSFER_OP_EVICT,
                                          victim_slot,
                                          victim->layer_id,
                                          (uint32_t)victim->token_start,
                                          victim->token_count);
                p->shadow_valid[victim_slot] = 1u;
            }
            clear_page_owner(p, victim);
            if (kv_pager_free_page(p->pager, victim) == 0) {
                p->page_evictions++;
                page = kv_pager_alloc_page(p->pager);
            }
        }
    }
    if (!page) return NULL;

    page->layer_id = (uint32_t)layer;
    page->head_group_id = 0u;
    page->token_start = ((uint64_t)pos / p->page_tokens) * p->page_tokens;
    page->token_count = p->page_tokens;
    if ((int)(page->token_start + page->token_count) > p->max_seq_len) {
        page->token_count = (uint32_t)(p->max_seq_len - (int)page->token_start);
    }
    page->tier = KV_PAGE_TIER_VRAM;
    page->dirty = 0u;
    p->page_table[slot] = page;
    p->page_binds++;

    if (p->shadow_valid[slot]) {
        (void)enqueue_transfer_op(p,
                                  VK_KV_TRANSFER_OP_PROMOTE,
                                  slot,
                                  page->layer_id,
                                  (uint32_t)page->token_start,
                                  page->token_count);
        p->shadow_valid[slot] = 0u;
    }

    return page;
}

static int touch_range(vk_kv_pager_t *p, int layer, int start_pos, int end_pos, int write_touch) {
    if (!p || !p->pager || !p->page_table) return 0;
    if (layer < 0 || layer >= p->num_layers) return -1;
    if (start_pos > end_pos) return 0;
    if (end_pos < 0 || start_pos >= p->max_seq_len) return 0;

    int clamped_start = (start_pos < 0) ? 0 : start_pos;
    int clamped_end = (end_pos >= p->max_seq_len) ? (p->max_seq_len - 1) : end_pos;
    if (clamped_start > clamped_end) return 0;

    if ((clamped_end + 1) > p->latest_seq_pos_exclusive) {
        p->latest_seq_pos_exclusive = clamped_end + 1;
    }

    int first_page_pos = (clamped_start / (int)p->page_tokens) * (int)p->page_tokens;
    int last_page_pos = (clamped_end / (int)p->page_tokens) * (int)p->page_tokens;

    for (int pos = first_page_pos; pos <= last_page_pos; pos += (int)p->page_tokens) {
        kv_page_t *page = bind_page_for_pos(p, layer, pos);
        if (!page) {
            LOG_WARN("VK KV pager could not bind page (layer=%d pos=%d)", layer, pos);
            continue;
        }
        if (write_touch) page->dirty = 1u;
    }

    if (write_touch) p->write_touches++;
    else p->read_touches++;
    return 0;
}

int vk_kv_pager_config_from_env(vk_kv_pager_config_t *cfg,
                                int num_layers,
                                int max_seq_len) {
    if (!cfg) return -1;
    memset(cfg, 0, sizeof(*cfg));

    cfg->enabled = (uint8_t)read_env_u32("SAPPHIRE_VK_KV_PAGING", 0u);
    if (!cfg->enabled) return 0;

    kv_pager_config_t base;
    if (kv_pager_config_from_env(&base) != 0) return -1;

    cfg->page_tokens = base.page_tokens;
    cfg->max_pages = base.max_pages;
    if (cfg->max_pages == 4096u) {
        uint32_t computed = compute_default_max_pages(num_layers, max_seq_len, cfg->page_tokens);
        if (computed > 0u) cfg->max_pages = computed;
    }
    return 0;
}

vk_kv_pager_t *vk_kv_pager_create(const vk_kv_pager_config_t *cfg,
                                  int num_layers,
                                  int max_seq_len,
                                  unsigned long long layer_types_mask,
                                  int sliding_window) {
    if (!cfg || !cfg->enabled) return NULL;
    if (num_layers <= 0 || max_seq_len <= 0 || cfg->page_tokens == 0u || cfg->max_pages == 0u) return NULL;

    kv_pager_config_t base;
    if (kv_pager_config_from_env(&base) != 0) return NULL;
    base.page_tokens = cfg->page_tokens;
    base.max_pages = cfg->max_pages;

    vk_kv_pager_t *p = (vk_kv_pager_t *)calloc(1, sizeof(vk_kv_pager_t));
    if (!p) return NULL;

    p->pager = kv_pager_create(&base);
    if (!p->pager) {
        free(p);
        return NULL;
    }

    p->num_layers = num_layers;
    p->max_seq_len = max_seq_len;
    p->sliding_window = sliding_window;
    p->latest_seq_pos_exclusive = 0;
    p->layer_types_mask = layer_types_mask;
    p->page_tokens = kv_pager_page_tokens(p->pager);
    p->pages_per_layer = (uint32_t)(((uint32_t)max_seq_len + p->page_tokens - 1u) / p->page_tokens);
    p->page_table_slots = p->pages_per_layer * (uint32_t)num_layers;
    p->page_table = (kv_page_t **)calloc((size_t)p->page_table_slots, sizeof(kv_page_t *));
    p->shadow_valid = (uint8_t *)calloc((size_t)p->page_table_slots, sizeof(uint8_t));
    p->pending_promote = (uint8_t *)calloc((size_t)p->page_table_slots, sizeof(uint8_t));
    p->pending_evict = (uint8_t *)calloc((size_t)p->page_table_slots, sizeof(uint8_t));
    p->pending_capacity = p->page_table_slots * 2u;
    p->pending_ops = (vk_kv_transfer_op_t *)calloc((size_t)p->pending_capacity, sizeof(vk_kv_transfer_op_t));
    if (!p->page_table || !p->shadow_valid || !p->pending_promote || !p->pending_evict || !p->pending_ops) {
        kv_pager_destroy(p->pager);
        free(p->page_table);
        free(p->shadow_valid);
        free(p->pending_promote);
        free(p->pending_evict);
        free(p->pending_ops);
        free(p);
        return NULL;
    }

    LOG_INFO("VK KV pager enabled: page_tokens=%u max_pages=%u layers=%d max_seq=%d",
             p->page_tokens,
             cfg->max_pages,
             num_layers,
             max_seq_len);
    return p;
}

void vk_kv_pager_destroy(vk_kv_pager_t *pager) {
    if (!pager) return;
    if (pager->pager) kv_pager_destroy(pager->pager);
    free(pager->page_table);
    free(pager->shadow_valid);
    free(pager->pending_promote);
    free(pager->pending_evict);
    free(pager->pending_ops);
    free(pager);
}

void vk_kv_pager_reset(vk_kv_pager_t *pager) {
    if (!pager || !pager->pager) return;
    kv_pager_reset(pager->pager);
    if (pager->page_table && pager->page_table_slots > 0u) {
        memset(pager->page_table, 0, sizeof(kv_page_t *) * pager->page_table_slots);
    }
    if (pager->shadow_valid && pager->page_table_slots > 0u) {
        memset(pager->shadow_valid, 0, sizeof(uint8_t) * pager->page_table_slots);
    }
    if (pager->pending_promote && pager->page_table_slots > 0u) {
        memset(pager->pending_promote, 0, sizeof(uint8_t) * pager->page_table_slots);
    }
    if (pager->pending_evict && pager->page_table_slots > 0u) {
        memset(pager->pending_evict, 0, sizeof(uint8_t) * pager->page_table_slots);
    }
    pager->pending_count = 0u;
    pager->page_hits = 0u;
    pager->page_misses = 0u;
    pager->page_binds = 0u;
    pager->page_evictions = 0u;
    pager->write_touches = 0u;
    pager->read_touches = 0u;
    pager->latest_seq_pos_exclusive = 0;
}

int vk_kv_pager_touch_read_range(vk_kv_pager_t *pager,
                                 int layer,
                                 int start_pos,
                                 int end_pos) {
    return touch_range(pager, layer, start_pos, end_pos, 0);
}

int vk_kv_pager_touch_write_range(vk_kv_pager_t *pager,
                                  int layer,
                                  int start_pos,
                                  int end_pos) {
    return touch_range(pager, layer, start_pos, end_pos, 1);
}

int vk_kv_pager_get_stats(const vk_kv_pager_t *pager,
                          vk_kv_pager_stats_t *out_stats) {
    if (!pager || !out_stats) return -1;
    out_stats->page_hits = pager->page_hits;
    out_stats->page_misses = pager->page_misses;
    out_stats->page_binds = pager->page_binds;
    out_stats->page_evictions = pager->page_evictions;
    out_stats->write_touches = pager->write_touches;
    out_stats->read_touches = pager->read_touches;
    return 0;
}

uint32_t vk_kv_pager_page_tokens(const vk_kv_pager_t *pager) {
    if (!pager) return 0u;
    return pager->page_tokens;
}

uint32_t vk_kv_pager_table_slots(const vk_kv_pager_t *pager) {
    if (!pager) return 0u;
    return pager->page_table_slots;
}

uint32_t vk_kv_pager_drain_transfer_ops(vk_kv_pager_t *pager,
                                        vk_kv_transfer_op_t *out_ops,
                                        uint32_t max_ops) {
    if (!pager || !out_ops || max_ops == 0u || pager->pending_count == 0u) return 0u;

    uint32_t n = (pager->pending_count < max_ops) ? pager->pending_count : max_ops;
    memcpy(out_ops, pager->pending_ops, sizeof(vk_kv_transfer_op_t) * n);

    for (uint32_t i = 0; i < n; ++i) {
        clear_pending_for_slot(pager, out_ops[i].table_slot);
    }

    if (n < pager->pending_count) {
        memmove(pager->pending_ops,
                pager->pending_ops + n,
                sizeof(vk_kv_transfer_op_t) * (pager->pending_count - n));
    }
    pager->pending_count -= n;
    return n;
}