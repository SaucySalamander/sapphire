/*
 * @file ternary_hessian_sidecar.c
 * @brief Mmap-backed diagonal curvature sidecar for ternary calibration.
 */

#include "ternary_hessian_sidecar.h"

#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "log.h"
#include "ternary_io.h"

struct ternary_hessian_sidecar_t {
    int fd;
    void *mmap_ptr;
    size_t mmap_size;
    ternary_hessian_sidecar_header_t header;
    const ternary_hessian_sidecar_entry_t *manifest;
    const float *data_section;
    uint32_t crc32;
};

static int sidecar_validate_tensor_name(const char *tensor_name)
{
    if (!tensor_name || tensor_name[0] == '\0') {
        LOG_ERROR("hessian sidecar: tensor name is NULL or empty");
        return -1;
    }
    if (strlen(tensor_name) >= TERNARY_HESSIAN_SIDECAR_TENSOR_NAME_MAX) {
        LOG_ERROR("hessian sidecar: tensor name too long: %s", tensor_name);
        return -1;
    }
    return 0;
}

static int sidecar_open_mmap(const char *path, int *out_fd, void **out_ptr, size_t *out_size)
{
    struct stat st;
    int fd = -1;
    void *mmap_ptr = NULL;
    size_t mmap_size = 0u;

    if (!path || !out_fd || !out_ptr || !out_size) {
        return -1;
    }

    fd = open(path, O_RDONLY);
    if (fd < 0) {
        LOG_ERROR("hessian sidecar: open %s failed: %s", path, strerror(errno));
        return -1;
    }
    if (fstat(fd, &st) != 0) {
        LOG_ERROR("hessian sidecar: fstat %s failed: %s", path, strerror(errno));
        close(fd);
        return -1;
    }
    if (!S_ISREG(st.st_mode) || st.st_size <= 0) {
        LOG_ERROR("hessian sidecar: invalid file type or size: %s", path);
        close(fd);
        return -1;
    }

    mmap_size = (size_t)st.st_size;
    mmap_ptr = mmap(NULL, mmap_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mmap_ptr == MAP_FAILED) {
        LOG_ERROR("hessian sidecar: mmap failed for %s: %s", path, strerror(errno));
        close(fd);
        return -1;
    }

    *out_fd = fd;
    *out_ptr = mmap_ptr;
    *out_size = mmap_size;
    return 0;
}

static uint32_t sidecar_compute_crc32(const void *mmap_ptr, size_t mmap_size)
{
    uint32_t crc32 = 0u;
    size_t header_prefix = offsetof(ternary_hessian_sidecar_header_t, crc32);
    size_t header_size = sizeof(ternary_hessian_sidecar_header_t);

    if (!mmap_ptr || mmap_size < header_size) {
        return 0u;
    }

    crc32 = io_crc32_update(crc32, mmap_ptr, header_prefix);
    if (mmap_size > header_size) {
        crc32 = io_crc32_update(crc32,
                                (const uint8_t *)mmap_ptr + header_size,
                                mmap_size - header_size);
    }
    return crc32;
}

static int sidecar_resolve_primary_index(const ternary_hessian_sidecar_t *sidecar,
                                         uint32_t entry_idx,
                                         uint32_t *out_primary_idx)
{
    uint32_t cursor = entry_idx;

    if (!sidecar || !out_primary_idx || entry_idx >= sidecar->header.entry_count) {
        return -1;
    }

    for (uint32_t depth = 0; depth < sidecar->header.entry_count; ++depth) {
        const ternary_hessian_sidecar_entry_t *entry = sidecar->manifest + cursor;

        if (entry->alias_of_entry == TERNARY_HESSIAN_SIDECAR_NO_ALIAS) {
            *out_primary_idx = cursor;
            return 0;
        }
        if (entry->alias_of_entry >= sidecar->header.entry_count) {
            return -1;
        }
        cursor = entry->alias_of_entry;
    }

    return -1;
}

static const ternary_hessian_sidecar_entry_t *sidecar_lookup_exact(const ternary_hessian_sidecar_t *sidecar,
                                                                   const char *tensor_name,
                                                                   int *out_entry_index)
{
    if (out_entry_index) {
        *out_entry_index = -1;
    }
    if (!sidecar || sidecar->header.entry_count == 0u || sidecar->manifest == NULL) {
        return NULL;
    }
    if (sidecar_validate_tensor_name(tensor_name) != 0) {
        return NULL;
    }

    for (uint32_t entry_idx = 0; entry_idx < sidecar->header.entry_count; ++entry_idx) {
        const ternary_hessian_sidecar_entry_t *entry = sidecar->manifest + entry_idx;

        if (strcmp(entry->tensor_name, tensor_name) == 0) {
            if (out_entry_index) {
                *out_entry_index = (int)entry_idx;
            }
            return entry;
        }
    }

    return NULL;
}

static int sidecar_validate_manifest(const ternary_hessian_sidecar_t *sidecar)
{
    uint64_t expected_data_offset = 0u;

    if (!sidecar || !sidecar->manifest) {
        return -1;
    }

    for (uint32_t entry_idx = 0; entry_idx < sidecar->header.entry_count; ++entry_idx) {
        const ternary_hessian_sidecar_entry_t *entry = sidecar->manifest + entry_idx;
        uint32_t primary_idx = UINT32_MAX;
        const ternary_hessian_sidecar_entry_t *primary = NULL;

        if (memchr(entry->tensor_name, '\0', sizeof(entry->tensor_name)) == NULL || entry->tensor_name[0] == '\0') {
            LOG_ERROR("hessian sidecar: entry %u has an invalid tensor name", entry_idx);
            return -1;
        }
        if (entry->vector_dim == 0u || entry->sample_count != sidecar->header.sample_count) {
            LOG_ERROR("hessian sidecar: invalid shape metadata for %s", entry->tensor_name);
            return -1;
        }
        if (sidecar_resolve_primary_index(sidecar, entry_idx, &primary_idx) != 0) {
            LOG_ERROR("hessian sidecar: failed to resolve alias chain for %s", entry->tensor_name);
            return -1;
        }

        primary = sidecar->manifest + primary_idx;
        if (primary_idx == entry_idx) {
            uint64_t expected_bytes = (uint64_t)entry->vector_dim * sizeof(float);

            if (entry->alias_of_entry != TERNARY_HESSIAN_SIDECAR_NO_ALIAS) {
                LOG_ERROR("hessian sidecar: primary entry %s must not alias another entry", entry->tensor_name);
                return -1;
            }
            if ((entry->data_offset % sizeof(float)) != 0u ||
                (entry->data_bytes % sizeof(float)) != 0u) {
                LOG_ERROR("hessian sidecar: unaligned float payload for %s", entry->tensor_name);
                return -1;
            }
            if (entry->data_offset != expected_data_offset) {
                LOG_ERROR("hessian sidecar: unexpected data offset for %s (have=%llu expected=%llu)",
                          entry->tensor_name,
                          (unsigned long long)entry->data_offset,
                          (unsigned long long)expected_data_offset);
                return -1;
            }
            if (entry->data_bytes != expected_bytes) {
                LOG_ERROR("hessian sidecar: data byte count mismatch for %s", entry->tensor_name);
                return -1;
            }
            if (entry->data_offset + entry->data_bytes > sidecar->header.data_section_size) {
                LOG_ERROR("hessian sidecar: data range out of bounds for %s", entry->tensor_name);
                return -1;
            }
            expected_data_offset += entry->data_bytes;
            continue;
        }

        if (entry->alias_of_entry != primary_idx) {
            LOG_ERROR("hessian sidecar: alias mismatch for %s", entry->tensor_name);
            return -1;
        }
        if (entry->vector_dim != primary->vector_dim ||
            entry->sample_count != primary->sample_count ||
            entry->data_offset != primary->data_offset ||
            entry->data_bytes != primary->data_bytes ||
            entry->layer_type != primary->layer_type ||
            entry->mean != primary->mean ||
            entry->max != primary->max) {
            LOG_ERROR("hessian sidecar: alias metadata does not match primary entry for %s", entry->tensor_name);
            return -1;
        }
    }

    if (expected_data_offset != sidecar->header.data_section_size) {
        LOG_ERROR("hessian sidecar: data section size mismatch (expected=%llu actual=%llu)",
                  (unsigned long long)expected_data_offset,
                  (unsigned long long)sidecar->header.data_section_size);
        return -1;
    }

    return 0;
}

ternary_hessian_sidecar_t *ternary_hessian_sidecar_open(const char *sidecar_path)
{
    ternary_hessian_sidecar_t *sidecar = NULL;
    const ternary_hessian_sidecar_header_t *header = NULL;
    size_t manifest_bytes = 0u;
    uint32_t computed_crc32 = 0u;
    uint64_t expected_data_offset = 0u;
    size_t data_section_offset = 0u;

    if (!sidecar_path || sidecar_path[0] == '\0') {
        LOG_ERROR("hessian sidecar: path is empty");
        return NULL;
    }

    sidecar = (ternary_hessian_sidecar_t *)calloc(1u, sizeof(*sidecar));
    if (!sidecar) {
        LOG_ERROR("hessian sidecar: allocation failed");
        return NULL;
    }

    if (sidecar_open_mmap(sidecar_path, &sidecar->fd, &sidecar->mmap_ptr, &sidecar->mmap_size) != 0) {
        free(sidecar);
        return NULL;
    }

    if (sidecar->mmap_size < sizeof(ternary_hessian_sidecar_header_t)) {
        LOG_ERROR("hessian sidecar: file too small: %s", sidecar_path);
        goto fail;
    }

    header = (const ternary_hessian_sidecar_header_t *)sidecar->mmap_ptr;
    memcpy(&sidecar->header, header, sizeof(sidecar->header));

    if (sidecar->header.magic != TERNARY_HESSIAN_SIDECAR_MAGIC ||
        sidecar->header.version != TERNARY_HESSIAN_SIDECAR_VERSION) {
        LOG_ERROR("hessian sidecar: unsupported file version or magic in %s", sidecar_path);
        goto fail;
    }
    if (memchr(sidecar->header.teacher_model_name,
               '\0',
               sizeof(sidecar->header.teacher_model_name)) == NULL ||
        sidecar->header.teacher_model_name[0] == '\0') {
        LOG_ERROR("hessian sidecar: teacher model name is missing in %s", sidecar_path);
        goto fail;
    }
    if (sidecar->header.reserved != 0u) {
        LOG_ERROR("hessian sidecar: reserved header field must be zero in %s", sidecar_path);
        goto fail;
    }
    if (sidecar->header.manifest_entry_size != sizeof(ternary_hessian_sidecar_entry_t)) {
        LOG_ERROR("hessian sidecar: manifest entry size mismatch in %s", sidecar_path);
        goto fail;
    }
    if (sidecar->header.entry_count == 0u || sidecar->header.sample_count == 0u) {
        LOG_ERROR("hessian sidecar: empty manifest in %s", sidecar_path);
        goto fail;
    }

    computed_crc32 = sidecar_compute_crc32(sidecar->mmap_ptr, sidecar->mmap_size);
    if (computed_crc32 != sidecar->header.crc32) {
        LOG_ERROR("hessian sidecar: checksum mismatch in %s (expected=%08x actual=%08x)",
                  sidecar_path,
                  sidecar->header.crc32,
                  computed_crc32);
        goto fail;
    }

    if (sidecar->header.entry_count > SIZE_MAX / sizeof(ternary_hessian_sidecar_entry_t)) {
        LOG_ERROR("hessian sidecar: manifest too large in %s", sidecar_path);
        goto fail;
    }

    manifest_bytes = (size_t)sidecar->header.entry_count * sizeof(ternary_hessian_sidecar_entry_t);
    expected_data_offset = (uint64_t)sizeof(ternary_hessian_sidecar_header_t) + (uint64_t)manifest_bytes;
    data_section_offset = (size_t)sidecar->header.data_section_offset;
    if (sidecar->header.data_section_offset != expected_data_offset) {
        LOG_ERROR("hessian sidecar: data section offset mismatch in %s", sidecar_path);
        goto fail;
    }
    if ((data_section_offset % sizeof(float)) != 0u) {
        LOG_ERROR("hessian sidecar: data section alignment mismatch in %s", sidecar_path);
        goto fail;
    }
    if (data_section_offset > sidecar->mmap_size ||
        sidecar->header.data_section_size > (uint64_t)(sidecar->mmap_size - data_section_offset)) {
        LOG_ERROR("hessian sidecar: data section out of bounds in %s", sidecar_path);
        goto fail;
    }

    sidecar->manifest = (const ternary_hessian_sidecar_entry_t *)((const uint8_t *)sidecar->mmap_ptr + sizeof(ternary_hessian_sidecar_header_t));
    sidecar->data_section = (const float *)(const void *)((const uint8_t *)sidecar->mmap_ptr + data_section_offset);
    if (sidecar_validate_manifest(sidecar) != 0) {
        goto fail;
    }

    sidecar->crc32 = computed_crc32;
    LOG_INFO("Loaded Hessian sidecar: %s (teacher=%s entries=%u samples=%u crc32=%08x)",
             sidecar_path,
             sidecar->header.teacher_model_name,
             sidecar->header.entry_count,
             sidecar->header.sample_count,
             sidecar->crc32);
    return sidecar;

fail:
    ternary_hessian_sidecar_close(sidecar);
    return NULL;
}

const ternary_hessian_sidecar_header_t *ternary_hessian_sidecar_header(const ternary_hessian_sidecar_t *sidecar)
{
    return sidecar ? &sidecar->header : NULL;
}

uint32_t ternary_hessian_sidecar_crc32(const ternary_hessian_sidecar_t *sidecar)
{
    return sidecar ? sidecar->crc32 : 0u;
}

const char *ternary_hessian_sidecar_teacher_model_name(const ternary_hessian_sidecar_t *sidecar)
{
    return sidecar ? sidecar->header.teacher_model_name : NULL;
}

uint32_t ternary_hessian_sidecar_tape_crc32(const ternary_hessian_sidecar_t *sidecar)
{
    return sidecar ? sidecar->header.tape_crc32 : 0u;
}

int ternary_hessian_sidecar_entry_index(const ternary_hessian_sidecar_t *sidecar,
                                       const char *tensor_name)
{
    int entry_idx = -1;

    if (!sidecar || sidecar_validate_tensor_name(tensor_name) != 0) {
        return -1;
    }

    if (sidecar_lookup_exact(sidecar, tensor_name, &entry_idx) == NULL) {
        return -1;
    }

    return entry_idx;
}

const ternary_hessian_sidecar_entry_t *ternary_hessian_sidecar_entry(const ternary_hessian_sidecar_t *sidecar,
                                                                     const char *tensor_name)
{
    return sidecar_lookup_exact(sidecar, tensor_name, NULL);
}

uint32_t ternary_hessian_sidecar_vector_dim(const ternary_hessian_sidecar_t *sidecar,
                                           const char *tensor_name)
{
    const ternary_hessian_sidecar_entry_t *entry = sidecar_lookup_exact(sidecar, tensor_name, NULL);

    return entry ? entry->vector_dim : 0u;
}

int ternary_hessian_sidecar_sample_count(const ternary_hessian_sidecar_t *sidecar)
{
    return sidecar ? (int)sidecar->header.sample_count : -1;
}

const float *ternary_hessian_sidecar_diagonal(const ternary_hessian_sidecar_t *sidecar,
                                              const char *tensor_name)
{
    int entry_idx = -1;
    uint32_t primary_idx = UINT32_MAX;

    if (!sidecar || sidecar_validate_tensor_name(tensor_name) != 0) {
        return NULL;
    }

    if (sidecar_lookup_exact(sidecar, tensor_name, &entry_idx) == NULL) {
        return NULL;
    }
    if (entry_idx < 0 || sidecar_resolve_primary_index(sidecar, (uint32_t)entry_idx, &primary_idx) != 0) {
        return NULL;
    }

    return sidecar->data_section + (sidecar->manifest[primary_idx].data_offset / sizeof(float));
}

int ternary_hessian_sidecar_stats(const ternary_hessian_sidecar_t *sidecar,
                                  const char *tensor_name,
                                  ternary_hessian_proxy_stats_t *out_stats)
{
    const ternary_hessian_sidecar_entry_t *entry = sidecar_lookup_exact(sidecar, tensor_name, NULL);

    if (!entry || !out_stats) {
        return -1;
    }

    out_stats->mean = entry->mean;
    out_stats->max = entry->max;
    return 0;
}

void ternary_hessian_sidecar_close(ternary_hessian_sidecar_t *sidecar)
{
    if (!sidecar) {
        return;
    }

    if (sidecar->mmap_ptr && sidecar->mmap_ptr != MAP_FAILED) {
        munmap(sidecar->mmap_ptr, sidecar->mmap_size);
    }
    if (sidecar->fd >= 0) {
        close(sidecar->fd);
    }
    memset(sidecar, 0, sizeof(*sidecar));
    free(sidecar);
}