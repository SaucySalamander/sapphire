#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/kv_cache_state.h"
#include "../include/log.h"

#define KV_STATE_MAGIC "SPKVv1\0"
#define KV_STATE_VERSION 1u
#define KV_STATE_TRANSCRIPT_MAGIC "SPTRv1\0"
#define KV_STATE_TRANSCRIPT_VERSION 1u

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t token_count;
    uint64_t token_bytes;
} kv_state_transcript_footer_t;

struct kv_state_writer_t {
    FILE *fp;
    kv_state_file_header_t header;
    uint64_t pages_written;
};

struct kv_state_reader_t {
    FILE *fp;
    kv_state_file_header_t header;
    uint64_t pages_read;
};

static int write_exact(FILE *fp, const void *ptr, size_t bytes) {
    if (!fp || !ptr) return -1;
    size_t written = fwrite(ptr, 1, bytes, fp);
    return (written == bytes) ? 0 : -1;
}

static int read_exact(FILE *fp, void *ptr, size_t bytes) {
    if (!fp || !ptr) return -1;
    size_t read_n = fread(ptr, 1, bytes, fp);
    return (read_n == bytes) ? 0 : -1;
}

int kv_state_writer_open(const char *path,
                         const kv_state_file_header_t *header,
                         kv_state_writer_t **out_writer) {
    if (!path || !header || !out_writer) return -1;

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        LOG_ERROR("KV snapshot open for write failed: %s (%s)", path, strerror(errno));
        return -1;
    }

    kv_state_writer_t *writer = (kv_state_writer_t *)calloc(1, sizeof(kv_state_writer_t));
    if (!writer) {
        fclose(fp);
        return -1;
    }

    writer->fp = fp;
    writer->header = *header;

    if (write_exact(fp, &writer->header, sizeof(writer->header)) != 0) {
        LOG_ERROR("KV snapshot header write failed: %s", path);
        fclose(fp);
        free(writer);
        return -1;
    }

    *out_writer = writer;
    return 0;
}

int kv_state_writer_write_page(kv_state_writer_t *writer,
                               const kv_state_page_record_t *record,
                               const float *k_page,
                               const float *v_page,
                               size_t page_float_count) {
    if (!writer || !record || !k_page || !v_page || page_float_count == 0u) return -1;
    size_t page_bytes = page_float_count * sizeof(float);

    if (write_exact(writer->fp, record, sizeof(*record)) != 0) return -1;
    if (write_exact(writer->fp, k_page, page_bytes) != 0) return -1;
    if (write_exact(writer->fp, v_page, page_bytes) != 0) return -1;

    writer->pages_written++;
    return 0;
}

int kv_state_writer_close(kv_state_writer_t *writer) {
    if (!writer) return -1;
    int rc = 0;
    if (writer->pages_written != writer->header.page_count) {
        LOG_WARN("KV snapshot page count mismatch: expected=%llu wrote=%llu",
                 (unsigned long long)writer->header.page_count,
                 (unsigned long long)writer->pages_written);
    }
    if (fclose(writer->fp) != 0) rc = -1;
    free(writer);
    return rc;
}

int kv_state_reader_open(const char *path,
                         kv_state_file_header_t *out_header,
                         kv_state_reader_t **out_reader) {
    if (!path || !out_header || !out_reader) return -1;

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        LOG_ERROR("KV snapshot open for read failed: %s (%s)", path, strerror(errno));
        return -1;
    }

    kv_state_reader_t *reader = (kv_state_reader_t *)calloc(1, sizeof(kv_state_reader_t));
    if (!reader) {
        fclose(fp);
        return -1;
    }

    reader->fp = fp;
    if (read_exact(fp, &reader->header, sizeof(reader->header)) != 0) {
        LOG_ERROR("KV snapshot header read failed: %s", path);
        fclose(fp);
        free(reader);
        return -1;
    }

    if (memcmp(reader->header.magic, KV_STATE_MAGIC, sizeof(reader->header.magic)) != 0) {
        LOG_ERROR("KV snapshot magic mismatch: %s", path);
        fclose(fp);
        free(reader);
        return -1;
    }

    if (reader->header.version != KV_STATE_VERSION) {
        LOG_ERROR("KV snapshot version mismatch: got=%u expected=%u", reader->header.version, KV_STATE_VERSION);
        fclose(fp);
        free(reader);
        return -1;
    }

    if (reader->header.header_size != sizeof(kv_state_file_header_t)) {
        LOG_ERROR("KV snapshot header size mismatch: got=%u expected=%zu",
                  reader->header.header_size,
                  sizeof(kv_state_file_header_t));
        fclose(fp);
        free(reader);
        return -1;
    }

    *out_header = reader->header;
    *out_reader = reader;
    return 0;
}

int kv_state_reader_read_page(kv_state_reader_t *reader,
                              kv_state_page_record_t *out_record,
                              float *out_k_page,
                              float *out_v_page,
                              size_t page_float_count) {
    if (!reader || !out_record || !out_k_page || !out_v_page || page_float_count == 0u) return -1;
    if (reader->pages_read >= reader->header.page_count) return 0;

    size_t page_bytes = page_float_count * sizeof(float);
    if (read_exact(reader->fp, out_record, sizeof(*out_record)) != 0) return -1;
    if (read_exact(reader->fp, out_k_page, page_bytes) != 0) return -1;
    if (read_exact(reader->fp, out_v_page, page_bytes) != 0) return -1;

    reader->pages_read++;
    return 1;
}

int kv_state_reader_close(kv_state_reader_t *reader) {
    if (!reader) return -1;
    int rc = 0;
    if (fclose(reader->fp) != 0) rc = -1;
    free(reader);
    return rc;
}

int kv_state_append_transcript(const char *path,
                               const int *tokens,
                               uint32_t token_count) {
    if (!path) return -1;
    if (token_count > 0u && !tokens) return -1;

    FILE *fp = fopen(path, "ab");
    if (!fp) {
        LOG_ERROR("KV snapshot open for transcript append failed: %s (%s)", path, strerror(errno));
        return -1;
    }

    int32_t *encoded = NULL;
    if (token_count > 0u) {
        encoded = (int32_t *)malloc((size_t)token_count * sizeof(int32_t));
        if (!encoded) {
            fclose(fp);
            return -1;
        }

        for (uint32_t i = 0; i < token_count; ++i) {
            encoded[i] = (int32_t)tokens[i];
        }

        if (write_exact(fp, encoded, (size_t)token_count * sizeof(int32_t)) != 0) {
            free(encoded);
            fclose(fp);
            LOG_ERROR("KV snapshot transcript payload write failed: %s", path);
            return -1;
        }
    }

    kv_state_transcript_footer_t footer;
    memset(&footer, 0, sizeof(footer));
    memcpy(footer.magic, KV_STATE_TRANSCRIPT_MAGIC, sizeof(footer.magic));
    footer.version = KV_STATE_TRANSCRIPT_VERSION;
    footer.token_count = token_count;
    footer.token_bytes = (uint64_t)token_count * sizeof(int32_t);

    if (write_exact(fp, &footer, sizeof(footer)) != 0) {
        if (encoded) free(encoded);
        fclose(fp);
        LOG_ERROR("KV snapshot transcript footer write failed: %s", path);
        return -1;
    }

    if (encoded) free(encoded);
    if (fclose(fp) != 0) {
        LOG_ERROR("KV snapshot transcript append close failed: %s", path);
        return -1;
    }

    return 0;
}

int kv_state_read_transcript(const char *path,
                             int *out_tokens,
                             uint32_t max_tokens,
                             uint32_t *out_token_count) {
    if (!path || !out_token_count) return -1;
    *out_token_count = 0u;

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        LOG_ERROR("KV snapshot open for transcript read failed: %s (%s)", path, strerror(errno));
        return -1;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return -1;
    }

    long file_size = ftell(fp);
    if (file_size < 0) {
        fclose(fp);
        return -1;
    }

    if ((uint64_t)file_size < sizeof(kv_state_transcript_footer_t)) {
        fclose(fp);
        return 1;
    }

    if (fseek(fp, -(long)sizeof(kv_state_transcript_footer_t), SEEK_END) != 0) {
        fclose(fp);
        return -1;
    }

    kv_state_transcript_footer_t footer;
    if (read_exact(fp, &footer, sizeof(footer)) != 0) {
        fclose(fp);
        return -1;
    }

    if (memcmp(footer.magic, KV_STATE_TRANSCRIPT_MAGIC, sizeof(footer.magic)) != 0) {
        fclose(fp);
        return 1;
    }

    if (footer.version != KV_STATE_TRANSCRIPT_VERSION) {
        LOG_ERROR("KV snapshot transcript version mismatch: got=%u expected=%u",
                  footer.version,
                  KV_STATE_TRANSCRIPT_VERSION);
        fclose(fp);
        return -1;
    }

    uint64_t expected_bytes = (uint64_t)footer.token_count * sizeof(int32_t);
    if (footer.token_bytes != expected_bytes) {
        LOG_ERROR("KV snapshot transcript byte size mismatch: token_count=%u token_bytes=%llu",
                  footer.token_count,
                  (unsigned long long)footer.token_bytes);
        fclose(fp);
        return -1;
    }

    if (footer.token_count > max_tokens) {
        LOG_ERROR("KV snapshot transcript too large for context: tokens=%u capacity=%u",
                  footer.token_count,
                  max_tokens);
        fclose(fp);
        return -1;
    }

    uint64_t footer_and_payload = footer.token_bytes + sizeof(kv_state_transcript_footer_t);
    if (footer_and_payload > (uint64_t)file_size) {
        LOG_ERROR("KV snapshot transcript footer points outside file: %s", path);
        fclose(fp);
        return -1;
    }

    if (footer.token_count == 0u) {
        fclose(fp);
        return 0;
    }

    if (!out_tokens) {
        fclose(fp);
        return -1;
    }

    long payload_offset = file_size - (long)footer_and_payload;
    if (fseek(fp, payload_offset, SEEK_SET) != 0) {
        fclose(fp);
        return -1;
    }

    int32_t *encoded = (int32_t *)malloc((size_t)footer.token_count * sizeof(int32_t));
    if (!encoded) {
        fclose(fp);
        return -1;
    }

    if (read_exact(fp, encoded, (size_t)footer.token_count * sizeof(int32_t)) != 0) {
        free(encoded);
        fclose(fp);
        return -1;
    }

    for (uint32_t i = 0; i < footer.token_count; ++i) {
        out_tokens[i] = (int)encoded[i];
    }
    *out_token_count = footer.token_count;

    free(encoded);
    fclose(fp);
    return 0;
}
