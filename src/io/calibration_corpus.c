/**
 * @file calibration_corpus.c
 * @brief Local and remote calibration corpus loading helpers for ternary conversion.
 */

#include "calibration_corpus.h"

#include "file_reader.h"
#include "log.h"

#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef SAPPHIRE_HAVE_LIBCURL
#include <curl/curl.h>
#endif

#define CALIBRATION_CORPUS_DEFAULT_SAMPLES 64
#define CALIBRATION_CORPUS_INITIAL_CAPACITY 16
#define CALIBRATION_CORPUS_MAX_REMOTE_BYTES (8u * 1024u * 1024u)
#define CALIBRATION_CORPUS_MAX_MANIFEST_FIELDS 3

typedef struct {
    char *source;
    float weight;
    int quota;
} calibration_manifest_entry_t;

typedef struct {
    calibration_corpus_t corpus;
    float weight;
    float credit;
    int quota;
    int next_index;
} calibration_source_state_t;

static int calibration_corpus_is_remote_source(const char *path) {
    if (!path) {
        return 0;
    }
    return (strncmp(path, "http://", 7) == 0) || (strncmp(path, "https://", 8) == 0);
}

static int calibration_corpus_is_header_label(const char *text) {
    if (!text) {
        return 0;
    }
    return strcmp(text, "source") == 0 || strcmp(text, "url") == 0;
}

static char *trim_sample_in_place(char *text) {
    char *start = text;
    char *end = NULL;

    if (!text) {
        return NULL;
    }

    while (*start != '\0' && isspace((unsigned char)*start)) {
        start++;
    }
    if (*start == '\0') {
        return start;
    }

    end = start + strlen(start);
    while (end > start && isspace((unsigned char)end[-1])) {
        end--;
    }
    *end = '\0';
    return start;
}

static int calibration_corpus_load_local_text(const char *path,
                                              char **out_buffer,
                                              size_t *out_size) {
    char *raw_buffer = NULL;
    size_t raw_size = 0;
    char *owned_buffer = NULL;

    if (file_read_to_buffer(path, &raw_buffer, &raw_size) != 0) {
        return -1;
    }

    owned_buffer = (char *)malloc(raw_size + 1u);
    if (!owned_buffer) {
        LOG_ERROR("calibration_corpus_load: allocation failed");
        free(raw_buffer);
        return -1;
    }

    memcpy(owned_buffer, raw_buffer, raw_size);
    owned_buffer[raw_size] = '\0';
    free(raw_buffer);

    *out_buffer = owned_buffer;
    *out_size = raw_size;
    return 0;
}

#ifdef SAPPHIRE_HAVE_LIBCURL
typedef struct {
    char *buffer;
    size_t size;
} calibration_remote_buffer_t;

static size_t calibration_corpus_curl_write(void *contents,
                                            size_t size,
                                            size_t nmemb,
                                            void *userdata) {
    calibration_remote_buffer_t *download = (calibration_remote_buffer_t *)userdata;
    char *new_buffer = NULL;
    size_t chunk_size = size * nmemb;
    size_t new_size = 0;

    if (!download || !contents) {
        return 0;
    }
    if (chunk_size == 0) {
        return 0;
    }

    new_size = download->size + chunk_size;
    if (new_size > CALIBRATION_CORPUS_MAX_REMOTE_BYTES) {
        LOG_ERROR("calibration corpus download exceeded %u bytes", (unsigned int)CALIBRATION_CORPUS_MAX_REMOTE_BYTES);
        return 0;
    }

    new_buffer = (char *)realloc(download->buffer, new_size + 1u);
    if (!new_buffer) {
        LOG_ERROR("calibration corpus download buffer growth failed");
        return 0;
    }

    download->buffer = new_buffer;
    memcpy(download->buffer + download->size, contents, chunk_size);
    download->size = new_size;
    download->buffer[download->size] = '\0';
    return chunk_size;
}

static int calibration_corpus_fetch_remote(const char *url,
                                           char **out_buffer,
                                           size_t *out_size) {
    calibration_remote_buffer_t download;
    CURL *curl = NULL;
    CURLcode rc = CURLE_OK;

    memset(&download, 0, sizeof(download));
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != 0) {
        LOG_ERROR("calibration_corpus_load: curl_global_init failed");
        return -1;
    }

    curl = curl_easy_init();
    if (!curl) {
        LOG_ERROR("calibration_corpus_load: curl_easy_init failed");
        curl_global_cleanup();
        return -1;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "sapphire/1.0");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, calibration_corpus_curl_write);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &download);

    rc = curl_easy_perform(curl);
    if (rc != CURLE_OK) {
        LOG_ERROR("calibration_corpus_load: failed to fetch %s: %s", url, curl_easy_strerror(rc));
        curl_easy_cleanup(curl);
        curl_global_cleanup();
        free(download.buffer);
        return -1;
    }
    if (!download.buffer || download.size == 0) {
        LOG_ERROR("calibration_corpus_load: remote corpus is empty: %s", url);
        curl_easy_cleanup(curl);
        curl_global_cleanup();
        free(download.buffer);
        return -1;
    }

    curl_easy_cleanup(curl);
    curl_global_cleanup();
    *out_buffer = download.buffer;
    *out_size = download.size;
    return 0;
}
#else
static int calibration_corpus_fetch_remote(const char *url,
                                           char **out_buffer,
                                           size_t *out_size) {
    (void)out_buffer;
    (void)out_size;
    LOG_ERROR("calibration_corpus_load: remote corpus URL requires libcurl support: %s", url ? url : "<null>");
    return -1;
}
#endif

static int calibration_corpus_load_text(const char *path,
                                        char **out_buffer,
                                        size_t *out_size) {
    if (calibration_corpus_is_remote_source(path)) {
        return calibration_corpus_fetch_remote(path, out_buffer, out_size);
    }
    return calibration_corpus_load_local_text(path, out_buffer, out_size);
}

static int calibration_corpus_parse_source_list(char *line,
                                                char **fields,
                                                int max_fields) {
    char *cursor = line;
    int count = 0;
    int use_whitespace_split = 1;

    if (!line || !fields || max_fields <= 0) {
        return 0;
    }

    for (const char *scan = line; *scan != '\0'; ++scan) {
        if (*scan == '\t' || *scan == ',') {
            use_whitespace_split = 0;
            break;
        }
    }

    while (*cursor != '\0' && count < max_fields) {
        char *start = cursor;

        if (use_whitespace_split) {
            while (*start != '\0' && isspace((unsigned char)*start)) {
                start++;
            }
            cursor = start;
            while (*cursor != '\0' && !isspace((unsigned char)*cursor)) {
                cursor++;
            }
        } else {
            while (*cursor != '\0' && *cursor != '\t' && *cursor != ',') {
                cursor++;
            }
        }
        if (*cursor != '\0') {
            *cursor++ = '\0';
        }
        fields[count] = trim_sample_in_place(start);
        if (fields[count] && fields[count][0] != '\0') {
            count++;
        }
    }

    return count;
}

static int calibration_corpus_parse_quota(const char *text,
                                          int default_quota,
                                          int *out_quota) {
    char *endptr = NULL;
    long value = 0;

    if (!out_quota) {
        return -1;
    }
    if (!text || text[0] == '\0') {
        *out_quota = default_quota;
        return 0;
    }

    value = strtol(text, &endptr, 10);
    if (endptr == text || *endptr != '\0' || value < 0 || value > INT_MAX) {
        return -1;
    }

    *out_quota = (int)value;
    return 0;
}

static int calibration_corpus_parse_weight(const char *text,
                                           float *out_weight) {
    char *endptr = NULL;
    float value = 0.0f;

    if (!out_weight) {
        return -1;
    }
    if (!text || text[0] == '\0') {
        *out_weight = 1.0f;
        return 0;
    }

    value = strtof(text, &endptr);
    if (endptr == text || *endptr != '\0' || !isfinite(value) || value <= 0.0f || value > FLT_MAX) {
        return -1;
    }

    *out_weight = value;
    return 0;
}

static int calibration_corpus_append_manifest_entry(calibration_manifest_entry_t **entries,
                                                    int *count,
                                                    int *capacity,
                                                    const calibration_manifest_entry_t *entry) {
    calibration_manifest_entry_t *new_entries = NULL;
    int new_capacity = 0;

    if (!entries || !count || !capacity || !entry) {
        return -1;
    }
    if (*count < *capacity) {
        (*entries)[(*count)++] = *entry;
        return 0;
    }

    new_capacity = (*capacity <= 0) ? CALIBRATION_CORPUS_INITIAL_CAPACITY : (*capacity * 2);
    new_entries = (calibration_manifest_entry_t *)realloc(*entries,
                                                          (size_t)new_capacity * sizeof(calibration_manifest_entry_t));
    if (!new_entries) {
        return -1;
    }

    *entries = new_entries;
    *capacity = new_capacity;
    (*entries)[(*count)++] = *entry;
    return 0;
}

static int calibration_corpus_parse_manifest(char *manifest_buffer,
                                             int max_samples,
                                             calibration_manifest_entry_t **out_entries,
                                             int *out_entry_count) {
    char *cursor = manifest_buffer;
    calibration_manifest_entry_t *entries = NULL;
    int capacity = 0;
    int count = 0;
    int line_number = 0;

    if (!manifest_buffer || !out_entries || !out_entry_count) {
        return -1;
    }

    while (*cursor != '\0') {
        calibration_manifest_entry_t entry;
        char *line_start = cursor;
        char *fields[CALIBRATION_CORPUS_MAX_MANIFEST_FIELDS] = { 0 };
        int field_count = 0;

        while (*cursor != '\0' && *cursor != '\n' && *cursor != '\r') {
            cursor++;
        }
        if (*cursor != '\0') {
            *cursor++ = '\0';
            if (*cursor == '\n' || *cursor == '\r') {
                *cursor++ = '\0';
            }
        }

        line_number++;
        line_start = trim_sample_in_place(line_start);
        if (!line_start || line_start[0] == '\0' || line_start[0] == '#') {
            continue;
        }

        field_count = calibration_corpus_parse_source_list(line_start,
                                                           fields,
                                                           CALIBRATION_CORPUS_MAX_MANIFEST_FIELDS);
        if (field_count <= 0 || !fields[0] || fields[0][0] == '\0') {
            continue;
        }
        if (line_number == 1 && calibration_corpus_is_header_label(fields[0])) {
            continue;
        }

        memset(&entry, 0, sizeof(entry));
        entry.source = fields[0];
        if (calibration_corpus_parse_weight((field_count > 1) ? fields[1] : NULL,
                                            &entry.weight) != 0) {
            LOG_ERROR("calibration_corpus_load_manifest: invalid weight on line %d", line_number);
            free(entries);
            return -1;
        }
        if (calibration_corpus_parse_quota((field_count > 2) ? fields[2] : NULL,
                                           max_samples,
                                           &entry.quota) != 0) {
            LOG_ERROR("calibration_corpus_load_manifest: invalid quota on line %d", line_number);
            free(entries);
            return -1;
        }
        if (calibration_corpus_append_manifest_entry(&entries,
                                                     &count,
                                                     &capacity,
                                                     &entry) != 0) {
            LOG_ERROR("calibration_corpus_load_manifest: manifest entry allocation failed");
            free(entries);
            return -1;
        }
    }

    if (count <= 0) {
        LOG_ERROR("calibration_corpus_load_manifest: no manifest entries found");
        free(entries);
        return -1;
    }

    *out_entries = entries;
    *out_entry_count = count;
    return 0;
}

static int calibration_corpus_parse_samples(char *owned_buffer,
                                            const char *source,
                                            int max_samples,
                                            calibration_corpus_t *out_corpus) {
    char *cursor = NULL;
    char **samples = NULL;
    int capacity = 0;
    int count = 0;

    capacity = (max_samples < CALIBRATION_CORPUS_INITIAL_CAPACITY)
        ? max_samples
        : CALIBRATION_CORPUS_INITIAL_CAPACITY;
    if (capacity <= 0) {
        capacity = CALIBRATION_CORPUS_INITIAL_CAPACITY;
    }

    samples = (char **)malloc((size_t)capacity * sizeof(char *));
    if (!samples) {
        LOG_ERROR("calibration_corpus_load: sample pointer allocation failed");
        free(owned_buffer);
        return -1;
    }

    cursor = owned_buffer;
    while (*cursor != '\0' && count < max_samples) {
        char *line_start = cursor;
        char *sample = NULL;

        while (*cursor != '\0' && *cursor != '\n' && *cursor != '\r') {
            cursor++;
        }
        if (*cursor != '\0') {
            *cursor++ = '\0';
            if (*cursor == '\n' || *cursor == '\r') {
                *cursor++ = '\0';
            }
        }

        sample = trim_sample_in_place(line_start);
        if (!sample || sample[0] == '\0' || sample[0] == '#') {
            continue;
        }

        if (count == capacity) {
            int new_capacity = capacity * 2;
            char **new_samples = NULL;

            if (new_capacity > max_samples) {
                new_capacity = max_samples;
            }
            new_samples = (char **)realloc(samples, (size_t)new_capacity * sizeof(char *));
            if (!new_samples) {
                LOG_ERROR("calibration_corpus_load: sample pointer growth failed");
                free(samples);
                free(owned_buffer);
                return -1;
            }
            samples = new_samples;
            capacity = new_capacity;
        }

        samples[count++] = sample;
    }

    if (count <= 0) {
        LOG_ERROR("calibration_corpus_load: no usable samples found in %s", source);
        free(samples);
        free(owned_buffer);
        return -1;
    }

    out_corpus->buffer = owned_buffer;
    out_corpus->samples = samples;
    out_corpus->sample_count = count;
    LOG_INFO("Loaded calibration corpus: %s (%d samples)%s",
             source,
             count,
             calibration_corpus_is_remote_source(source) ? " [remote]" : "");
    return 0;
}

static int calibration_corpus_load_source(const calibration_manifest_entry_t *entry,
                                          int max_samples,
                                          calibration_source_state_t *out_state) {
    int source_limit = max_samples;

    if (!entry || !out_state) {
        return -1;
    }

    memset(out_state, 0, sizeof(*out_state));
    if (entry->quota > 0 && entry->quota < source_limit) {
        source_limit = entry->quota;
    }
    if (source_limit <= 0) {
        source_limit = max_samples;
    }
    if (calibration_corpus_load(entry->source, source_limit, &out_state->corpus) != 0) {
        return -1;
    }

    out_state->weight = entry->weight;
    out_state->credit = 0.0f;
    out_state->quota = (entry->quota > 0 && entry->quota < out_state->corpus.sample_count)
        ? entry->quota
        : out_state->corpus.sample_count;
    out_state->next_index = 0;
    return 0;
}

static int calibration_corpus_select_source(calibration_source_state_t *states,
                                            int state_count) {
    float best_credit = -FLT_MAX;
    int best_index = -1;

    for (int i = 0; i < state_count; ++i) {
        int remaining = states[i].quota - states[i].next_index;
        if (remaining <= 0) {
            continue;
        }
        states[i].credit += states[i].weight;
        if (best_index < 0 || states[i].credit > best_credit) {
            best_credit = states[i].credit;
            best_index = i;
        }
    }

    if (best_index >= 0) {
        states[best_index].credit -= 1.0f;
    }
    return best_index;
}

static int calibration_corpus_finalize_mixed(char **selected,
                                             int selected_count,
                                             calibration_corpus_t *out_corpus) {
    char *buffer = NULL;
    char **samples = NULL;
    size_t total_bytes = 0;
    size_t offset = 0;

    for (int i = 0; i < selected_count; ++i) {
        total_bytes += strlen(selected[i]) + 1u;
    }

    buffer = (char *)malloc(total_bytes > 0 ? total_bytes : 1u);
    samples = (char **)malloc((size_t)selected_count * sizeof(char *));
    if (!buffer || !samples) {
        free(buffer);
        free(samples);
        return -1;
    }

    for (int i = 0; i < selected_count; ++i) {
        size_t sample_len = strlen(selected[i]) + 1u;
        memcpy(buffer + offset, selected[i], sample_len);
        samples[i] = buffer + offset;
        offset += sample_len;
    }

    out_corpus->buffer = buffer;
    out_corpus->samples = samples;
    out_corpus->sample_count = selected_count;
    return 0;
}

static int calibration_corpus_mix_sources(calibration_source_state_t *states,
                                          int state_count,
                                          int max_samples,
                                          calibration_corpus_t *out_corpus) {
    char **selected = NULL;
    int selected_count = 0;

    selected = (char **)malloc((size_t)max_samples * sizeof(char *));
    if (!selected) {
        LOG_ERROR("calibration_corpus_load_manifest: mixed sample allocation failed");
        return -1;
    }

    while (selected_count < max_samples) {
        int source_idx = calibration_corpus_select_source(states, state_count);
        if (source_idx < 0) {
            break;
        }

        selected[selected_count++] = states[source_idx].corpus.samples[states[source_idx].next_index++];
    }

    if (selected_count <= 0) {
        LOG_ERROR("calibration_corpus_load_manifest: no samples selected from manifest sources");
        free(selected);
        return -1;
    }
    if (calibration_corpus_finalize_mixed(selected, selected_count, out_corpus) != 0) {
        LOG_ERROR("calibration_corpus_load_manifest: final mixed corpus allocation failed");
        free(selected);
        return -1;
    }

    free(selected);
    return 0;
}

static void calibration_corpus_destroy_source_states(calibration_source_state_t *states,
                                                     int state_count) {
    if (!states) {
        return;
    }

    for (int i = 0; i < state_count; ++i) {
        calibration_corpus_free(&states[i].corpus);
    }
    free(states);
}

void calibration_corpus_free(calibration_corpus_t *corpus) {
    if (!corpus) {
        return;
    }

    free(corpus->samples);
    free(corpus->buffer);
    memset(corpus, 0, sizeof(*corpus));
}

int calibration_corpus_load(const char *path,
                            int max_samples,
                            calibration_corpus_t *out_corpus) {
    char *owned_buffer = NULL;
    size_t owned_size = 0;

    if (!path || !out_corpus) {
        LOG_ERROR("calibration_corpus_load: invalid arguments");
        return -1;
    }
    if (max_samples <= 0) {
        max_samples = CALIBRATION_CORPUS_DEFAULT_SAMPLES;
    }

    memset(out_corpus, 0, sizeof(*out_corpus));
    if (calibration_corpus_load_text(path, &owned_buffer, &owned_size) != 0) {
        return -1;
    }
    (void)owned_size;
    return calibration_corpus_parse_samples(owned_buffer, path, max_samples, out_corpus);
}

int calibration_corpus_load_manifest(const char *path,
                                     int max_samples,
                                     calibration_corpus_t *out_corpus) {
    calibration_manifest_entry_t *entries = NULL;
    calibration_source_state_t *states = NULL;
    char *manifest_buffer = NULL;
    size_t manifest_size = 0;
    int entry_count = 0;
    int rc = -1;

    if (!path || !out_corpus) {
        LOG_ERROR("calibration_corpus_load_manifest: invalid arguments");
        return -1;
    }
    if (calibration_corpus_is_remote_source(path)) {
        LOG_ERROR("calibration_corpus_load_manifest: manifest must be loaded from a local file: %s", path);
        return -1;
    }
    if (max_samples <= 0) {
        max_samples = CALIBRATION_CORPUS_DEFAULT_SAMPLES;
    }

    memset(out_corpus, 0, sizeof(*out_corpus));
    if (calibration_corpus_load_local_text(path, &manifest_buffer, &manifest_size) != 0) {
        return -1;
    }
    (void)manifest_size;
    if (calibration_corpus_parse_manifest(manifest_buffer,
                                          max_samples,
                                          &entries,
                                          &entry_count) != 0) {
        free(manifest_buffer);
        return -1;
    }

    states = (calibration_source_state_t *)calloc((size_t)entry_count, sizeof(calibration_source_state_t));
    if (!states) {
        LOG_ERROR("calibration_corpus_load_manifest: source state allocation failed");
        free(entries);
        free(manifest_buffer);
        return -1;
    }

    for (int i = 0; i < entry_count; ++i) {
        if (calibration_corpus_load_source(&entries[i], max_samples, &states[i]) != 0) {
            LOG_WARN("calibration_corpus_load_manifest: skipping source %s", entries[i].source);
            memset(&states[i], 0, sizeof(states[i]));
            continue;
        }
    }

    rc = calibration_corpus_mix_sources(states, entry_count, max_samples, out_corpus);
    if (rc == 0) {
        LOG_INFO("Loaded corpus manifest: %s (sources=%d mixed_samples=%d)",
                 path,
                 entry_count,
                 out_corpus->sample_count);
    }

    calibration_corpus_destroy_source_states(states, entry_count);
    free(entries);
    free(manifest_buffer);
    return rc;
}