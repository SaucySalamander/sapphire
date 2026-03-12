/**
 * @file calibration_corpus.c
 * @brief Local calibration corpus loading helpers for ternary conversion.
 */

#include "calibration_corpus.h"

#include "file_reader.h"
#include "log.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

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
    char *raw_buffer = NULL;
    size_t raw_size = 0;
    char *owned_buffer = NULL;
    char *cursor = NULL;
    char **samples = NULL;
    int capacity = 0;
    int count = 0;

    if (!path || !out_corpus) {
        LOG_ERROR("calibration_corpus_load: invalid arguments");
        return -1;
    }
    if (max_samples <= 0) {
        max_samples = 64;
    }

    memset(out_corpus, 0, sizeof(*out_corpus));
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

    capacity = (max_samples < 16) ? max_samples : 16;
    if (capacity <= 0) {
        capacity = 16;
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
        LOG_ERROR("calibration_corpus_load: no usable samples found in %s", path);
        free(samples);
        free(owned_buffer);
        return -1;
    }

    out_corpus->buffer = owned_buffer;
    out_corpus->samples = samples;
    out_corpus->sample_count = count;
    LOG_INFO("Loaded calibration corpus: %s (%d samples)", path, count);
    return 0;
}