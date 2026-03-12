/**
 * @file calibration_corpus.h
 * @brief Local calibration corpus loading helpers for ternary conversion.
 */

#ifndef CALIBRATION_CORPUS_H
#define CALIBRATION_CORPUS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char *buffer;
    char **samples;
    int sample_count;
} calibration_corpus_t;

int calibration_corpus_load(const char *path,
                            int max_samples,
                            calibration_corpus_t *out_corpus);

void calibration_corpus_free(calibration_corpus_t *corpus);

#ifdef __cplusplus
}
#endif

#endif /* CALIBRATION_CORPUS_H */