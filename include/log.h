/*
 * @file log.h
 * @brief Minimal writev-based logger: thread-local buffer, LOG_* macros, atomic writev output.
 *
 * - Uses a thread-local formatting buffer (no heap allocations)
 * - Exposes LOG_DEBUG/INFO/WARN/ERROR macros
 * - Writes log prefix + message + '\n' atomically via writev to avoid extra copies
 *
 * This is intentionally minimal and dependency-free.
 */

#ifndef SAPPHIRE_LOG_H
#define SAPPHIRE_LOG_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    LOG_LEVEL_DEBUG = 10,
    LOG_LEVEL_INFO  = 20,
    LOG_LEVEL_WARN  = 30,
    LOG_LEVEL_ERROR = 40,
    LOG_LEVEL_NONE  = 100
} log_level_t;

void log_set_level(log_level_t level);
log_level_t log_get_level(void);
void log_set_level_from_env(const char *env_var_name);

void log_log(log_level_t level, const char *file, int line, const char *func, const char *fmt, ...);

#define LOG_DEBUG(fmt, ...) log_log(LOG_LEVEL_DEBUG, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  log_log(LOG_LEVEL_INFO,  __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  log_log(LOG_LEVEL_WARN,  __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) log_log(LOG_LEVEL_ERROR, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif /* SAPPHIRE_LOG_H */
