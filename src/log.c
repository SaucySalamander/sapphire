/*
 * @file log.c
 * @brief Minimal logger implementation using writev and a thread-local format buffer
 */

#define _POSIX_C_SOURCE 200809L

#include "log.h"
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <unistd.h>
#include <sys/uio.h>

/* Thread-local buffer for formatted messages (no heap allocs) */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Thread_local static char tls_log_buf[4096];
#else
/* Fallback to GCC/Clang __thread */
__thread static char tls_log_buf[4096];
#endif

/* Global log level (simple, relaxed atomicity) */
static volatile int g_log_level = LOG_LEVEL_INFO;

static const char *level_to_str(log_level_t l) {
    switch (l) {
        case LOG_LEVEL_DEBUG: return "DEBUG";
        case LOG_LEVEL_INFO:  return "INFO";
        case LOG_LEVEL_WARN:  return "WARN";
        case LOG_LEVEL_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

void log_set_level(log_level_t level) {
    g_log_level = (int)level;
}

log_level_t log_get_level(void) {
    return (log_level_t)g_log_level;
}

void log_set_level_from_env(const char *env_var_name) {
    if (!env_var_name) return;
    const char *v = getenv(env_var_name);
    if (!v) return;

    /* trim and lowercase */
    while (*v && isspace((unsigned char)*v)) v++;
    if (!*v) return;

    char buf[32];
    size_t i = 0;
    while (*v && !isspace((unsigned char)*v) && i + 1 < sizeof(buf)) {
        buf[i++] = (char)tolower((unsigned char)*v);
        v++;
    }
    buf[i] = '\0';

    if (strcmp(buf, "debug") == 0) log_set_level(LOG_LEVEL_DEBUG);
    else if (strcmp(buf, "info") == 0) log_set_level(LOG_LEVEL_INFO);
    else if (strcmp(buf, "warn") == 0 || strcmp(buf, "warning") == 0) log_set_level(LOG_LEVEL_WARN);
    else if (strcmp(buf, "error") == 0) log_set_level(LOG_LEVEL_ERROR);
    else {
        /* numeric fallback */
        char *end = NULL;
        long lv = strtol(buf, &end, 10);
        if (end != buf) {
            if (lv <= LOG_LEVEL_DEBUG) log_set_level(LOG_LEVEL_DEBUG);
            else if (lv <= LOG_LEVEL_INFO) log_set_level(LOG_LEVEL_INFO);
            else if (lv <= LOG_LEVEL_WARN) log_set_level(LOG_LEVEL_WARN);
            else log_set_level(LOG_LEVEL_ERROR);
        }
    }
}

void log_log(log_level_t level, const char *file, int line, const char *func, const char *fmt, ...) {
    if ((int)level < g_log_level) return;

    /* Build prefix into a small stack buffer */
    char prefix[256];
    time_t t = time(NULL);
    struct tm tm_buf;
    struct tm *tm = localtime_r(&t, &tm_buf);
    int pre_len = 0;
    if (tm) {
        pre_len = snprintf(prefix, sizeof(prefix), "%04d-%02d-%02d %02d:%02d:%02d [%s] %s:%d: ",
                           tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
                           tm->tm_hour, tm->tm_min, tm->tm_sec,
                           level_to_str(level), file ? file : "?", line);
        if (pre_len < 0) pre_len = 0;
        if (pre_len >= (int)sizeof(prefix)) pre_len = sizeof(prefix) - 1;
    } else {
        pre_len = snprintf(prefix, sizeof(prefix), "[%-5s] %s:%d: ", level_to_str(level), file ? file : "?", line);
        if (pre_len < 0) pre_len = 0;
    }

    /* Format message into thread-local buffer (no allocations) */
    char *msg = tls_log_buf;
    int msg_cap = (int)sizeof(tls_log_buf);
    va_list ap;
    va_start(ap, fmt);
    int msg_len = vsnprintf(msg, msg_cap, fmt, ap);
    va_end(ap);
    if (msg_len < 0) msg_len = 0;
    if (msg_len >= msg_cap) msg_len = msg_cap - 1; /* truncated */

    /* Ensure trailing newline via separate iovec */
    const char nl = '\n';

    struct iovec iov[3];
    iov[0].iov_base = prefix;
    iov[0].iov_len = (size_t)pre_len;
    iov[1].iov_base = msg;
    iov[1].iov_len = (size_t)msg_len;
    iov[2].iov_base = (void*)&nl;
    iov[2].iov_len = 1;

    /* Choose FD: WARN/ERROR -> stderr, else stdout */
    int fd = (level >= LOG_LEVEL_WARN) ? STDERR_FILENO : STDOUT_FILENO;

    /* Use writev for atomic multi-buffer write (best-effort). */
    ssize_t rc = writev(fd, iov, 3);
    (void)rc; /* best-effort: don't fail hard */
}
