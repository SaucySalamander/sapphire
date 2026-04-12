#ifndef TRACY_PROFILE_H
#define TRACY_PROFILE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t token;
} sapphire_tracy_zone_t;

typedef struct {
    uint64_t token;
} sapphire_tracy_vk_zone_t;

typedef struct {
    const char *name;
    const char *function;
    const char *file;
    uint32_t line;
    uint32_t color;
} sapphire_tracy_source_location_t;

typedef struct sapphire_tracy_vk_context_t sapphire_tracy_vk_context_t;

#if defined(SAPPHIRE_ENABLE_TRACY)

#include <vulkan/vulkan.h>

int sapphire_tracy_enabled(void);
sapphire_tracy_zone_t sapphire_tracy_zone_begin_static(const sapphire_tracy_source_location_t *location);
sapphire_tracy_zone_t sapphire_tracy_zone_begin(const char *name,
                                                const char *function,
                                                const char *file,
                                                uint32_t line);
void sapphire_tracy_zone_end(sapphire_tracy_zone_t *zone);
void sapphire_tracy_zone_text(const sapphire_tracy_zone_t *zone,
                              const char *text,
                              size_t size);
void sapphire_tracy_zone_name(const sapphire_tracy_zone_t *zone,
                              const char *text,
                              size_t size);
void sapphire_tracy_zone_color(const sapphire_tracy_zone_t *zone,
                               uint32_t color);
void sapphire_tracy_zone_value(const sapphire_tracy_zone_t *zone,
                               uint64_t value);
void sapphire_tracy_name_thread(const char *name);
void sapphire_tracy_frame_mark(const char *name);
void sapphire_tracy_message(const char *message, size_t size);
void sapphire_tracy_plot_f64(const char *name, double value);
void sapphire_tracy_plot_i64(const char *name, int64_t value);
sapphire_tracy_vk_context_t *sapphire_tracy_vk_context_create(VkInstance instance,
                                                              VkPhysicalDevice physical_device,
                                                              VkDevice device,
                                                              uint32_t queue_family,
                                                              VkQueue queue,
                                                              VkCommandBuffer command_buffer);
void sapphire_tracy_vk_context_name(sapphire_tracy_vk_context_t *context,
                                    const char *name,
                                    size_t size);
sapphire_tracy_vk_zone_t sapphire_tracy_vk_zone_begin_static(sapphire_tracy_vk_context_t *context,
                                                             VkCommandBuffer command_buffer,
                                                             const sapphire_tracy_source_location_t *location);
void sapphire_tracy_vk_zone_end(sapphire_tracy_vk_zone_t *zone);
void sapphire_tracy_vk_collect(sapphire_tracy_vk_context_t *context,
                               VkCommandBuffer command_buffer);
void sapphire_tracy_vk_context_destroy(sapphire_tracy_vk_context_t *context);

#else

static inline int sapphire_tracy_enabled(void)
{
    return 0;
}

static inline sapphire_tracy_zone_t sapphire_tracy_zone_begin_static(const sapphire_tracy_source_location_t *location)
{
    sapphire_tracy_zone_t zone = {0u};
    (void)location;
    return zone;
}

static inline sapphire_tracy_zone_t sapphire_tracy_zone_begin(const char *name,
                                                              const char *function,
                                                              const char *file,
                                                              uint32_t line)
{
    sapphire_tracy_zone_t zone = {0u};
    (void)name;
    (void)function;
    (void)file;
    (void)line;
    return zone;
}

static inline void sapphire_tracy_zone_end(sapphire_tracy_zone_t *zone)
{
    (void)zone;
}

static inline void sapphire_tracy_zone_text(const sapphire_tracy_zone_t *zone,
                                            const char *text,
                                            size_t size)
{
    (void)zone;
    (void)text;
    (void)size;
}

static inline void sapphire_tracy_zone_name(const sapphire_tracy_zone_t *zone,
                                            const char *text,
                                            size_t size)
{
    (void)zone;
    (void)text;
    (void)size;
}

static inline void sapphire_tracy_zone_color(const sapphire_tracy_zone_t *zone,
                                             uint32_t color)
{
    (void)zone;
    (void)color;
}

static inline void sapphire_tracy_zone_value(const sapphire_tracy_zone_t *zone,
                                             uint64_t value)
{
    (void)zone;
    (void)value;
}

static inline void sapphire_tracy_name_thread(const char *name)
{
    (void)name;
}

static inline void sapphire_tracy_frame_mark(const char *name)
{
    (void)name;
}

static inline void sapphire_tracy_message(const char *message, size_t size)
{
    (void)message;
    (void)size;
}

static inline void sapphire_tracy_plot_f64(const char *name, double value)
{
    (void)name;
    (void)value;
}

static inline void sapphire_tracy_plot_i64(const char *name, int64_t value)
{
    (void)name;
    (void)value;
}

static inline sapphire_tracy_vk_context_t *sapphire_tracy_vk_context_create(void *instance,
                                                                            void *physical_device,
                                                                            void *device,
                                                                            uint32_t queue_family,
                                                                            void *queue,
                                                                            void *command_buffer)
{
    (void)instance;
    (void)physical_device;
    (void)device;
    (void)queue_family;
    (void)queue;
    (void)command_buffer;
    return (sapphire_tracy_vk_context_t *)0;
}

static inline void sapphire_tracy_vk_context_name(sapphire_tracy_vk_context_t *context,
                                                  const char *name,
                                                  size_t size)
{
    (void)context;
    (void)name;
    (void)size;
}

static inline sapphire_tracy_vk_zone_t sapphire_tracy_vk_zone_begin_static(sapphire_tracy_vk_context_t *context,
                                                                            void *command_buffer,
                                                                            const sapphire_tracy_source_location_t *location)
{
    sapphire_tracy_vk_zone_t zone = {0u};
    (void)context;
    (void)command_buffer;
    (void)location;
    return zone;
}

static inline void sapphire_tracy_vk_zone_end(sapphire_tracy_vk_zone_t *zone)
{
    (void)zone;
}

static inline void sapphire_tracy_vk_collect(sapphire_tracy_vk_context_t *context,
                                             void *command_buffer)
{
    (void)context;
    (void)command_buffer;
}

static inline void sapphire_tracy_vk_context_destroy(sapphire_tracy_vk_context_t *context)
{
    (void)context;
}

#endif

#define SAPPHIRE_TRACY_CONCAT_INNER(a, b) a##b
#define SAPPHIRE_TRACY_CONCAT(a, b) SAPPHIRE_TRACY_CONCAT_INNER(a, b)

#define SAPPHIRE_TRACY_ZONE_SCOPE(var, name_literal) \
    static const sapphire_tracy_source_location_t \
        SAPPHIRE_TRACY_CONCAT(_sapphire_tracy_srcloc_, __LINE__) = { \
            (name_literal), __func__, __FILE__, (uint32_t)__LINE__, 0u \
        }; \
    sapphire_tracy_zone_t var = \
        sapphire_tracy_zone_begin_static(&SAPPHIRE_TRACY_CONCAT(_sapphire_tracy_srcloc_, __LINE__))

#define SAPPHIRE_TRACY_ZONE_BEGIN(name) \
    sapphire_tracy_zone_begin((name), __func__, __FILE__, (uint32_t)__LINE__)

#define SAPPHIRE_TRACY_VK_ZONE_SCOPE(var, context, command_buffer, name_literal) \
    static const sapphire_tracy_source_location_t \
        SAPPHIRE_TRACY_CONCAT(_sapphire_tracy_vk_srcloc_, __LINE__) = { \
            (name_literal), __func__, __FILE__, (uint32_t)__LINE__, 0u \
        }; \
    sapphire_tracy_vk_zone_t var = \
        sapphire_tracy_vk_zone_begin_static((context), \
                                            (command_buffer), \
                                            &SAPPHIRE_TRACY_CONCAT(_sapphire_tracy_vk_srcloc_, __LINE__))

#define SAPPHIRE_TRACY_MESSAGE_LITERAL(message) \
    sapphire_tracy_message((message), sizeof(message) - 1u)

#ifdef __cplusplus
}
#endif

#endif