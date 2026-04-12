#include "tracy_profile.h"

#if defined(SAPPHIRE_ENABLE_TRACY)

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>
#include <strings.h>
#include <type_traits>

#include <tracy/TracyC.h>
#include <tracy/TracyVulkan.hpp>

namespace {

static_assert(sizeof(sapphire_tracy_source_location_t) == sizeof(___tracy_source_location_data),
              "sapphire_tracy_source_location_t must match Tracy source location layout");
static_assert(sizeof(sapphire_tracy_source_location_t) == sizeof(tracy::SourceLocationData),
              "sapphire_tracy_source_location_t must match Tracy C++ source location layout");
static_assert(offsetof(sapphire_tracy_source_location_t, name) == offsetof(___tracy_source_location_data, name),
              "Tracy source location name offset mismatch");
static_assert(offsetof(sapphire_tracy_source_location_t, function) == offsetof(___tracy_source_location_data, function),
              "Tracy source location function offset mismatch");
static_assert(offsetof(sapphire_tracy_source_location_t, file) == offsetof(___tracy_source_location_data, file),
              "Tracy source location file offset mismatch");
static_assert(offsetof(sapphire_tracy_source_location_t, line) == offsetof(___tracy_source_location_data, line),
              "Tracy source location line offset mismatch");
static_assert(offsetof(sapphire_tracy_source_location_t, color) == offsetof(___tracy_source_location_data, color),
              "Tracy source location color offset mismatch");

constexpr uint32_t kTracyZoneSlotCount = 256u;
constexpr uint32_t kTracyVkZoneSlotCount = 256u;

struct ThreadZoneStorage {
    TracyCZoneCtx zones[kTracyZoneSlotCount];
    uint8_t active[kTracyZoneSlotCount];
    uint32_t next_hint;
};

struct ThreadVkZoneStorage {
    typename std::aligned_storage<sizeof(tracy::VkCtxScope), alignof(tracy::VkCtxScope)>::type zones[kTracyVkZoneSlotCount];
    uint8_t active[kTracyVkZoneSlotCount];
    uint32_t next_hint;
};

thread_local ThreadZoneStorage g_thread_zones = {};
thread_local ThreadVkZoneStorage g_thread_vk_zones = {};
std::atomic<int> g_tracy_runtime_enabled(-1);

static int env_flag_enabled(const char *value)
{
    if (!value || value[0] == '\0') {
        return 0;
    }

    return strcmp(value, "1") == 0 ||
           strcasecmp(value, "true") == 0 ||
           strcasecmp(value, "yes") == 0 ||
           strcasecmp(value, "on") == 0;
}

static uint32_t next_slot_index(void)
{
    ThreadZoneStorage &storage = g_thread_zones;

    for (uint32_t scan = 0; scan < kTracyZoneSlotCount; ++scan) {
        uint32_t idx = (storage.next_hint + scan) % kTracyZoneSlotCount;

        if (!storage.active[idx]) {
            storage.next_hint = (idx + 1u) % kTracyZoneSlotCount;
            return idx;
        }
    }

    return kTracyZoneSlotCount;
}

static TracyCZoneCtx *lookup_zone_ctx(uint64_t token)
{
    if (token == 0u || token > kTracyZoneSlotCount) {
        return nullptr;
    }

    ThreadZoneStorage &storage = g_thread_zones;
    const uint32_t idx = (uint32_t)(token - 1u);

    if (!storage.active[idx]) {
        return nullptr;
    }

    return &storage.zones[idx];
}

static uint32_t next_vk_slot_index(void)
{
    ThreadVkZoneStorage &storage = g_thread_vk_zones;

    for (uint32_t scan = 0; scan < kTracyVkZoneSlotCount; ++scan) {
        uint32_t idx = (storage.next_hint + scan) % kTracyVkZoneSlotCount;

        if (!storage.active[idx]) {
            storage.next_hint = (idx + 1u) % kTracyVkZoneSlotCount;
            return idx;
        }
    }

    return kTracyVkZoneSlotCount;
}

static tracy::VkCtxScope *lookup_vk_zone_ctx(uint64_t token)
{
    if (token == 0u || token > kTracyVkZoneSlotCount) {
        return nullptr;
    }

    ThreadVkZoneStorage &storage = g_thread_vk_zones;
    const uint32_t idx = (uint32_t)(token - 1u);

    if (!storage.active[idx]) {
        return nullptr;
    }

    return reinterpret_cast<tracy::VkCtxScope *>(&storage.zones[idx]);
}

} // namespace

struct sapphire_tracy_vk_context_t {
    TracyVkCtx context;
};

extern "C" int sapphire_tracy_enabled(void)
{
    int cached = g_tracy_runtime_enabled.load(std::memory_order_acquire);

    if (cached >= 0) {
        return cached;
    }

    cached = env_flag_enabled(getenv("SAPPHIRE_TRACY")) ? 1 : 0;
    g_tracy_runtime_enabled.store(cached, std::memory_order_release);
    return cached;
}

extern "C" sapphire_tracy_zone_t sapphire_tracy_zone_begin_static(const sapphire_tracy_source_location_t *location)
{
    sapphire_tracy_zone_t zone = {0u};

    if (!location || sapphire_tracy_enabled() == 0) {
        return zone;
    }

    const uint32_t idx = next_slot_index();
    if (idx >= kTracyZoneSlotCount) {
        return zone;
    }

    ThreadZoneStorage &storage = g_thread_zones;
    storage.zones[idx] = ___tracy_emit_zone_begin(
        reinterpret_cast<const ___tracy_source_location_data *>(location), 1);
    storage.active[idx] = 1u;
    zone.token = (uint64_t)idx + 1u;
    return zone;
}

extern "C" sapphire_tracy_zone_t sapphire_tracy_zone_begin(const char *name,
                                                             const char *function,
                                                             const char *file,
                                                             uint32_t line)
{
    sapphire_tracy_zone_t zone = {0u};

    if (!name || !function || !file || sapphire_tracy_enabled() == 0) {
        return zone;
    }

    const uint32_t idx = next_slot_index();
    if (idx >= kTracyZoneSlotCount) {
        return zone;
    }

    const uint64_t srcloc = ___tracy_alloc_srcloc_name(line,
                                                       file,
                                                       strlen(file) + 1u,
                                                       function,
                                                       strlen(function) + 1u,
                                                       name,
                                                       strlen(name) + 1u,
                                                       0u);

    ThreadZoneStorage &storage = g_thread_zones;
    storage.zones[idx] = ___tracy_emit_zone_begin_alloc(srcloc, 1);
    storage.active[idx] = 1u;
    zone.token = (uint64_t)idx + 1u;
    return zone;
}

extern "C" void sapphire_tracy_zone_end(sapphire_tracy_zone_t *zone)
{
    TracyCZoneCtx *ctx = nullptr;
    ThreadZoneStorage &storage = g_thread_zones;
    uint32_t idx = 0u;

    if (!zone || zone->token == 0u) {
        return;
    }

    ctx = lookup_zone_ctx(zone->token);
    if (!ctx) {
        zone->token = 0u;
        return;
    }

    idx = (uint32_t)(zone->token - 1u);
    ___tracy_emit_zone_end(*ctx);
    storage.active[idx] = 0u;
    storage.zones[idx].id = 0u;
    storage.zones[idx].active = 0;
    storage.next_hint = idx;
    zone->token = 0u;
}

extern "C" void sapphire_tracy_zone_text(const sapphire_tracy_zone_t *zone,
                                          const char *text,
                                          size_t size)
{
    TracyCZoneCtx *ctx = nullptr;

    if (!zone || !text || size == 0u || sapphire_tracy_enabled() == 0) {
        return;
    }

    ctx = lookup_zone_ctx(zone->token);
    if (ctx) {
        ___tracy_emit_zone_text(*ctx, text, size);
    }
}

extern "C" void sapphire_tracy_zone_name(const sapphire_tracy_zone_t *zone,
                                          const char *text,
                                          size_t size)
{
    TracyCZoneCtx *ctx = nullptr;

    if (!zone || !text || size == 0u || sapphire_tracy_enabled() == 0) {
        return;
    }

    ctx = lookup_zone_ctx(zone->token);
    if (ctx) {
        ___tracy_emit_zone_name(*ctx, text, size);
    }
}

extern "C" void sapphire_tracy_zone_color(const sapphire_tracy_zone_t *zone,
                                           uint32_t color)
{
    TracyCZoneCtx *ctx = nullptr;

    if (!zone || sapphire_tracy_enabled() == 0) {
        return;
    }

    ctx = lookup_zone_ctx(zone->token);
    if (ctx) {
        ___tracy_emit_zone_color(*ctx, color);
    }
}

extern "C" void sapphire_tracy_zone_value(const sapphire_tracy_zone_t *zone,
                                           uint64_t value)
{
    TracyCZoneCtx *ctx = nullptr;

    if (!zone || sapphire_tracy_enabled() == 0) {
        return;
    }

    ctx = lookup_zone_ctx(zone->token);
    if (ctx) {
        ___tracy_emit_zone_value(*ctx, value);
    }
}

extern "C" void sapphire_tracy_name_thread(const char *name)
{
    if (!name || name[0] == '\0' || sapphire_tracy_enabled() == 0) {
        return;
    }

    TracyCSetThreadName(name);
}

extern "C" void sapphire_tracy_frame_mark(const char *name)
{
    if (sapphire_tracy_enabled() == 0) {
        return;
    }

    if (name && name[0] != '\0') {
        TracyCFrameMarkNamed(name);
    } else {
        TracyCFrameMark;
    }
}

extern "C" void sapphire_tracy_message(const char *message, size_t size)
{
    if (!message || size == 0u || sapphire_tracy_enabled() == 0) {
        return;
    }

    TracyCMessage(message, size);
}

extern "C" void sapphire_tracy_plot_f64(const char *name, double value)
{
    if (!name || name[0] == '\0' || sapphire_tracy_enabled() == 0) {
        return;
    }

    TracyCPlot(name, value);
}

extern "C" void sapphire_tracy_plot_i64(const char *name, int64_t value)
{
    if (!name || name[0] == '\0' || sapphire_tracy_enabled() == 0) {
        return;
    }

    TracyCPlotI(name, value);
}

extern "C" sapphire_tracy_vk_context_t *sapphire_tracy_vk_context_create(VkInstance instance,
                                                                            VkPhysicalDevice physical_device,
                                                                            VkDevice device,
                                                                            uint32_t queue_family,
                                                                            VkQueue queue,
                                                                            VkCommandBuffer command_buffer)
{
    (void)queue_family;

    if (sapphire_tracy_enabled() == 0 || !instance || !physical_device || !device || !queue || !command_buffer) {
        return nullptr;
    }

    sapphire_tracy_vk_context_t *wrapper = new (std::nothrow) sapphire_tracy_vk_context_t{};
    if (!wrapper) {
        return nullptr;
    }

#if defined(TRACY_VK_USE_SYMBOL_TABLE)
    wrapper->context = TracyVkContextCalibrated(instance,
                                                physical_device,
                                                device,
                                                queue,
                                                command_buffer,
                                                vkGetInstanceProcAddr,
                                                vkGetDeviceProcAddr);
#else
    PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT gpdctd =
        reinterpret_cast<PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT>(
            vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCalibrateableTimeDomainsEXT"));
    PFN_vkGetCalibratedTimestampsEXT gct =
        reinterpret_cast<PFN_vkGetCalibratedTimestampsEXT>(
            vkGetDeviceProcAddr(device, "vkGetCalibratedTimestampsEXT"));
    wrapper->context = TracyVkContextCalibrated(physical_device,
                                                device,
                                                queue,
                                                command_buffer,
                                                gpdctd,
                                                gct);
#endif

    if (!wrapper->context) {
        delete wrapper;
        return nullptr;
    }

    return wrapper;
}

extern "C" void sapphire_tracy_vk_context_name(sapphire_tracy_vk_context_t *context,
                                                  const char *name,
                                                  size_t size)
{
    if (!context || !context->context || !name || size == 0u || sapphire_tracy_enabled() == 0) {
        return;
    }

    TracyVkContextName(context->context, name, size);
}

extern "C" sapphire_tracy_vk_zone_t sapphire_tracy_vk_zone_begin_static(sapphire_tracy_vk_context_t *context,
                                                                            VkCommandBuffer command_buffer,
                                                                            const sapphire_tracy_source_location_t *location)
{
    sapphire_tracy_vk_zone_t zone = {0u};

    if (!context || !context->context || !command_buffer || !location || sapphire_tracy_enabled() == 0) {
        return zone;
    }

    const uint32_t idx = next_vk_slot_index();
    if (idx >= kTracyVkZoneSlotCount) {
        return zone;
    }

    ThreadVkZoneStorage &storage = g_thread_vk_zones;
    void *slot = &storage.zones[idx];
    new (slot) tracy::VkCtxScope(context->context,
                                 reinterpret_cast<const tracy::SourceLocationData *>(location),
                                 command_buffer,
                                 true);
    storage.active[idx] = 1u;
    zone.token = (uint64_t)idx + 1u;
    return zone;
}

extern "C" void sapphire_tracy_vk_zone_end(sapphire_tracy_vk_zone_t *zone)
{
    tracy::VkCtxScope *ctx = nullptr;
    ThreadVkZoneStorage &storage = g_thread_vk_zones;
    uint32_t idx = 0u;

    if (!zone || zone->token == 0u) {
        return;
    }

    ctx = lookup_vk_zone_ctx(zone->token);
    if (!ctx) {
        zone->token = 0u;
        return;
    }

    idx = (uint32_t)(zone->token - 1u);
    ctx->~VkCtxScope();
    storage.active[idx] = 0u;
    storage.next_hint = idx;
    zone->token = 0u;
}

extern "C" void sapphire_tracy_vk_collect(sapphire_tracy_vk_context_t *context,
                                             VkCommandBuffer command_buffer)
{
    if (!context || !context->context || !command_buffer || sapphire_tracy_enabled() == 0) {
        return;
    }

    TracyVkCollect(context->context, command_buffer);
}

extern "C" void sapphire_tracy_vk_context_destroy(sapphire_tracy_vk_context_t *context)
{
    if (!context) {
        return;
    }

    if (context->context && sapphire_tracy_enabled() != 0) {
        TracyVkDestroy(context->context);
        context->context = nullptr;
    }

    delete context;
}

#endif