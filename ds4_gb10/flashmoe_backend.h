#ifndef FLASHMOE_BACKEND_H
#define FLASHMOE_BACKEND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum {
    DS4_MOE_BACKEND_NATIVE = 0,
    DS4_MOE_BACKEND_FLASHMOE = 1,
} ds4_moe_backend;

typedef struct {
    ds4_moe_backend backend;
    const char *manifest_path;
    const char *expert_root;
    const char *cache_dir;
    bool warn_fallback;
} ds4_flashmoe_config;

typedef struct {
    uint16_t layer_id;
    uint16_t expert_id;
    char *path;
    uint64_t offset;
    uint64_t size_bytes;
} ds4_flashmoe_manifest_entry;

typedef struct {
    ds4_flashmoe_manifest_entry *entries;
    size_t count;
    size_t layer_count;
    size_t min_entries_per_layer;
    size_t max_entries_per_layer;
} ds4_flashmoe_manifest;

const ds4_flashmoe_config *ds4_flashmoe_config_get(void);
const char *ds4_moe_backend_name(ds4_moe_backend backend);
bool ds4_flashmoe_backend_requested(void);
bool ds4_flashmoe_backend_enabled(void);
void ds4_flashmoe_warn_fallback_once(void);
int ds4_flashmoe_manifest_load(const char *path,
                               uint32_t max_layers,
                               uint32_t max_experts,
                               ds4_flashmoe_manifest *out,
                               char *err,
                               size_t errlen);
int ds4_flashmoe_manifest_validate_files(const ds4_flashmoe_manifest *manifest,
                                         char *err,
                                         size_t errlen);
void ds4_flashmoe_manifest_free(ds4_flashmoe_manifest *manifest);
const ds4_flashmoe_manifest_entry *ds4_flashmoe_manifest_find(
        const ds4_flashmoe_manifest *manifest,
        uint16_t layer_id,
        uint16_t expert_id);

#endif
