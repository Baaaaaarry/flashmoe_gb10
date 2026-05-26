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
    uint64_t cache_limit_bytes;
    bool warn_fallback;
} ds4_flashmoe_config;

typedef struct {
    uint16_t layer_id;
    char *path;
    uint64_t expert_size;
    uint64_t num_experts;
    uint64_t gate_bytes;
    uint64_t up_bytes;
    uint64_t down_bytes;
    uint64_t gate_row_bytes;
    uint64_t up_row_bytes;
    uint64_t down_row_bytes;
} ds4_flashmoe_layer_pack;

typedef struct {
    uint32_t version;
    uint32_t layout_version;
    ds4_flashmoe_layer_pack *layers;
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
const ds4_flashmoe_layer_pack *ds4_flashmoe_manifest_find_layer(
        const ds4_flashmoe_manifest *manifest,
        uint16_t layer_id);
int ds4_flashmoe_runtime_open(const ds4_flashmoe_manifest *manifest,
                              uint64_t cache_limit_bytes,
                              char *err,
                              size_t errlen);
void ds4_flashmoe_runtime_close(void);
bool ds4_flashmoe_runtime_ready(void);
const uint8_t *ds4_flashmoe_runtime_get_blob(uint16_t layer_id,
                                             uint16_t expert_id,
                                             uint64_t expected_size,
                                             uint64_t *actual_size,
                                             char *err,
                                             size_t errlen);
int ds4_flashmoe_runtime_load_selected_pack(uint16_t layer_id,
                                            const uint16_t *expert_ids,
                                            uint32_t n_experts,
                                            uint8_t *gate_dst,
                                            uint8_t *up_dst,
                                            uint8_t *down_dst,
                                            uint64_t gate_bytes,
                                            uint64_t up_bytes,
                                            uint64_t down_bytes,
                                            char *err,
                                            size_t errlen);

#endif
