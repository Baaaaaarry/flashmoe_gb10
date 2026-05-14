#ifndef FLASHMOE_BACKEND_H
#define FLASHMOE_BACKEND_H

#include <stdbool.h>

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

const ds4_flashmoe_config *ds4_flashmoe_config_get(void);
const char *ds4_moe_backend_name(ds4_moe_backend backend);
bool ds4_flashmoe_backend_requested(void);
bool ds4_flashmoe_backend_enabled(void);
void ds4_flashmoe_warn_fallback_once(void);

#endif
