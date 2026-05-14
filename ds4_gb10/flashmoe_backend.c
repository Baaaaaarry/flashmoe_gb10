#include "flashmoe_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool parse_truthy(const char *s) {
    if (!s || !*s) return false;
    return strcmp(s, "1") == 0 ||
           strcmp(s, "true") == 0 ||
           strcmp(s, "TRUE") == 0 ||
           strcmp(s, "yes") == 0 ||
           strcmp(s, "on") == 0;
}

static ds4_moe_backend parse_backend_name(const char *name) {
    if (!name || !*name) return DS4_MOE_BACKEND_NATIVE;
    if (strcmp(name, "flashmoe") == 0) return DS4_MOE_BACKEND_FLASHMOE;
    if (strcmp(name, "native") == 0) return DS4_MOE_BACKEND_NATIVE;
    return DS4_MOE_BACKEND_NATIVE;
}

const char *ds4_moe_backend_name(ds4_moe_backend backend) {
    switch (backend) {
        case DS4_MOE_BACKEND_FLASHMOE: return "flashmoe";
        case DS4_MOE_BACKEND_NATIVE:
        default: return "native";
    }
}

const ds4_flashmoe_config *ds4_flashmoe_config_get(void) {
    static bool initialized = false;
    static ds4_flashmoe_config cfg;
    if (!initialized) {
        const char *backend = getenv("DS4_MOE_BACKEND");
        cfg.backend = parse_backend_name(backend);
        cfg.manifest_path = getenv("DS4_FLASHMOE_MANIFEST");
        cfg.expert_root = getenv("DS4_FLASHMOE_EXPERT_ROOT");
        cfg.cache_dir = getenv("DS4_FLASHMOE_CACHE_DIR");
        cfg.warn_fallback = !parse_truthy(getenv("DS4_FLASHMOE_SILENT_FALLBACK"));
        initialized = true;
    }
    return &cfg;
}

bool ds4_flashmoe_backend_requested(void) {
    return ds4_flashmoe_config_get()->backend == DS4_MOE_BACKEND_FLASHMOE;
}

bool ds4_flashmoe_backend_enabled(void) {
    const ds4_flashmoe_config *cfg = ds4_flashmoe_config_get();
    return cfg->backend == DS4_MOE_BACKEND_FLASHMOE &&
           cfg->manifest_path && cfg->manifest_path[0] != '\0' &&
           cfg->expert_root && cfg->expert_root[0] != '\0';
}

void ds4_flashmoe_warn_fallback_once(void) {
    static bool warned = false;
    const ds4_flashmoe_config *cfg = ds4_flashmoe_config_get();
    if (warned || !cfg->warn_fallback) return;
    warned = true;
    fprintf(stderr,
            "ds4: DS4_MOE_BACKEND=flashmoe requested, but manifest/root are not fully configured yet; falling back to native routed experts\n");
}
