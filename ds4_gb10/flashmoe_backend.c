#include "flashmoe_backend.h"

#include <errno.h>
#include <inttypes.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void set_err(char *err, size_t errlen, const char *msg) {
    if (err && errlen) snprintf(err, errlen, "%s", msg);
}

static bool parse_truthy(const char *s) {
    if (!s || !*s) return false;
    return strcmp(s, "1") == 0 ||
           strcmp(s, "true") == 0 ||
           strcmp(s, "TRUE") == 0 ||
           strcmp(s, "yes") == 0 ||
           strcmp(s, "on") == 0;
}

static uint64_t parse_gib_bytes(const char *s, uint64_t fallback) {
    if (!s || !*s) return fallback;
    char *end = NULL;
    errno = 0;
    double v = strtod(s, &end);
    if (errno != 0 || end == s || v <= 0.0) return fallback;
    double bytes = v * 1024.0 * 1024.0 * 1024.0;
    if (bytes > (double)UINT64_MAX) return fallback;
    return (uint64_t)bytes;
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
        cfg.cache_limit_bytes = parse_gib_bytes(getenv("DS4_FLASHMOE_CACHE_LIMIT_GB"),
                                                4ull * 1024ull * 1024ull * 1024ull);
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

static char *read_text_file(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long end = ftell(fp);
    if (end < 0) {
        fclose(fp);
        return NULL;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return NULL;
    }
    char *buf = (char *)malloc((size_t)end + 1u);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    const size_t n = fread(buf, 1, (size_t)end, fp);
    fclose(fp);
    if (n != (size_t)end) {
        free(buf);
        return NULL;
    }
    buf[n] = '\0';
    if (len_out) *len_out = n;
    return buf;
}

static char *dup_json_string_field(const char *obj, const char *key) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(obj, pattern);
    if (!p) return NULL;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (*p != ':') return NULL;
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (*p != '"') return NULL;
    p++;
    const char *q = p;
    while (*q && *q != '"') q++;
    if (*q != '"') return NULL;
    const size_t n = (size_t)(q - p);
    char *out = (char *)malloc(n + 1u);
    if (!out) return NULL;
    memcpy(out, p, n);
    out[n] = '\0';
    return out;
}

static bool parse_json_u64_field(const char *obj, const char *key, uint64_t *out) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(obj, pattern);
    if (!p) return false;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (*p != ':') return false;
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    errno = 0;
    char *end = NULL;
    unsigned long long v = strtoull(p, &end, 10);
    if (errno != 0 || end == p) return false;
    *out = (uint64_t)v;
    return true;
}

static bool manifest_push(ds4_flashmoe_manifest *manifest,
                          ds4_flashmoe_manifest_entry *entry,
                          size_t *cap) {
    if (manifest->count == *cap) {
        size_t new_cap = *cap ? *cap * 2u : 256u;
        void *new_entries = realloc(manifest->entries, new_cap * sizeof(manifest->entries[0]));
        if (!new_entries) return false;
        manifest->entries = (ds4_flashmoe_manifest_entry *)new_entries;
        *cap = new_cap;
    }
    manifest->entries[manifest->count++] = *entry;
    return true;
}

int ds4_flashmoe_manifest_load(const char *path,
                               uint32_t max_layers,
                               uint32_t max_experts,
                               ds4_flashmoe_manifest *out,
                               char *err,
                               size_t errlen) {
    if (!out) {
        set_err(err, errlen, "null manifest output");
        return 1;
    }
    memset(out, 0, sizeof(*out));
    size_t text_len = 0;
    char *text = read_text_file(path, &text_len);
    if (!text) {
        set_err(err, errlen, "failed to read flashmoe manifest");
        return 1;
    }

    size_t cap = 0;
    char *p = text;
    while ((p = strchr(p, '{')) != NULL) {
        char *q = strchr(p, '}');
        if (!q) break;
        const size_t n = (size_t)(q - p + 1);
        char *obj = (char *)malloc(n + 1u);
        if (!obj) {
            free(text);
            ds4_flashmoe_manifest_free(out);
            set_err(err, errlen, "out of memory parsing manifest");
            return 1;
        }
        memcpy(obj, p, n);
        obj[n] = '\0';

        uint64_t layer = 0, expert = 0, offset = 0, size_bytes = 0;
        char *entry_path = NULL;
        bool ok = parse_json_u64_field(obj, "layer_id", &layer) &&
                  parse_json_u64_field(obj, "expert_id", &expert) &&
                  parse_json_u64_field(obj, "offset", &offset) &&
                  parse_json_u64_field(obj, "size_bytes", &size_bytes);
        if (ok) entry_path = dup_json_string_field(obj, "path");
        if (ok && entry_path) {
            if (layer >= max_layers || expert >= max_experts) {
                free(entry_path);
                free(obj);
                free(text);
                ds4_flashmoe_manifest_free(out);
                set_err(err, errlen, "manifest entry is outside ds4 model bounds");
                return 1;
            }
            ds4_flashmoe_manifest_entry entry;
            entry.layer_id = (uint16_t)layer;
            entry.expert_id = (uint16_t)expert;
            entry.path = entry_path;
            entry.offset = offset;
            entry.size_bytes = size_bytes;
            if (!manifest_push(out, &entry, &cap)) {
                free(entry_path);
                free(obj);
                free(text);
                ds4_flashmoe_manifest_free(out);
                set_err(err, errlen, "out of memory growing manifest");
                return 1;
            }
        }
        free(obj);
        p = q + 1;
    }
    free(text);

    if (out->count == 0) {
        ds4_flashmoe_manifest_free(out);
        set_err(err, errlen, "no manifest entries parsed");
        return 1;
    }

    for (size_t i = 0; i < out->count; i++) {
        for (size_t j = i + 1; j < out->count; j++) {
            const ds4_flashmoe_manifest_entry *a = &out->entries[i];
            const ds4_flashmoe_manifest_entry *b = &out->entries[j];
            if (a->layer_id == b->layer_id && a->expert_id == b->expert_id) {
                ds4_flashmoe_manifest_free(out);
                set_err(err, errlen, "duplicate layer/expert entry in manifest");
                return 1;
            }
        }
    }

    size_t *layer_counts = (size_t *)calloc(max_layers, sizeof(size_t));
    if (!layer_counts) {
        ds4_flashmoe_manifest_free(out);
        set_err(err, errlen, "out of memory building layer counts");
        return 1;
    }
    for (size_t i = 0; i < out->count; i++) {
        layer_counts[out->entries[i].layer_id]++;
    }
    out->min_entries_per_layer = (size_t)-1;
    for (uint32_t il = 0; il < max_layers; il++) {
        const size_t n = layer_counts[il];
        if (n == 0) continue;
        out->layer_count++;
        if (n < out->min_entries_per_layer) out->min_entries_per_layer = n;
        if (n > out->max_entries_per_layer) out->max_entries_per_layer = n;
    }
    if (out->layer_count == 0) out->min_entries_per_layer = 0;
    free(layer_counts);
    return 0;
}

int ds4_flashmoe_manifest_validate_files(const ds4_flashmoe_manifest *manifest,
                                         char *err,
                                         size_t errlen) {
    if (!manifest || !manifest->entries || manifest->count == 0) {
        set_err(err, errlen, "manifest has no entries");
        return 1;
    }
    for (size_t i = 0; i < manifest->count; i++) {
        const ds4_flashmoe_manifest_entry *entry = &manifest->entries[i];
        struct stat st;
        if (stat(entry->path, &st) != 0) {
            set_err(err, errlen, "manifest entry file is missing");
            return 1;
        }
        if ((uint64_t)st.st_size < entry->offset) {
            set_err(err, errlen, "manifest entry offset is outside file");
            return 1;
        }
        if ((uint64_t)st.st_size - entry->offset < entry->size_bytes) {
            set_err(err, errlen, "manifest entry size exceeds file bounds");
            return 1;
        }
    }
    return 0;
}

void ds4_flashmoe_manifest_free(ds4_flashmoe_manifest *manifest) {
    if (!manifest) return;
    if (manifest->entries) {
        for (size_t i = 0; i < manifest->count; i++) {
            free(manifest->entries[i].path);
        }
    }
    free(manifest->entries);
    memset(manifest, 0, sizeof(*manifest));
}

const ds4_flashmoe_manifest_entry *ds4_flashmoe_manifest_find(
        const ds4_flashmoe_manifest *manifest,
        uint16_t layer_id,
        uint16_t expert_id) {
    if (!manifest) return NULL;
    for (size_t i = 0; i < manifest->count; i++) {
        const ds4_flashmoe_manifest_entry *entry = &manifest->entries[i];
        if (entry->layer_id == layer_id && entry->expert_id == expert_id) {
            return entry;
        }
    }
    return NULL;
}

typedef struct {
    uint16_t layer_id;
    uint16_t expert_id;
    uint8_t *bytes;
    uint64_t size_bytes;
    uint64_t last_use;
} ds4_flashmoe_blob_entry;

typedef struct {
    const ds4_flashmoe_manifest *manifest;
    ds4_flashmoe_blob_entry *entries;
    size_t count;
    size_t cap;
    uint64_t used_bytes;
    uint64_t limit_bytes;
    uint64_t use_clock;
    bool ready;
} ds4_flashmoe_runtime_state;

static ds4_flashmoe_runtime_state g_runtime;

static void runtime_reset(ds4_flashmoe_runtime_state *rt) {
    if (!rt) return;
    for (size_t i = 0; i < rt->count; i++) {
        free(rt->entries[i].bytes);
    }
    free(rt->entries);
    memset(rt, 0, sizeof(*rt));
}

static ds4_flashmoe_blob_entry *runtime_find_blob(ds4_flashmoe_runtime_state *rt,
                                                  uint16_t layer_id,
                                                  uint16_t expert_id) {
    if (!rt) return NULL;
    for (size_t i = 0; i < rt->count; i++) {
        ds4_flashmoe_blob_entry *entry = &rt->entries[i];
        if (entry->layer_id == layer_id && entry->expert_id == expert_id) return entry;
    }
    return NULL;
}

static bool runtime_reserve_slot(ds4_flashmoe_runtime_state *rt) {
    if (rt->count < rt->cap) return true;
    size_t new_cap = rt->cap ? rt->cap * 2u : 128u;
    void *new_entries = realloc(rt->entries, new_cap * sizeof(rt->entries[0]));
    if (!new_entries) return false;
    rt->entries = (ds4_flashmoe_blob_entry *)new_entries;
    rt->cap = new_cap;
    return true;
}

static bool runtime_evict_one(ds4_flashmoe_runtime_state *rt) {
    if (!rt || rt->count == 0) return false;
    size_t victim = 0;
    uint64_t oldest = rt->entries[0].last_use;
    for (size_t i = 1; i < rt->count; i++) {
        if (rt->entries[i].last_use < oldest) {
            oldest = rt->entries[i].last_use;
            victim = i;
        }
    }
    rt->used_bytes -= rt->entries[victim].size_bytes;
    free(rt->entries[victim].bytes);
    rt->entries[victim] = rt->entries[rt->count - 1];
    rt->count--;
    return true;
}

static int runtime_load_blob(ds4_flashmoe_runtime_state *rt,
                             const ds4_flashmoe_manifest_entry *manifest_entry,
                             uint64_t expected_size,
                             char *err,
                             size_t errlen) {
    if (!rt || !manifest_entry) {
        set_err(err, errlen, "invalid runtime manifest entry");
        return 1;
    }
    if (manifest_entry->size_bytes != expected_size) {
        snprintf(err,
                 errlen,
                 "FlashMoE manifest size mismatch for layer=%u expert=%u (manifest=%" PRIu64 " expected=%" PRIu64 ")",
                 (unsigned)manifest_entry->layer_id,
                 (unsigned)manifest_entry->expert_id,
                 manifest_entry->size_bytes,
                 expected_size);
        return 1;
    }
    if (expected_size > rt->limit_bytes && rt->limit_bytes > 0) {
        set_err(err, errlen, "FlashMoE expert blob exceeds cache budget");
        return 1;
    }
    while (rt->limit_bytes > 0 &&
           rt->used_bytes + expected_size > rt->limit_bytes &&
           rt->count > 0) {
        if (!runtime_evict_one(rt)) break;
    }
    int fd = open(manifest_entry->path, O_RDONLY);
    if (fd < 0) {
        set_err(err, errlen, "failed to open FlashMoE expert blob");
        return 1;
    }
    uint8_t *bytes = (uint8_t *)malloc((size_t)expected_size);
    if (!bytes) {
        close(fd);
        set_err(err, errlen, "out of memory allocating FlashMoE expert blob");
        return 1;
    }
    size_t done = 0;
    while (done < expected_size) {
        ssize_t n = pread(fd,
                          bytes + done,
                          (size_t)(expected_size - done),
                          (off_t)(manifest_entry->offset + done));
        if (n <= 0) {
            free(bytes);
            close(fd);
            set_err(err, errlen, "failed to read FlashMoE expert blob");
            return 1;
        }
        done += (size_t)n;
    }
    close(fd);
    if (!runtime_reserve_slot(rt)) {
        free(bytes);
        set_err(err, errlen, "out of memory growing FlashMoE blob cache");
        return 1;
    }
    ds4_flashmoe_blob_entry *entry = &rt->entries[rt->count++];
    entry->layer_id = manifest_entry->layer_id;
    entry->expert_id = manifest_entry->expert_id;
    entry->bytes = bytes;
    entry->size_bytes = expected_size;
    entry->last_use = ++rt->use_clock;
    rt->used_bytes += expected_size;
    return 0;
}

int ds4_flashmoe_runtime_open(const ds4_flashmoe_manifest *manifest,
                              uint64_t cache_limit_bytes,
                              char *err,
                              size_t errlen) {
    runtime_reset(&g_runtime);
    if (!manifest || !manifest->entries || manifest->count == 0) {
        set_err(err, errlen, "FlashMoE runtime needs a non-empty manifest");
        return 1;
    }
    g_runtime.manifest = manifest;
    g_runtime.limit_bytes = cache_limit_bytes;
    g_runtime.ready = true;
    return 0;
}

void ds4_flashmoe_runtime_close(void) {
    runtime_reset(&g_runtime);
}

bool ds4_flashmoe_runtime_ready(void) {
    return g_runtime.ready && g_runtime.manifest != NULL;
}

const uint8_t *ds4_flashmoe_runtime_get_blob(uint16_t layer_id,
                                             uint16_t expert_id,
                                             uint64_t expected_size,
                                             uint64_t *actual_size,
                                             char *err,
                                             size_t errlen) {
    if (!ds4_flashmoe_runtime_ready()) {
        set_err(err, errlen, "FlashMoE runtime cache is not ready");
        return NULL;
    }
    ds4_flashmoe_blob_entry *entry = runtime_find_blob(&g_runtime, layer_id, expert_id);
    if (!entry) {
        const ds4_flashmoe_manifest_entry *manifest_entry =
                ds4_flashmoe_manifest_find(g_runtime.manifest, layer_id, expert_id);
        if (!manifest_entry) {
            set_err(err, errlen, "FlashMoE manifest entry is missing");
            return NULL;
        }
        if (runtime_load_blob(&g_runtime,
                              manifest_entry,
                              expected_size,
                              err,
                              errlen) != 0) {
            return NULL;
        }
        entry = runtime_find_blob(&g_runtime, layer_id, expert_id);
        if (!entry) {
            set_err(err, errlen, "FlashMoE blob cache load failed");
            return NULL;
        }
    }
    entry->last_use = ++g_runtime.use_clock;
    if (actual_size) *actual_size = entry->size_bytes;
    return entry->bytes;
}
