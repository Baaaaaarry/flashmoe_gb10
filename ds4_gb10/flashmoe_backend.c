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

static bool manifest_push_layer(ds4_flashmoe_manifest *manifest,
                                ds4_flashmoe_layer_pack *layer,
                                size_t *cap) {
    if (manifest->count == *cap) {
        size_t new_cap = *cap ? *cap * 2u : 64u;
        void *new_layers = realloc(manifest->layers, new_cap * sizeof(manifest->layers[0]));
        if (!new_layers) return false;
        manifest->layers = (ds4_flashmoe_layer_pack *)new_layers;
        *cap = new_cap;
    }
    manifest->layers[manifest->count++] = *layer;
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

    uint64_t version = 0;
    uint64_t layout_version = 0;
    (void)parse_json_u64_field(text, "version", &version);
    (void)parse_json_u64_field(text, "layout_version", &layout_version);
    out->version = version != 0 ? (uint32_t)version : 1u;
    out->layout_version = layout_version != 0 ? (uint32_t)layout_version : 1u;

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

        uint64_t layer = 0;
        uint64_t expert_size = 0;
        uint64_t num_experts = 0;
        uint64_t gate_bytes = 0, up_bytes = 0, down_bytes = 0;
        uint64_t gate_row_bytes = 0, up_row_bytes = 0, down_row_bytes = 0;
        char *entry_path = NULL;
        bool ok = parse_json_u64_field(obj, "layer_id", &layer) &&
                  parse_json_u64_field(obj, "expert_size", &expert_size) &&
                  parse_json_u64_field(obj, "num_experts", &num_experts) &&
                  parse_json_u64_field(obj, "gate_bytes", &gate_bytes) &&
                  parse_json_u64_field(obj, "up_bytes", &up_bytes) &&
                  parse_json_u64_field(obj, "down_bytes", &down_bytes) &&
                  parse_json_u64_field(obj, "gate_row_bytes", &gate_row_bytes) &&
                  parse_json_u64_field(obj, "up_row_bytes", &up_row_bytes) &&
                  parse_json_u64_field(obj, "down_row_bytes", &down_row_bytes);
        if (ok) entry_path = dup_json_string_field(obj, "path");
        if (ok && entry_path) {
            if (layer >= max_layers || num_experts == 0 || num_experts > max_experts) {
                free(entry_path);
                free(obj);
                free(text);
                ds4_flashmoe_manifest_free(out);
                set_err(err, errlen, "manifest layer pack is outside ds4 model bounds");
                return 1;
            }
            if (expert_size == 0 || gate_bytes + up_bytes + down_bytes != expert_size) {
                free(entry_path);
                free(obj);
                free(text);
                ds4_flashmoe_manifest_free(out);
                set_err(err, errlen, "manifest layer pack layout is inconsistent");
                return 1;
            }
            ds4_flashmoe_layer_pack entry;
            memset(&entry, 0, sizeof(entry));
            entry.layer_id = (uint16_t)layer;
            entry.path = entry_path;
            entry.expert_size = expert_size;
            entry.num_experts = num_experts;
            entry.gate_bytes = gate_bytes;
            entry.up_bytes = up_bytes;
            entry.down_bytes = down_bytes;
            entry.gate_row_bytes = gate_row_bytes;
            entry.up_row_bytes = up_row_bytes;
            entry.down_row_bytes = down_row_bytes;
            if (!manifest_push_layer(out, &entry, &cap)) {
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
        set_err(err, errlen, "no manifest layer packs parsed");
        return 1;
    }

    for (size_t i = 0; i < out->count; i++) {
        for (size_t j = i + 1; j < out->count; j++) {
            const ds4_flashmoe_layer_pack *a = &out->layers[i];
            const ds4_flashmoe_layer_pack *b = &out->layers[j];
            if (a->layer_id == b->layer_id) {
                ds4_flashmoe_manifest_free(out);
                set_err(err, errlen, "duplicate layer pack entry in manifest");
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
        layer_counts[out->layers[i].layer_id] = (size_t)out->layers[i].num_experts;
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
    if (!manifest || !manifest->layers || manifest->count == 0) {
        set_err(err, errlen, "manifest has no layer packs");
        return 1;
    }
    for (size_t i = 0; i < manifest->count; i++) {
        const ds4_flashmoe_layer_pack *entry = &manifest->layers[i];
        struct stat st;
        if (stat(entry->path, &st) != 0) {
            set_err(err, errlen, "manifest layer pack file is missing");
            return 1;
        }
        if ((uint64_t)st.st_size < entry->expert_size * entry->num_experts) {
            set_err(err, errlen, "manifest layer pack size exceeds file bounds");
            return 1;
        }
    }
    return 0;
}

void ds4_flashmoe_manifest_free(ds4_flashmoe_manifest *manifest) {
    if (!manifest) return;
    if (manifest->layers) {
        for (size_t i = 0; i < manifest->count; i++) {
            free(manifest->layers[i].path);
        }
    }
    free(manifest->layers);
    memset(manifest, 0, sizeof(*manifest));
}

const ds4_flashmoe_layer_pack *ds4_flashmoe_manifest_find_layer(
        const ds4_flashmoe_manifest *manifest,
        uint16_t layer_id) {
    if (!manifest) return NULL;
    for (size_t i = 0; i < manifest->count; i++) {
        const ds4_flashmoe_layer_pack *entry = &manifest->layers[i];
        if (entry->layer_id == layer_id) {
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
    int *layer_fds;
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
    if (rt->layer_fds && rt->manifest) {
        for (size_t i = 0; i < rt->manifest->count; i++) {
            if (rt->layer_fds[i] >= 0) close(rt->layer_fds[i]);
        }
    }
    free(rt->layer_fds);
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

static bool read_fully_at(int fd, uint8_t *dst, uint64_t bytes, uint64_t offset) {
    uint64_t done = 0;
    while (done < bytes) {
        ssize_t n = pread(fd,
                          dst + done,
                          (size_t)(bytes - done),
                          (off_t)(offset + done));
        if (n <= 0) return false;
        done += (uint64_t)n;
    }
    return true;
}

static int runtime_layer_fd(ds4_flashmoe_runtime_state *rt,
                            const ds4_flashmoe_layer_pack *layer_pack) {
    if (!rt || !rt->manifest || !rt->layer_fds || !layer_pack) return -1;
    for (size_t i = 0; i < rt->manifest->count; i++) {
        if (&rt->manifest->layers[i] != layer_pack) continue;
        if (rt->layer_fds[i] >= 0) return rt->layer_fds[i];
        rt->layer_fds[i] = open(layer_pack->path, O_RDONLY);
        return rt->layer_fds[i];
    }
    return -1;
}

static int runtime_load_blob(ds4_flashmoe_runtime_state *rt,
                             const ds4_flashmoe_layer_pack *layer_pack,
                             uint16_t expert_id,
                             uint64_t expected_size,
                             char *err,
                             size_t errlen) {
    if (!rt || !layer_pack) {
        set_err(err, errlen, "invalid runtime layer pack entry");
        return 1;
    }
    if (layer_pack->expert_size != expected_size) {
        snprintf(err,
                 errlen,
                 "FlashMoE manifest size mismatch for layer=%u expert=%u (manifest=%" PRIu64 " expected=%" PRIu64 ")",
                 (unsigned)layer_pack->layer_id,
                 (unsigned)expert_id,
                 layer_pack->expert_size,
                 expected_size);
        return 1;
    }
    if (expert_id >= layer_pack->num_experts) {
        set_err(err, errlen, "FlashMoE expert id exceeds packed layer range");
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
    int fd = runtime_layer_fd(rt, layer_pack);
    if (fd < 0) {
        set_err(err, errlen, "failed to open FlashMoE layer pack");
        return 1;
    }
    uint8_t *bytes = (uint8_t *)malloc((size_t)expected_size);
    if (!bytes) {
        close(fd);
        set_err(err, errlen, "out of memory allocating FlashMoE expert blob");
        return 1;
    }
    const uint64_t base_offset = (uint64_t)expert_id * layer_pack->expert_size;
    if (!read_fully_at(fd, bytes, expected_size, base_offset)) {
        free(bytes);
        set_err(err, errlen, "failed to read FlashMoE expert blob");
        return 1;
    }
    if (!runtime_reserve_slot(rt)) {
        free(bytes);
        set_err(err, errlen, "out of memory growing FlashMoE blob cache");
        return 1;
    }
    ds4_flashmoe_blob_entry *entry = &rt->entries[rt->count++];
    entry->layer_id = layer_pack->layer_id;
    entry->expert_id = expert_id;
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
    if (!manifest || !manifest->layers || manifest->count == 0) {
        set_err(err, errlen, "FlashMoE runtime needs a non-empty manifest");
        return 1;
    }
    g_runtime.manifest = manifest;
    g_runtime.layer_fds = (int *)malloc(manifest->count * sizeof(g_runtime.layer_fds[0]));
    if (!g_runtime.layer_fds) {
        set_err(err, errlen, "out of memory allocating FlashMoE layer fd table");
        memset(&g_runtime, 0, sizeof(g_runtime));
        return 1;
    }
    for (size_t i = 0; i < manifest->count; i++) g_runtime.layer_fds[i] = -1;
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
        const ds4_flashmoe_layer_pack *layer_pack =
                ds4_flashmoe_manifest_find_layer(g_runtime.manifest, layer_id);
        if (!layer_pack) {
            set_err(err, errlen, "FlashMoE layer pack entry is missing");
            return NULL;
        }
        if (runtime_load_blob(&g_runtime,
                              layer_pack,
                              expert_id,
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

int ds4_flashmoe_runtime_load_selected_pack(uint16_t layer_id,
                                            const uint16_t *expert_ids,
                                            uint32_t n_experts,
                                            bool prefer_cache,
                                            uint8_t *gate_dst,
                                            uint8_t *up_dst,
                                            uint8_t *down_dst,
                                            uint64_t gate_bytes,
                                            uint64_t up_bytes,
                                            uint64_t down_bytes,
                                            char *err,
                                            size_t errlen) {
    if (!ds4_flashmoe_runtime_ready()) {
        set_err(err, errlen, "FlashMoE runtime cache is not ready");
        return 1;
    }
    const ds4_flashmoe_layer_pack *layer_pack =
            ds4_flashmoe_manifest_find_layer(g_runtime.manifest, layer_id);
    if (!layer_pack) {
        set_err(err, errlen, "FlashMoE layer pack entry is missing");
        return 1;
    }
    if (layer_pack->gate_bytes != gate_bytes ||
        layer_pack->up_bytes != up_bytes ||
        layer_pack->down_bytes != down_bytes) {
        set_err(err, errlen, "FlashMoE layer pack layout does not match runtime expectation");
        return 1;
    }
    const uint64_t expert_size = layer_pack->expert_size;
    int fd = -1;
    uint8_t *scratch = NULL;
    uint64_t scratch_bytes = 0;
    if (!prefer_cache) {
        fd = runtime_layer_fd(&g_runtime, layer_pack);
        if (fd < 0) {
            set_err(err, errlen, "failed to open FlashMoE layer pack");
            return 1;
        }
    }
    uint32_t i = 0;
    while (i < n_experts) {
        if (expert_ids[i] >= layer_pack->num_experts) {
            free(scratch);
            set_err(err, errlen, "FlashMoE selected expert exceeds packed layer range");
            return 1;
        }
        if (prefer_cache) {
            const uint8_t *blob = NULL;
            ds4_flashmoe_blob_entry *entry =
                    runtime_find_blob(&g_runtime, layer_id, expert_ids[i]);
            if (!entry) {
                if (runtime_load_blob(&g_runtime,
                                      layer_pack,
                                      expert_ids[i],
                                      expert_size,
                                      err,
                                      errlen) != 0) {
                    return 1;
                }
                entry = runtime_find_blob(&g_runtime, layer_id, expert_ids[i]);
            }
            if (!entry || entry->size_bytes != expert_size || !entry->bytes) {
                set_err(err, errlen, "FlashMoE blob cache load failed");
                return 1;
            }
            entry->last_use = ++g_runtime.use_clock;
            blob = entry->bytes;
            uint8_t *gate_ptr = gate_dst + (uint64_t)i * gate_bytes;
            uint8_t *up_ptr = up_dst + (uint64_t)i * up_bytes;
            uint8_t *down_ptr = down_dst + (uint64_t)i * down_bytes;
            memcpy(gate_ptr, blob, (size_t)gate_bytes);
            memcpy(up_ptr, blob + gate_bytes, (size_t)up_bytes);
            memcpy(down_ptr, blob + gate_bytes + up_bytes, (size_t)down_bytes);
            i++;
            continue;
        }

        uint32_t run_end = i + 1u;
        while (run_end < n_experts &&
               expert_ids[run_end] < layer_pack->num_experts &&
               expert_ids[run_end] == (uint16_t)(expert_ids[run_end - 1] + 1u)) {
            run_end++;
        }
        const uint32_t run_count = run_end - i;
        const uint64_t run_bytes = (uint64_t)run_count * expert_size;
        if (run_bytes > scratch_bytes) {
            uint8_t *new_scratch = (uint8_t *)realloc(scratch, (size_t)run_bytes);
            if (!new_scratch) {
                free(scratch);
                set_err(err, errlen, "out of memory growing FlashMoE direct read buffer");
                return 1;
            }
            scratch = new_scratch;
            scratch_bytes = run_bytes;
        }
        const uint64_t base = (uint64_t)expert_ids[i] * expert_size;
        if (!read_fully_at(fd, scratch, run_bytes, base)) {
            free(scratch);
            set_err(err, errlen, "failed to read FlashMoE expert blob run");
            return 1;
        }
        for (uint32_t j = 0; j < run_count; j++) {
            const uint8_t *blob = scratch + (uint64_t)j * expert_size;
            uint8_t *gate_ptr = gate_dst + (uint64_t)(i + j) * gate_bytes;
            uint8_t *up_ptr = up_dst + (uint64_t)(i + j) * up_bytes;
            uint8_t *down_ptr = down_dst + (uint64_t)(i + j) * down_bytes;
            memcpy(gate_ptr, blob, (size_t)gate_bytes);
            memcpy(up_ptr, blob + gate_bytes, (size_t)up_bytes);
            memcpy(down_ptr, blob + gate_bytes + up_bytes, (size_t)down_bytes);
        }
        i = run_end;
    }
    free(scratch);
    return 0;
}
