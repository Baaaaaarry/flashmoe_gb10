#include "ds4.h"
#include "linenoise.h"
#include "ds4_gpu.h"

/* ds4 CLI.
 *
 * One-shot mode builds a single DeepSeek chat prompt and exits.  Interactive
 * mode keeps a rendered token transcript plus one ds4_session, so follow-up
 * turns reuse the live Metal KV checkpoint just like the server does.  The CLI
 * deliberately keeps policy here and leaves graph/cache mechanics inside the
 * engine API. */

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    const char *prompt;
    const char *system;
    int n_predict;
    int ctx_size;
    float temperature;
    float top_p;
    uint64_t seed;
    bool dump_tokens;
    const char *dump_logprobs_path;
    int dump_logprobs_top_k;
    const char *imatrix_dataset_path;
    const char *imatrix_output_path;
    int imatrix_max_prompts;
    int imatrix_max_tokens;
    ds4_think_mode think_mode;
    bool head_test;
    bool first_token_test;
    bool metal_graph_test;
    bool metal_graph_full_test;
    bool metal_graph_prompt_test;
} cli_generation_options;

typedef struct {
    ds4_engine_options engine;
    cli_generation_options gen;
    char *prompt_owned;
    bool inspect;
    bool runtime_monitor;
    float runtime_monitor_interval;
    const char *flashmoe_export_dir;
    const char *flashmoe_export_manifest;
} cli_config;

typedef struct {
    bool enabled;
    float interval_s;
    volatile int stop;
    bool started;
    pthread_t thread;
    pthread_mutex_t mu;
    const char *phase;
    int current;
    int total;
    ds4_backend backend;
} cli_runtime_monitor;

static double cli_now_sec(void);

static cli_runtime_monitor *g_cli_runtime_monitor;

static volatile sig_atomic_t cli_interrupted;

static void cli_sigint_handler(int sig) {
    (void)sig;
    cli_interrupted = 1;
}

static bool cli_interrupt_requested(void) {
    return cli_interrupted != 0;
}

static void cli_interrupt_clear(void) {
    cli_interrupted = 0;
}

static void usage(FILE *fp) {
    fprintf(fp,
        "Usage: ds4 [(-p PROMPT | --prompt-file FILE)] [options]\n"
        "\n"
        "Invocation modes:\n"
        "  ds4\n"
        "      Start the interactive chat prompt with a session backend: ds4>\n"
        "  ds4 -p TEXT\n"
        "      Run one prompt and exit.\n"
        "  ds4 --prompt-file FILE\n"
        "      Run one prompt read from FILE and exit. Useful for long prompts.\n"
        "\n"
        "Model and runtime:\n"
        "  -m, --model FILE\n"
        "      GGUF model path. Default: ds4flash.gguf\n"
        "  --mtp FILE\n"
        "      Optional MTP support GGUF used for draft-token probes.\n"
        "  --mtp-draft N\n"
        "      Maximum autoregressive MTP draft tokens per speculative step. Default: 1\n"
        "  --mtp-margin F\n"
        "      Minimum recursive-draft confidence for the fast N=2 verifier. Default: 3\n"
        "  -c, --ctx N\n"
        "      Context size allocated for the session. Default: 32768\n"
        "  --metal\n"
        "      Use the Metal graph backend. This is the normal fast path on macOS.\n"
        "  --cuda\n"
        "      Use the CUDA graph backend. This is the normal fast path on CUDA builds.\n"
        "  --cpu\n"
        "      Use the CPU reference/debug backend. Not recommended for normal inference.\n"
        "  --backend NAME\n"
        "      Select backend explicitly: metal, cuda, or cpu.\n"
        "  -t, --threads N\n"
        "      CPU helper threads for host-side or reference work.\n"
        "  --quality\n"
        "      Prefer exact kernels where faster approximate paths exist; MTP uses strict verification.\n"
        "  --dir-steering-file FILE\n"
        "      Load one f32 direction vector per layer for directional steering.\n"
        "  --dir-steering-ffn F\n"
        "      Apply steering after FFN outputs: y -= F*v*dot(v,y). Default with file: 1\n"
        "  --dir-steering-attn F\n"
        "      Apply steering after attention outputs. Default: 0\n"
        "  --warm-weights\n"
        "      Touch mapped tensor pages before generation. Slower startup, fewer first-use stalls.\n"
        "\n"
        "Prompt and generation:\n"
        "  -p, --prompt TEXT\n"
        "      Prompt to generate from.\n"
        "  --prompt-file FILE\n"
        "      Read the prompt text from FILE.\n"
        "  -sys, --system TEXT\n"
        "      System prompt. Empty string disables the default. Default: You are a helpful assistant\n"
        "  -n, --tokens N\n"
        "      Maximum tokens to generate. Default: 50000\n"
        "  --temp F\n"
        "      Sampling temperature. 0 is greedy/deterministic. Default: 1\n"
        "  --top-p F\n"
        "      Nucleus sampling probability. Default: 1\n"
        "  --seed N\n"
        "      Sampling seed for reproducible non-greedy runs. Default: time-based\n"
        "  --think\n"
        "      Use normal thinking mode. This is the default.\n"
        "  --think-max\n"
        "      Use Think Max when --ctx is at least 393216 tokens; otherwise normal thinking.\n"
        "  --nothink\n"
        "      Start assistant turns with </think> for direct non-thinking replies.\n"
        "\n"
        "Interactive commands:\n"
        "  /help\n"
        "      Show interactive commands.\n"
        "  /think, /think-max, /nothink\n"
        "      Select normal thinking, context-gated Think Max, or non-thinking mode.\n"
        "  /ctx N\n"
        "      Recreate the interactive session with a new context size.\n"
        "  /read FILE\n"
        "      Read a prompt from FILE and run it as the next user message.\n"
        "  /quit, /exit\n"
        "      Leave the interactive prompt.\n"
        "  Ctrl+C\n"
        "      Stop the current generation and return to ds4> without exiting.\n"
        "\n"
        "Diagnostics:\n"
        "  --inspect\n"
        "      Load the model and print a summary only.\n"
        "  --runtime-monitor\n"
        "      Print periodic runtime metrics for RSS/HWM, process I/O, CUDA cache, and GPU load.\n"
        "  --runtime-monitor-interval F\n"
        "      Runtime monitor sampling interval in seconds. Default: 1.0\n"
        "  --flashmoe-export-dir DIR\n"
        "      Export routed experts into a FlashMoE pack directory.\n"
        "  --flashmoe-export-manifest FILE\n"
        "      Write the FlashMoE routed expert manifest JSON.\n"
        "  --dump-tokens\n"
        "      Tokenize -p/--prompt-file exactly as written, then exit without inference.\n"
        "  --dump-logprobs FILE\n"
        "      Write greedy continuation top-logprobs as JSON without printing text.\n"
        "  --logprobs-top-k N\n"
        "      Number of local alternatives stored by --dump-logprobs. Default: 20\n"
        "  --imatrix-dataset FILE\n"
        "      Rendered DS4 prompt dataset produced by misc/imatrix_dataset.\n"
        "  --imatrix-out FILE\n"
        "      Collect a routed-MoE activation imatrix and write llama-compatible .dat.\n"
        "  --imatrix-max-prompts N\n"
        "      Stop imatrix collection after N prompts. Default: no prompt limit\n"
        "  --imatrix-max-tokens N\n"
        "      Stop imatrix collection after N prompt tokens. Default: no token limit\n"
        "  --head-test\n"
        "      Run the output HC/logits head after the native slice.\n"
        "  --first-token-test\n"
        "      Run an exact CPU whole-model pass for the first prompt token.\n"
        "  --metal-graph-test\n"
        "      Compare first GPU-resident graph stages with CPU.\n"
        "  --metal-graph-full-test\n"
        "      Run the GPU-resident self-token graph across all layers.\n"
        "  --metal-graph-prompt-test\n"
        "      Compare CPU and GPU graph logits for the full prompt.\n"
        "\n"
        "Normal CLI commands:\n"
        "  ./ds4\n"
        "  ./ds4 -p \"Scrivi una storia su una papera scansafatiche\"\n"
        "  ./ds4 --think-max --prompt-file prompt.txt --ctx 393216\n"
        "\n"
        "Notes:\n"
        "  The CLI keeps KV cache state across interactive turns on session backends.\n"
        "  CPU mode supports interactive chat too, but it is a slow reference/debug path.\n"
        "  Long added input is processed with batched prefill; short continuations use decode.\n"
        "  Startup prints the extra context-buffer memory for the selected context size.\n"
        "\n"
        "  -h, --help\n"
        "      Show this help.\n");
}

static int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT32_MAX) {
        fprintf(stderr, "ds4: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static uint64_t parse_u64(const char *s, const char *opt) {
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v == 0) {
        fprintf(stderr, "ds4: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (uint64_t)v;
}

static float parse_float_range(const char *s, const char *opt, float min, float max) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v) || v < min || v > max) {
        fprintf(stderr, "ds4: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static ds4_backend parse_backend(const char *s) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4: invalid backend: %s\n", s);
    fprintf(stderr, "ds4: valid backends are: metal, cuda, cpu\n");
    exit(2);
}

static ds4_backend default_backend(void) {
#ifdef DS4_NO_GPU
    return DS4_BACKEND_CPU;
#elif defined(__APPLE__)
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static void log_context_memory(ds4_backend backend, int ctx_size) {
    ds4_context_memory m = ds4_context_memory_estimate(backend, ctx_size);
    fprintf(stderr,
            "ds4: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)\n",
            (double)m.total_bytes / (1024.0 * 1024.0),
            ctx_size,
            ds4_backend_name(backend),
            m.prefill_cap,
            m.raw_cap,
            m.comp_cap);
}

static uint64_t cli_read_status_kib(const char *key) {
    FILE *fp = fopen("/proc/self/status", "rb");
    if (!fp) return 0;
    char line[256];
    uint64_t value = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, key, strlen(key)) == 0) {
            unsigned long long kib = 0;
            if (sscanf(line + strlen(key), ": %llu kB", &kib) == 1) {
                value = (uint64_t)kib;
                break;
            }
        }
    }
    fclose(fp);
    return value;
}

static bool cli_read_proc_io(uint64_t *read_bytes, uint64_t *write_bytes) {
    FILE *fp = fopen("/proc/self/io", "rb");
    if (!fp) return false;
    char line[256];
    uint64_t r = 0, w = 0;
    while (fgets(line, sizeof(line), fp)) {
        unsigned long long v = 0;
        if (sscanf(line, "read_bytes: %llu", &v) == 1) r = (uint64_t)v;
        else if (sscanf(line, "write_bytes: %llu", &v) == 1) w = (uint64_t)v;
    }
    fclose(fp);
    if (read_bytes) *read_bytes = r;
    if (write_bytes) *write_bytes = w;
    return true;
}

static bool cli_query_nvidia_smi(
        int *gpu_util,
        int *mem_util,
        double *mem_used_mb,
        double *mem_total_mb,
        double *power_w) {
    FILE *fp = popen(
        "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits 2>/dev/null",
        "r");
    if (!fp) return false;
    char line[256];
    bool ok = false;
    if (fgets(line, sizeof(line), fp)) {
        int gu = -1, mu = -1;
        double used = 0.0, total = 0.0, power = 0.0;
        if (sscanf(line, " %d , %d , %lf , %lf , %lf", &gu, &mu, &used, &total, &power) == 5 ||
            sscanf(line, " %d, %d, %lf, %lf, %lf", &gu, &mu, &used, &total, &power) == 5) {
            if (gpu_util) *gpu_util = gu;
            if (mem_util) *mem_util = mu;
            if (mem_used_mb) *mem_used_mb = used;
            if (mem_total_mb) *mem_total_mb = total;
            if (power_w) *power_w = power;
            ok = true;
        }
    }
    (void)pclose(fp);
    return ok;
}

static void cli_runtime_monitor_set_phase(cli_runtime_monitor *m, const char *phase) {
    if (!m || !m->enabled) return;
    pthread_mutex_lock(&m->mu);
    m->phase = phase;
    pthread_mutex_unlock(&m->mu);
}

static void cli_runtime_monitor_set_progress(cli_runtime_monitor *m, int current, int total) {
    if (!m || !m->enabled) return;
    pthread_mutex_lock(&m->mu);
    m->current = current;
    m->total = total;
    pthread_mutex_unlock(&m->mu);
}

static void *cli_runtime_monitor_main(void *arg) {
    cli_runtime_monitor *m = arg;
    uint64_t last_read = 0, last_write = 0;
    bool have_io = cli_read_proc_io(&last_read, &last_write);
    double last_t = cli_now_sec();
    while (!m->stop) {
        const useconds_t sleep_us = (useconds_t)(m->interval_s > 0.01f ? m->interval_s * 1000000.0f : 1000000.0f);
        usleep(sleep_us);
        const double now = cli_now_sec();
        const double dt = now - last_t > 1.0e-6 ? now - last_t : 1.0;
        last_t = now;

        const uint64_t rss_kib = cli_read_status_kib("VmRSS");
        const uint64_t hwm_kib = cli_read_status_kib("VmHWM");
        uint64_t read_b = 0, write_b = 0;
        double read_mibs = 0.0, write_mibs = 0.0;
        if (cli_read_proc_io(&read_b, &write_b)) {
            if (have_io) {
                read_mibs = (double)(read_b - last_read) / dt / 1048576.0;
                write_mibs = (double)(write_b - last_write) / dt / 1048576.0;
            }
            have_io = true;
            last_read = read_b;
            last_write = write_b;
        }

        const char *phase = "idle";
        int current = 0, total = 0;
        pthread_mutex_lock(&m->mu);
        if (m->phase) phase = m->phase;
        current = m->current;
        total = m->total;
        pthread_mutex_unlock(&m->mu);

        char cuda_buf[256];
        cuda_buf[0] = '\0';
#ifndef DS4_NO_GPU
        if (m->backend == DS4_BACKEND_CUDA) {
            ds4_gpu_runtime_stats s;
            if (ds4_gpu_get_runtime_stats(&s)) {
                snprintf(cuda_buf,
                         sizeof(cuda_buf),
                         " cuda_model=%.2fGiB q8f16=%.2fGiB q8f32=%.2fGiB cuda_free=%.2fGiB cuda_total=%.2fGiB",
                         (double)s.model_cache_bytes / 1073741824.0,
                         (double)s.q8_f16_cache_bytes / 1073741824.0,
                         (double)s.q8_f32_cache_bytes / 1073741824.0,
                         (double)s.free_bytes / 1073741824.0,
                         (double)s.total_bytes / 1073741824.0);
            }
        }
#endif
        char smi_buf[192];
        smi_buf[0] = '\0';
        if (m->backend == DS4_BACKEND_CUDA) {
            int gpu_util = -1, mem_util = -1;
            double mem_used_mb = 0.0, mem_total_mb = 0.0, power_w = 0.0;
            if (cli_query_nvidia_smi(&gpu_util, &mem_util, &mem_used_mb, &mem_total_mb, &power_w)) {
                snprintf(smi_buf,
                         sizeof(smi_buf),
                         " gpu_util=%d%% gpu_mem_util=%d%% gpu_mem=%.2f/%.2fGiB power=%.1fW",
                         gpu_util,
                         mem_util,
                         mem_used_mb / 1024.0,
                         mem_total_mb / 1024.0,
                         power_w);
            }
        }

        fprintf(stderr,
                "ds4: runtime-monitor phase=%s progress=%d/%d rss=%.2fGiB hwm=%.2fGiB io_read=%.2fMiB/s io_write=%.2fMiB/s%s%s\n",
                phase,
                current,
                total,
                (double)rss_kib / 1048576.0,
                (double)hwm_kib / 1048576.0,
                read_mibs,
                write_mibs,
                cuda_buf,
                smi_buf);
    }
    return NULL;
}

static void cli_runtime_monitor_start(cli_runtime_monitor *m, const cli_config *cfg) {
    memset(m, 0, sizeof(*m));
    if (!cfg->runtime_monitor) return;
    m->enabled = true;
    m->interval_s = cfg->runtime_monitor_interval > 0.0f ? cfg->runtime_monitor_interval : 1.0f;
    m->backend = cfg->engine.backend;
    m->phase = "startup";
    pthread_mutex_init(&m->mu, NULL);
    if (pthread_create(&m->thread, NULL, cli_runtime_monitor_main, m) == 0) {
        m->started = true;
    } else {
        pthread_mutex_destroy(&m->mu);
        memset(m, 0, sizeof(*m));
    }
}

static void cli_runtime_monitor_stop(cli_runtime_monitor *m) {
    if (!m || !m->enabled) return;
    m->stop = 1;
    if (m->started) (void)pthread_join(m->thread, NULL);
    pthread_mutex_destroy(&m->mu);
    memset(m, 0, sizeof(*m));
}

static ds4_think_mode cli_effective_think_mode(const cli_generation_options *gen) {
    return ds4_think_mode_for_context(gen->think_mode, gen->ctx_size);
}

static bool cli_think_max_downgraded(const cli_generation_options *gen) {
    return gen->think_mode == DS4_THINK_MAX &&
           cli_effective_think_mode(gen) != DS4_THINK_MAX;
}

static void cli_warn_think_max_downgraded(const cli_generation_options *gen, const char *name) {
    if (!cli_think_max_downgraded(gen)) return;
    ds4_log(stderr,
        DS4_LOG_WARNING,
        "ds4: warning: %s needs --ctx >= %u; ctx=%d uses normal thinking instead\n",
        name,
        ds4_think_max_min_context(),
        gen->ctx_size);
}

static double cli_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static char *read_prompt_file(const char *path, bool fatal);

typedef struct {
    int base_tokens;
    int input_tokens;
    bool use_color;
    cli_runtime_monitor *monitor;
} cli_prefill_progress;

static void cli_prefill_progress_cb(void *ud, const char *event, int current, int total) {
    (void)total;
    cli_prefill_progress *p = ud;
    if (!p || !event || strcmp(event, "prefill_chunk") || p->input_tokens <= 0) return;

    int processed = current - p->base_tokens;
    if (processed < 0) processed = 0;
    if (processed > p->input_tokens) processed = p->input_tokens;
    double pct = 100.0 * (double)processed / (double)p->input_tokens;
    if (pct > 100.0) pct = 100.0;
    cli_runtime_monitor_set_phase(p->monitor, "prefill");
    cli_runtime_monitor_set_progress(p->monitor, processed, p->input_tokens);

    if (p->use_color) {
        fputc('\r', stderr);
        ds4_log(stderr,
                DS4_LOG_PREFILL,
                "processing %d input tokens: %d/%d (%.1f%%)",
                p->input_tokens,
                processed,
                p->input_tokens,
                pct);
        fputs("\x1b[K", stderr);
        if (processed >= p->input_tokens) fputc('\n', stderr);
    } else {
        fprintf(stderr,
                "processing %d input tokens: %d/%d (%.1f%%)\n",
                p->input_tokens,
                processed,
                p->input_tokens,
                pct);
    }
    fflush(stderr);
}

static bool is_rendered_chat_prompt(const char *prompt) {
    const char *bos = "<｜begin▁of▁sentence｜>";
    return prompt && strncmp(prompt, bos, strlen(bos)) == 0;
}

typedef struct {
    ds4_engine *engine;
    FILE *fp;
    bool format_thinking;
    bool in_think;
    bool color_open;
    bool use_color;
    bool last_output_newline;
    char pending[16];
    size_t pending_len;
} token_printer;

static bool bytes_has_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n >= plen && memcmp(p, prefix, plen) == 0;
}

static bool bytes_is_partial_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n < plen && memcmp(prefix, p, n) == 0;
}

static void token_printer_set_grey(token_printer *p) {
    if (p->use_color && !p->color_open) {
        fputs("\x1b[90m", p->fp);
        p->color_open = true;
    }
}

static void token_printer_reset_color(token_printer *p) {
    if (p->use_color && p->color_open) {
        fputs("\x1b[0m", p->fp);
        p->color_open = false;
    }
}

static void token_printer_write_char(token_printer *p, char c) {
    if (p->in_think) token_printer_set_grey(p);
    fputc((unsigned char)c, p->fp);
    p->last_output_newline = c == '\n';
}

static void token_printer_process(token_printer *p, const char *text, size_t len, bool finish) {
    const char *think_open = "<think>";
    const char *think_close = "</think>";
    size_t total = p->pending_len + len;
    char *buf = malloc(total ? total : 1);
    if (!buf) return;
    if (p->pending_len) memcpy(buf, p->pending, p->pending_len);
    if (len) memcpy(buf + p->pending_len, text, len);
    p->pending_len = 0;

    size_t i = 0;
    while (i < total) {
        const char *cur = buf + i;
        const size_t rem = total - i;
        if (bytes_has_prefix(cur, rem, think_open)) {
            p->in_think = true;
            i += strlen(think_open);
            continue;
        }
        if (bytes_has_prefix(cur, rem, think_close)) {
            p->in_think = false;
            token_printer_reset_color(p);
            if (!p->last_output_newline) {
                fputc('\n', p->fp);
                p->last_output_newline = true;
            }
            i += strlen(think_close);
            continue;
        }
        if (!finish && cur[0] == '<' &&
            (bytes_is_partial_prefix(cur, rem, think_open) ||
             bytes_is_partial_prefix(cur, rem, think_close)))
        {
            if (rem < sizeof(p->pending)) {
                memcpy(p->pending, cur, rem);
                p->pending_len = rem;
            }
            break;
        }
        token_printer_write_char(p, cur[0]);
        i++;
    }

    free(buf);
}

static void token_printer_finish(token_printer *p) {
    if (p->format_thinking) {
        token_printer_process(p, NULL, 0, true);
        token_printer_reset_color(p);
    }
    fflush(p->fp);
}

static void generation_done(void *ud) {
    token_printer *p = ud;
    token_printer_finish(p);
    if (!p->last_output_newline) {
        fputc('\n', p->fp);
        p->last_output_newline = true;
    }
    fflush(p->fp);
}

static void token_printer_write_text(token_printer *p, const char *text, size_t len) {
    if (p->format_thinking) {
        token_printer_process(p, text, len, false);
    } else if (len) {
        fwrite(text, 1, len, p->fp);
        p->last_output_newline = text[len - 1] == '\n';
    }
}

static void print_generated_token(void *ud, int token) {
    token_printer *p = ud;
    size_t len = 0;
    char *text = ds4_token_text(p->engine, token, &len);
    token_printer_write_text(p, text, len);
    fflush(p->fp);
    free(text);
}

typedef struct {
    token_printer *printer;
    cli_runtime_monitor *monitor;
    int generated;
    int prompt_tokens;
    int max_tokens;
    double t0;
    double t_first;
    double t1;
} cli_emit_wrapper;

static void cli_print_generated_token_monitored(void *ud, int token) {
    cli_emit_wrapper *w = ud;
    if (w) {
        if (w->generated == 0) w->t_first = cli_now_sec();
        w->generated++;
    }
    if (w && w->monitor) {
        cli_runtime_monitor_set_phase(w->monitor, "generation");
        cli_runtime_monitor_set_progress(w->monitor, w->generated, w->max_tokens);
    }
    if (w && w->printer) print_generated_token(w->printer, token);
}

static void cli_generation_done_monitored(void *ud) {
    cli_emit_wrapper *w = ud;
    if (!w) return;
    w->t1 = cli_now_sec();
    if (w->monitor) {
        cli_runtime_monitor_set_phase(w->monitor, "generation");
        cli_runtime_monitor_set_progress(w->monitor, w->generated, w->max_tokens);
    }
    if (w->printer) generation_done(w->printer);
}

static void build_prompt(ds4_engine *engine, const cli_generation_options *gen, ds4_tokens *out) {
    if (is_rendered_chat_prompt(gen->prompt)) {
        ds4_tokenize_rendered_chat(engine, gen->prompt, out);
    } else {
        ds4_encode_chat_prompt(engine, gen->system, gen->prompt,
                               cli_effective_think_mode(gen), out);
    }
}

static int run_sampled_generation(ds4_engine *engine, const cli_config *cfg, const ds4_tokens *prompt) {
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg->gen.ctx_size) != 0) {
        fprintf(stderr, "ds4: sampled CLI generation requires a session backend\n");
        return 1;
    }

    char err[160];
    ds4_think_mode think_mode = cli_effective_think_mode(&cfg->gen);
    token_printer printer = {
        .engine = engine,
        .fp = stdout,
        .format_thinking = ds4_think_mode_enabled(think_mode),
        .in_think = ds4_think_mode_enabled(think_mode),
        .use_color = isatty(fileno(stdout)) != 0,
        .last_output_newline = true,
    };
    cli_prefill_progress progress = {
        .base_tokens = 0,
        .input_tokens = prompt->len,
        .use_color = ds4_log_is_tty(stderr),
        .monitor = g_cli_runtime_monitor,
    };

    const double t_prefill0 = cli_now_sec();
    ds4_session_set_progress(session, cli_prefill_progress_cb, &progress);
    if (ds4_session_sync(session, prompt, err, sizeof(err)) != 0) {
        ds4_session_set_progress(session, NULL, NULL);
        fprintf(stderr, "ds4: prompt processing failed: %s\n", err);
        ds4_session_free(session);
        return 1;
    }
    ds4_session_set_progress(session, NULL, NULL);
    const double t_prefill1 = cli_now_sec();

    int max_tokens = cfg->gen.n_predict;
    int room = ds4_session_ctx(session) - ds4_session_pos(session);
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;

    uint64_t rng = cfg->gen.seed ? cfg->gen.seed :
        ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());
    int generated = 0;
    const double t_decode0 = cli_now_sec();
    cli_runtime_monitor_set_phase(g_cli_runtime_monitor, "generation");
    cli_runtime_monitor_set_progress(g_cli_runtime_monitor, 0, max_tokens);
    while (generated < max_tokens && !cli_interrupt_requested()) {
        int token = ds4_session_sample(session, cfg->gen.temperature, 0, cfg->gen.top_p, 0.0f, &rng);
        if (token == ds4_token_eos(engine)) break;

        int toks[17];
        int ntok = 0;
        if (cfg->gen.temperature <= 0.0f && ds4_engine_mtp_draft_tokens(engine) > 1 &&
            getenv("DS4_MTP_SPEC_DISABLE") == NULL) {
            ntok = ds4_session_eval_speculative_argmax(session,
                                                       token,
                                                       max_tokens - generated,
                                                       ds4_token_eos(engine),
                                                       toks,
                                                       (int)(sizeof(toks) / sizeof(toks[0])),
                                                       err,
                                                       sizeof(err));
            if (ntok < 0) {
                fprintf(stderr, "ds4: decode failed: %s\n", err);
                ds4_session_free(session);
                return 1;
            }
        } else {
            if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4: decode failed: %s\n", err);
                ds4_session_free(session);
                return 1;
            }
            toks[0] = token;
            ntok = 1;
        }

        bool stop = false;
        for (int j = 0; j < ntok; j++) {
            if (toks[j] == ds4_token_eos(engine)) {
                stop = true;
                break;
            }
            size_t piece_len = 0;
            char *piece = ds4_token_text(engine, toks[j], &piece_len);
            token_printer_write_text(&printer, piece, piece_len);
            fflush(stdout);
            free(piece);
            generated++;
            cli_runtime_monitor_set_phase(g_cli_runtime_monitor, "generation");
            cli_runtime_monitor_set_progress(g_cli_runtime_monitor, generated, max_tokens);
            if (generated >= max_tokens) break;
        }
        if (stop) break;
    }
    const double t_decode1 = cli_now_sec();
    generation_done(&printer);
    if (cli_interrupt_requested()) cli_interrupt_clear();

    const double prefill_s = t_prefill1 - t_prefill0;
    const double decode_s = t_decode1 - t_decode0;
    ds4_log(stderr,
            DS4_LOG_TIMING,
            "ds4: prefill: %.2f t/s, generation: %.2f t/s\n",
            prefill_s > 0.0 ? (double)prompt->len / prefill_s : 0.0,
            decode_s > 0.0 ? (double)generated / decode_s : 0.0);

    ds4_session_free(session);
    return 0;
}

static bool json_utf8_valid(const char *s, size_t n) {
    size_t i = 0;
    while (i < n) {
        unsigned char c = (unsigned char)s[i++];
        if (c < 0x80) continue;
        int need = 0;
        if (c >= 0xc2 && c <= 0xdf) need = 1;
        else if (c >= 0xe0 && c <= 0xef) need = 2;
        else if (c >= 0xf0 && c <= 0xf4) need = 3;
        else return false;
        if (i + (size_t)need > n) return false;
        unsigned char c1 = (unsigned char)s[i];
        if (c == 0xe0 && c1 < 0xa0) return false;
        if (c == 0xed && c1 >= 0xa0) return false;
        if (c == 0xf0 && c1 < 0x90) return false;
        if (c == 0xf4 && c1 >= 0x90) return false;
        for (int j = 0; j < need; j++) {
            unsigned char cc = (unsigned char)s[i + (size_t)j];
            if ((cc & 0xc0) != 0x80) return false;
        }
        i += (size_t)need;
    }
    return true;
}

static void json_write_string(FILE *fp, const char *s, size_t n) {
    bool valid_utf8 = json_utf8_valid(s, n);
    fputc('"', fp);
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') {
            fputc('\\', fp);
            fputc((char)c, fp);
        } else if (c == '\n') {
            fputs("\\n", fp);
        } else if (c == '\r') {
            fputs("\\r", fp);
        } else if (c == '\t') {
            fputs("\\t", fp);
        } else if (c < 0x20) {
            fprintf(fp, "\\u%04x", (unsigned)c);
        } else if (!valid_utf8 && c >= 0x80) {
            /* Tokenizer pieces can be arbitrary byte fragments.  The bytes
             * array is authoritative; this escape keeps the JSON valid. */
            fprintf(fp, "\\u%04x", (unsigned)c);
        } else {
            fputc((char)c, fp);
        }
    }
    fputc('"', fp);
}

static void json_write_token(FILE *fp, ds4_engine *engine, int token) {
    size_t n = 0;
    char *text = ds4_token_text(engine, token, &n);
    fprintf(fp, "{\"id\":%d,\"text\":", token);
    json_write_string(fp, text, n);
    fputs(",\"bytes\":[", fp);
    for (size_t i = 0; i < n; i++) {
        if (i) fputc(',', fp);
        fprintf(fp, "%u", (unsigned)(unsigned char)text[i]);
    }
    fputc(']', fp);
    fputc('}', fp);
    free(text);
}

static int run_logprob_dump(ds4_engine *engine, const cli_config *cfg, const ds4_tokens *prompt) {
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg->gen.ctx_size) != 0) {
        fprintf(stderr, "ds4: --dump-logprobs requires a graph session backend\n");
        return 1;
    }

    char err[160];
    cli_prefill_progress progress = {
        .base_tokens = 0,
        .input_tokens = prompt->len,
        .use_color = ds4_log_is_tty(stderr),
        .monitor = g_cli_runtime_monitor,
    };
    ds4_session_set_progress(session, cli_prefill_progress_cb, &progress);
    if (ds4_session_sync(session, prompt, err, sizeof(err)) != 0) {
        ds4_session_set_progress(session, NULL, NULL);
        fprintf(stderr, "ds4: prompt processing failed: %s\n", err);
        ds4_session_free(session);
        return 1;
    }
    ds4_session_set_progress(session, NULL, NULL);

    FILE *fp = fopen(cfg->gen.dump_logprobs_path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open --dump-logprobs file: %s\n", cfg->gen.dump_logprobs_path);
        ds4_session_free(session);
        return 1;
    }

    int k = cfg->gen.dump_logprobs_top_k > 0 ? cfg->gen.dump_logprobs_top_k : 20;
    if (k > 128) k = 128;
    ds4_token_score *scores = calloc((size_t)k, sizeof(scores[0]));
    if (!scores) {
        fclose(fp);
        ds4_session_free(session);
        return 1;
    }

    fprintf(fp, "{\n  \"source\":\"ds4\",\n  \"prompt_tokens\":%d,\n  \"ctx\":%d,\n  \"top_k\":%d,\n  \"steps\":[\n",
            prompt->len, cfg->gen.ctx_size, k);
    int generated = 0;
    int max_tokens = cfg->gen.n_predict;
    int room = ds4_session_ctx(session) - ds4_session_pos(session);
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;
    cli_runtime_monitor_set_phase(g_cli_runtime_monitor, "generation");
    cli_runtime_monitor_set_progress(g_cli_runtime_monitor, 0, max_tokens);
    for (; generated < max_tokens; generated++) {
        int n = ds4_session_top_logprobs(session, scores, k);
        int token = ds4_session_argmax(session);
        if (generated) fputs(",\n", fp);
        fprintf(fp, "    {\"step\":%d,\"selected\":", generated);
        json_write_token(fp, engine, token);
        fputs(",\"top_logprobs\":[", fp);
        for (int i = 0; i < n && scores[i].id >= 0; i++) {
            if (i) fputc(',', fp);
            fputs("{\"token\":", fp);
            json_write_token(fp, engine, scores[i].id);
            fprintf(fp, ",\"logit\":%.9g,\"logprob\":%.9g}", scores[i].logit, scores[i].logprob);
        }
        fputs("]}", fp);

        if (token == ds4_token_eos(engine)) break;
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4: decode failed while dumping logprobs: %s\n", err);
            free(scores);
            fclose(fp);
            ds4_session_free(session);
            return 1;
        }
        cli_runtime_monitor_set_phase(g_cli_runtime_monitor, "generation");
        cli_runtime_monitor_set_progress(g_cli_runtime_monitor, generated + 1, max_tokens);
    }
    fputs("\n  ]\n}\n", fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close --dump-logprobs file: %s\n", cfg->gen.dump_logprobs_path);
        free(scores);
        ds4_session_free(session);
        return 1;
    }
    free(scores);
    ds4_session_free(session);
    return 0;
}

static int run_generation(ds4_engine *engine, const cli_config *cfg) {
    ds4_tokens prompt = {0};
    build_prompt(engine, &cfg->gen, &prompt);

    int rc = 0;
    if (cfg->gen.metal_graph_test) {
        rc = ds4_engine_metal_graph_test(engine, &prompt);
        ds4_tokens_free(&prompt);
        return rc;
    }
    if (cfg->gen.metal_graph_full_test) {
        rc = ds4_engine_metal_graph_full_test(engine, &prompt);
        ds4_tokens_free(&prompt);
        return rc;
    }
    if (cfg->gen.metal_graph_prompt_test) {
        rc = ds4_engine_metal_graph_prompt_test(engine, &prompt, cfg->gen.ctx_size);
        ds4_tokens_free(&prompt);
        return rc;
    }
    if (cfg->gen.dump_logprobs_path) {
        rc = run_logprob_dump(engine, cfg, &prompt);
        ds4_tokens_free(&prompt);
        return rc;
    }

    const bool diagnostic = cfg->gen.dump_tokens ||
                            cfg->gen.head_test ||
                            cfg->gen.first_token_test;
    if (cfg->gen.head_test) {
        rc = ds4_engine_head_test(engine, &prompt);
    }
    if (rc == 0 && cfg->gen.first_token_test) {
        rc = ds4_engine_first_token_test(engine, &prompt);
    }
    if (cfg->gen.dump_tokens) {
        ds4_engine_dump_tokens(engine, &prompt);
    }

    if (diagnostic) {
        if (rc == 0) {
            fprintf(stderr, "ds4: diagnostic run completed on the native %s path.\n",
                    ds4_backend_name(cfg->engine.backend));
        }
    } else if (cfg->gen.temperature > 0.0f || ds4_engine_mtp_draft_tokens(engine) > 1) {
        rc = run_sampled_generation(engine, cfg, &prompt);
    } else {
        token_printer printer = {
            .engine = engine,
            .fp = stdout,
            .format_thinking = ds4_think_mode_enabled(cli_effective_think_mode(&cfg->gen)),
            .in_think = ds4_think_mode_enabled(cli_effective_think_mode(&cfg->gen)),
            .use_color = isatty(fileno(stdout)) != 0,
            .last_output_newline = true,
        };
        cli_prefill_progress progress = {
            .base_tokens = 0,
            .input_tokens = prompt.len,
            .use_color = ds4_log_is_tty(stderr),
            .monitor = g_cli_runtime_monitor,
        };
        cli_emit_wrapper emit = {
            .printer = &printer,
            .monitor = g_cli_runtime_monitor,
            .prompt_tokens = prompt.len,
            .max_tokens = cfg->gen.n_predict,
        };
        emit.t0 = cli_now_sec();
        rc = ds4_engine_generate_argmax(engine, &prompt, cfg->gen.n_predict,
                                        cfg->gen.ctx_size,
                                        cli_print_generated_token_monitored,
                                        cli_generation_done_monitored,
                                        &emit,
                                        cli_prefill_progress_cb,
                                        &progress);
        if (emit.t1 == 0.0) emit.t1 = cli_now_sec();
        int effective_prefill = progress.input_tokens;
        int effective_generated = emit.generated;
        if (rc == 0) {
            const double prefill_s = emit.t_first > emit.t0 ? emit.t_first - emit.t0 : 0.0;
            const double decode_s = emit.t1 > emit.t_first ? emit.t1 - emit.t_first : 0.0;
            ds4_log(stderr,
                    DS4_LOG_TIMING,
                    "ds4: prefill: %.2f t/s, generation: %.2f t/s\n",
                    prefill_s > 0.0 ? (double)effective_prefill / prefill_s : 0.0,
                    decode_s > 0.0 ? (double)effective_generated / decode_s : 0.0);
        }
    }

    ds4_tokens_free(&prompt);
    return rc;
}

static char *trim_inplace(char *s) {
    while (*s && isspace((unsigned char)*s)) s++;
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) end--;
    *end = '\0';
    return s;
}

static void print_repl_help(void) {
    puts("Commands:");
    puts("  /help          Show this help.");
    puts("  /think         Use normal thinking mode.");
    puts("  /think-max     Use Think Max only when context is at least 393216 tokens.");
    puts("  /nothink       Disable thinking mode.");
    puts("  /ctx N         Set context size for following prompts.");
    puts("  /read FILE     Read a prompt from FILE and run it.");
    puts("  /quit, /exit   Leave the prompt.");
    puts("  Ctrl+C         Stop generation and return to the prompt.");
}

static void history_file_path(char *buf, size_t len) {
    const char *home = getenv("HOME");
    if (!home || !home[0]) home = ".";
    snprintf(buf, len, "%s/.ds4_history", home);
}

typedef struct {
    ds4_session *session;
    ds4_tokens transcript;
    int ctx_size;
    int max_prefix_tokens;
} repl_chat;

static void tokens_insert(ds4_tokens *dst, int pos, const ds4_tokens *src) {
    if (!src || src->len <= 0) return;
    if (pos < 0) pos = 0;
    if (pos > dst->len) pos = dst->len;
    while (dst->len + src->len > dst->cap) {
        dst->cap = dst->cap ? dst->cap * 2 : 64;
        int *next = realloc(dst->v, (size_t)dst->cap * sizeof(dst->v[0]));
        if (!next) {
            perror("ds4: realloc");
            exit(1);
        }
        dst->v = next;
    }
    memmove(dst->v + pos + src->len, dst->v + pos,
            (size_t)(dst->len - pos) * sizeof(dst->v[0]));
    memcpy(dst->v + pos, src->v, (size_t)src->len * sizeof(src->v[0]));
    dst->len += src->len;
}

static void tokens_remove(ds4_tokens *dst, int pos, int n) {
    if (n <= 0 || pos < 0 || pos >= dst->len) return;
    if (pos + n > dst->len) n = dst->len - pos;
    memmove(dst->v + pos, dst->v + pos + n,
            (size_t)(dst->len - pos - n) * sizeof(dst->v[0]));
    dst->len -= n;
}

/* Insert/remove the Think Max prefix inside the existing transcript.  The
 * prefix lives after BOS, before any system/developer text, which mirrors the
 * API rendering path.  Changing it invalidates the session because every later
 * token position would otherwise refer to the wrong prefix. */
static void repl_chat_apply_max_prefix(ds4_engine *engine, repl_chat *chat, bool enable) {
    if (enable && chat->max_prefix_tokens == 0) {
        ds4_tokens prefix = {0};
        ds4_chat_append_max_effort_prefix(engine, &prefix);
        tokens_insert(&chat->transcript, 1, &prefix);
        chat->max_prefix_tokens = prefix.len;
        ds4_tokens_free(&prefix);
        if (chat->session) ds4_session_invalidate(chat->session);
    } else if (!enable && chat->max_prefix_tokens > 0) {
        tokens_remove(&chat->transcript, 1, chat->max_prefix_tokens);
        chat->max_prefix_tokens = 0;
        if (chat->session) ds4_session_invalidate(chat->session);
    }
}

static int repl_chat_create_session(ds4_engine *engine, repl_chat *chat, int ctx_size) {
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, ctx_size) != 0) {
        fprintf(stderr, "ds4: interactive chat KV cache requires a session backend\n");
        return 1;
    }
    if (chat->session) ds4_session_free(chat->session);
    chat->session = session;
    chat->ctx_size = ctx_size;
    return 0;
}

static int repl_chat_init(ds4_engine *engine, repl_chat *chat, const cli_config *cfg) {
    memset(chat, 0, sizeof(*chat));
    ds4_chat_begin(engine, &chat->transcript);
    repl_chat_apply_max_prefix(engine, chat,
                               cli_effective_think_mode(&cfg->gen) == DS4_THINK_MAX);
    if (cfg->gen.system && cfg->gen.system[0]) {
        ds4_chat_append_message(engine, &chat->transcript, "system", cfg->gen.system);
    }
    return repl_chat_create_session(engine, chat, cfg->gen.ctx_size);
}

static void repl_chat_free(repl_chat *chat) {
    if (!chat) return;
    ds4_session_free(chat->session);
    ds4_tokens_free(&chat->transcript);
    memset(chat, 0, sizeof(*chat));
}

static int repl_chat_set_ctx(ds4_engine *engine, repl_chat *chat, int ctx_size) {
    ds4_session_free(chat->session);
    chat->session = NULL;
    chat->ctx_size = 0;
    return repl_chat_create_session(engine, chat, ctx_size);
}

/* Run one interactive turn.  The transcript is tentatively extended with user
 * and assistant markers, then ds4_session_sync() decides whether this is a KV
 * continuation.  If prompt processing fails, the transcript rolls back before
 * returning to the prompt. */
static int run_chat_turn(ds4_engine *engine, cli_config *cfg, repl_chat *chat, const char *user_text) {
    if (!chat->session) {
        fprintf(stderr, "ds4: no active interactive KV cache\n");
        return 1;
    }

    ds4_think_mode think_mode = ds4_think_mode_for_context(cfg->gen.think_mode,
                                                           chat->ctx_size);
    repl_chat_apply_max_prefix(engine, chat, think_mode == DS4_THINK_MAX);
    const int rollback_len = chat->transcript.len;
    ds4_chat_append_message(engine, &chat->transcript, "user", user_text);
    ds4_chat_append_assistant_prefix(engine, &chat->transcript, think_mode);

    const int old_pos = ds4_session_pos(chat->session);
    const int common = ds4_session_common_prefix(chat->session, &chat->transcript);
    const int cached = common == old_pos && chat->transcript.len >= old_pos ? common : 0;
    const int suffix = chat->transcript.len - cached;

    char err[160];
    cli_prefill_progress progress = {
        .base_tokens = cached,
        .input_tokens = suffix,
        .use_color = ds4_log_is_tty(stderr),
        .monitor = g_cli_runtime_monitor,
    };
    const double t_prefill0 = cli_now_sec();
    ds4_session_set_progress(chat->session, cli_prefill_progress_cb, &progress);
    if (ds4_session_sync(chat->session, &chat->transcript, err, sizeof(err)) != 0) {
        ds4_session_set_progress(chat->session, NULL, NULL);
        chat->transcript.len = rollback_len;
        fprintf(stderr, "ds4: prompt processing failed: %s\n", err);
        return 1;
    }
    ds4_session_set_progress(chat->session, NULL, NULL);
    const double t_prefill1 = cli_now_sec();

    token_printer printer = {
        .engine = engine,
        .fp = stdout,
        .format_thinking = ds4_think_mode_enabled(think_mode),
        .in_think = ds4_think_mode_enabled(think_mode),
        .use_color = isatty(fileno(stdout)) != 0,
        .last_output_newline = true,
    };

    int max_tokens = cfg->gen.n_predict;
    int room = ds4_session_ctx(chat->session) - ds4_session_pos(chat->session);
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;

    uint64_t rng = cfg->gen.seed ? cfg->gen.seed :
        ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());
    int generated = 0;
    const double t_decode0 = cli_now_sec();
    cli_runtime_monitor_set_phase(g_cli_runtime_monitor, "generation");
    cli_runtime_monitor_set_progress(g_cli_runtime_monitor, 0, max_tokens);
    while (generated < max_tokens && !cli_interrupt_requested()) {
        int token = ds4_session_sample(chat->session,
                                       cfg->gen.temperature,
                                       0,
                                       cfg->gen.top_p,
                                       0.0f,
                                       &rng);
        if (token == ds4_token_eos(engine)) break;

        int toks[17];
        int ntok = 0;
        if (cfg->gen.temperature <= 0.0f && ds4_engine_mtp_draft_tokens(engine) > 1 &&
            getenv("DS4_MTP_SPEC_DISABLE") == NULL) {
            ntok = ds4_session_eval_speculative_argmax(chat->session,
                                                       token,
                                                       max_tokens - generated,
                                                       ds4_token_eos(engine),
                                                       toks,
                                                       (int)(sizeof(toks) / sizeof(toks[0])),
                                                       err,
                                                       sizeof(err));
            if (ntok < 0) {
                fprintf(stderr, "ds4: decode failed: %s\n", err);
                return 1;
            }
        } else {
            if (ds4_session_eval(chat->session, token, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4: decode failed: %s\n", err);
                return 1;
            }
            toks[0] = token;
            ntok = 1;
        }

        bool stop = false;
        for (int j = 0; j < ntok; j++) {
            if (toks[j] == ds4_token_eos(engine)) {
                stop = true;
                break;
            }
            size_t piece_len = 0;
            char *piece = ds4_token_text(engine, toks[j], &piece_len);
            ds4_tokens_push(&chat->transcript, toks[j]);
            token_printer_write_text(&printer, piece, piece_len);
            fflush(stdout);
            free(piece);
            generated++;
            cli_runtime_monitor_set_phase(g_cli_runtime_monitor, "generation");
            cli_runtime_monitor_set_progress(g_cli_runtime_monitor, generated, max_tokens);
            if (generated >= max_tokens) break;
        }
        if (stop) break;
    }
    const double t_decode1 = cli_now_sec();
    generation_done(&printer);

    const bool interrupted = cli_interrupt_requested();
    if (interrupted && generated == 0) {
        chat->transcript.len = rollback_len;
        ds4_session_invalidate(chat->session);
    } else {
        ds4_tokens_push(&chat->transcript, ds4_token_eos(engine));
    }

    const double prefill_s = t_prefill1 - t_prefill0;
    const double decode_s = t_decode1 - t_decode0;
    if (interrupted) cli_interrupt_clear();
    ds4_log(stderr,
            DS4_LOG_TIMING,
            "ds4: prefill: %.2f t/s, generation: %.2f t/s\n",
            prefill_s > 0.0 ? (double)suffix / prefill_s : 0.0,
            decode_s > 0.0 ? (double)generated / decode_s : 0.0);
    return 0;
}

static int run_repl(ds4_engine *engine, cli_config *cfg) {
    repl_chat chat;
    if (repl_chat_init(engine, &chat, cfg) != 0) return 1;

    struct sigaction old_int;
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sigemptyset(&sa.sa_mask);
    sa.sa_handler = cli_sigint_handler;
    bool sigint_installed = sigaction(SIGINT, &sa, &old_int) == 0;
    cli_interrupt_clear();

    char hist[PATH_MAX];
    history_file_path(hist, sizeof(hist));
    linenoiseSetMultiLine(1);
    linenoiseHistorySetMaxLen(512);
    linenoiseHistoryLoad(hist);
    print_repl_help();

    int rc = 0;
    for (;;) {
        errno = 0;
        char *line = linenoise("ds4> ");
        if (!line) {
            if (errno == EAGAIN || cli_interrupt_requested()) {
                cli_interrupt_clear();
                continue;
            }
            break;
        }
        char *cmd = trim_inplace(line);
        if (!cmd[0]) {
            linenoiseFree(line);
            continue;
        }
        linenoiseHistoryAdd(cmd);
        linenoiseHistorySave(hist);

        if (!strcmp(cmd, "/help")) {
            print_repl_help();
        } else if (!strcmp(cmd, "/think")) {
            cfg->gen.think_mode = DS4_THINK_HIGH;
            repl_chat_apply_max_prefix(engine, &chat, false);
            puts("Thinking mode: high.");
        } else if (!strcmp(cmd, "/think-max")) {
            cfg->gen.think_mode = DS4_THINK_MAX;
            bool active = ds4_think_mode_for_context(cfg->gen.think_mode,
                                                     chat.ctx_size) == DS4_THINK_MAX;
            repl_chat_apply_max_prefix(engine, &chat, active);
            cli_warn_think_max_downgraded(&cfg->gen, "/think-max");
            printf("Thinking mode: %s.\n", active ? "max" : "high (ctx below 393216)");
        } else if (!strcmp(cmd, "/nothink")) {
            cfg->gen.think_mode = DS4_THINK_NONE;
            repl_chat_apply_max_prefix(engine, &chat, false);
            puts("Thinking mode: none.");
        } else if (!strncmp(cmd, "/ctx", 4) && (cmd[4] == '\0' || isspace((unsigned char)cmd[4]))) {
            char *arg = trim_inplace(cmd + 4);
            if (!arg[0]) {
                fprintf(stderr, "ds4: /ctx needs a positive integer\n");
            } else {
                cfg->gen.ctx_size = parse_int(arg, "/ctx");
                log_context_memory(cfg->engine.backend, cfg->gen.ctx_size);
                rc = repl_chat_set_ctx(engine, &chat, cfg->gen.ctx_size);
                if (rc != 0) {
                    linenoiseFree(line);
                    break;
                }
                bool active = ds4_think_mode_for_context(cfg->gen.think_mode,
                                                         chat.ctx_size) == DS4_THINK_MAX;
                repl_chat_apply_max_prefix(engine, &chat, active);
                cli_warn_think_max_downgraded(&cfg->gen, "/ctx");
            }
        } else if (!strcmp(cmd, "/quit") || !strcmp(cmd, "/exit")) {
            linenoiseFree(line);
            break;
        } else if (!strncmp(cmd, "/read", 5) && (cmd[5] == '\0' || isspace((unsigned char)cmd[5]))) {
            char *path = trim_inplace(cmd + 5);
            if (!path[0]) {
                fprintf(stderr, "ds4: /read needs a file path\n");
            } else {
                char *prompt = read_prompt_file(path, false);
                if (prompt) {
                    rc = run_chat_turn(engine, cfg, &chat, prompt);
                    free(prompt);
                }
            }
        } else if (cmd[0] == '/') {
            fprintf(stderr, "ds4: unknown command: %s\n", cmd);
            fprintf(stderr, "ds4: type /help for commands\n");
        } else {
            rc = run_chat_turn(engine, cfg, &chat, cmd);
        }
        linenoiseFree(line);
    }
    if (sigint_installed) sigaction(SIGINT, &old_int, NULL);
    repl_chat_free(&chat);
    return rc;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4: missing value for %s\n", opt);
        exit(2);
    }
    return argv[++(*i)];
}

static char *read_prompt_file(const char *path, bool fatal) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open prompt file: %s\n", path);
        if (fatal) exit(2);
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "ds4: failed to seek prompt file: %s\n", path);
        fclose(fp);
        if (fatal) exit(2);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0) {
        fprintf(stderr, "ds4: failed to size prompt file: %s\n", path);
        fclose(fp);
        if (fatal) exit(2);
        return NULL;
    }
    rewind(fp);

    char *buf = malloc((size_t)len + 1);
    if (!buf) {
        fprintf(stderr, "ds4: out of memory reading prompt file: %s\n", path);
        fclose(fp);
        if (fatal) exit(2);
        return NULL;
    }
    size_t nread = fread(buf, 1, (size_t)len, fp);
    if (nread != (size_t)len) {
        fprintf(stderr, "ds4: failed to read prompt file: %s\n", path);
        free(buf);
        fclose(fp);
        if (fatal) exit(2);
        return NULL;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close prompt file: %s\n", path);
        free(buf);
        if (fatal) exit(2);
        return NULL;
    }
    buf[len] = '\0';
    return buf;
}

static cli_config parse_options(int argc, char **argv) {
    cli_config c = {
        .engine = {
            .model_path = "ds4flash.gguf",
            .backend = default_backend(),
            .mtp_draft_tokens = 1,
            .mtp_margin = 3.0f,
        },
        .gen = {
            .prompt = NULL,
            .system = "You are a helpful assistant",
            .n_predict = 50000,
            .ctx_size = 32768,
            .temperature = 1.0f,
            .top_p = 1.0f,
            .dump_logprobs_top_k = 20,
            .think_mode = DS4_THINK_HIGH,
        },
    };

    bool directional_steering_scale_set = false;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "-p") || !strcmp(arg, "--prompt")) {
            if (c.gen.prompt) {
                fprintf(stderr, "ds4: specify only one prompt source\n");
                exit(2);
            }
            c.gen.prompt = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prompt-file")) {
            if (c.gen.prompt) {
                fprintf(stderr, "ds4: specify only one prompt source\n");
                exit(2);
            }
            c.prompt_owned = read_prompt_file(need_arg(&i, argc, argv, arg), true);
            c.gen.prompt = c.prompt_owned;
        } else if (!strcmp(arg, "-sys") || !strcmp(arg, "--system")) {
            c.gen.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.engine.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp")) {
            c.engine.mtp_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp-draft")) {
            c.engine.mtp_draft_tokens = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--mtp-margin")) {
            c.engine.mtp_margin = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1000.0f);
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens")) {
            c.gen.n_predict = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) {
            c.gen.ctx_size = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--temp")) {
            c.gen.temperature = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 100.0f);
        } else if (!strcmp(arg, "--top-p")) {
            c.gen.top_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--seed")) {
            c.gen.seed = parse_u64(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--quality")) {
            c.engine.quality = true;
        } else if (!strcmp(arg, "--dir-steering-file")) {
            c.engine.directional_steering_file = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dir-steering-ffn")) {
            c.engine.directional_steering_ffn = parse_float_range(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            directional_steering_scale_set = true;
        } else if (!strcmp(arg, "--dir-steering-attn")) {
            c.engine.directional_steering_attn = parse_float_range(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            directional_steering_scale_set = true;
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.engine.n_threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--backend")) {
            c.engine.backend = parse_backend(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--cpu")) {
            c.engine.backend = DS4_BACKEND_CPU;
        } else if (!strcmp(arg, "--metal")) {
            c.engine.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--cuda")) {
            c.engine.backend = DS4_BACKEND_CUDA;
        } else if (!strcmp(arg, "--dump-tokens")) {
            c.gen.dump_tokens = true;
        } else if (!strcmp(arg, "--dump-logprobs")) {
            c.gen.dump_logprobs_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--logprobs-top-k")) {
            c.gen.dump_logprobs_top_k = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--imatrix-dataset")) {
            c.gen.imatrix_dataset_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--imatrix-out")) {
            c.gen.imatrix_output_path = need_arg(&i, argc, argv, arg);
            c.engine.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--imatrix-max-prompts")) {
            c.gen.imatrix_max_prompts = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--imatrix-max-tokens")) {
            c.gen.imatrix_max_tokens = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--think")) {
            c.gen.think_mode = DS4_THINK_HIGH;
        } else if (!strcmp(arg, "--think-max")) {
            c.gen.think_mode = DS4_THINK_MAX;
        } else if (!strcmp(arg, "--nothink")) {
            c.gen.think_mode = DS4_THINK_NONE;
        } else if (!strcmp(arg, "--head-test")) {
            c.gen.head_test = true;
        } else if (!strcmp(arg, "--first-token-test")) {
            c.gen.first_token_test = true;
        } else if (!strcmp(arg, "--metal-graph-test")) {
            c.gen.metal_graph_test = true;
            c.engine.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--metal-graph-full-test")) {
            c.gen.metal_graph_full_test = true;
            c.engine.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--metal-graph-prompt-test")) {
            c.gen.metal_graph_prompt_test = true;
            c.engine.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--metal-graph-generate")) {
            fprintf(stderr, "ds4: --metal-graph-generate was removed; --metal is the graph path\n");
            exit(2);
        } else if (!strcmp(arg, "--inspect")) {
            c.inspect = true;
        } else if (!strcmp(arg, "--runtime-monitor")) {
            c.runtime_monitor = true;
        } else if (!strcmp(arg, "--runtime-monitor-interval")) {
            c.runtime_monitor = true;
            c.runtime_monitor_interval = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.05f, 3600.0f);
        } else if (!strcmp(arg, "--flashmoe-export-dir")) {
            c.flashmoe_export_dir = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--flashmoe-export-manifest")) {
            c.flashmoe_export_manifest = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--warm-weights")) {
            c.engine.warm_weights = true;
        } else if (!strcmp(arg, "--server")) {
            fprintf(stderr, "ds4: use ds4-server for the HTTP server\n");
            exit(2);
        } else {
            fprintf(stderr, "ds4: unknown option: %s\n", arg);
            usage(stderr);
            exit(2);
        }
    }

    if (c.engine.directional_steering_file && !directional_steering_scale_set) {
        c.engine.directional_steering_ffn = 1.0f;
    }
    if ((c.flashmoe_export_dir == NULL) != (c.flashmoe_export_manifest == NULL)) {
        fprintf(stderr, "ds4: --flashmoe-export-dir and --flashmoe-export-manifest must be used together\n");
        exit(2);
    }
    if (c.gen.imatrix_output_path && !c.gen.imatrix_dataset_path) {
        fprintf(stderr, "ds4: --imatrix-out requires --imatrix-dataset\n");
        exit(2);
    }
    if (c.gen.imatrix_dataset_path && !c.gen.imatrix_output_path) {
        fprintf(stderr, "ds4: --imatrix-dataset requires --imatrix-out\n");
        exit(2);
    }

    return c;
}

int main(int argc, char **argv) {
    cli_config cfg = parse_options(argc, argv);
    if (cfg.gen.dump_tokens) {
        if (cfg.gen.prompt == NULL) {
            fprintf(stderr, "ds4: --dump-tokens requires -p or --prompt-file\n");
            free(cfg.prompt_owned);
            return 2;
        }
        int rc = ds4_dump_text_tokenization(cfg.engine.model_path,
                                            cfg.gen.prompt,
                                            stdout);
        free(cfg.prompt_owned);
        return rc;
    }
    if (!cfg.inspect) {
        log_context_memory(cfg.engine.backend, cfg.gen.ctx_size);
        cli_warn_think_max_downgraded(&cfg.gen, "--think-max");
    }
    cli_runtime_monitor monitor = {0};
    cli_runtime_monitor_start(&monitor, &cfg);
    g_cli_runtime_monitor = &monitor;
    cli_runtime_monitor_set_phase(g_cli_runtime_monitor, "startup");
    cli_runtime_monitor_set_progress(g_cli_runtime_monitor, 0, 0);
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &cfg.engine) != 0) {
        cli_runtime_monitor_stop(&monitor);
        g_cli_runtime_monitor = NULL;
        free(cfg.prompt_owned);
        return 1;
    }
    int rc = 0;
    if (cfg.inspect) {
        cli_runtime_monitor_set_phase(g_cli_runtime_monitor, "inspect");
        ds4_engine_summary(engine);
        if (cfg.flashmoe_export_dir) {
            cli_runtime_monitor_set_phase(g_cli_runtime_monitor, "flashmoe-export");
            rc = ds4_engine_export_flashmoe(engine,
                                            cfg.flashmoe_export_dir,
                                            cfg.flashmoe_export_manifest);
        }
    } else if (cfg.flashmoe_export_dir) {
        cli_runtime_monitor_set_phase(g_cli_runtime_monitor, "flashmoe-export");
        rc = ds4_engine_export_flashmoe(engine,
                                        cfg.flashmoe_export_dir,
                                        cfg.flashmoe_export_manifest);
    } else if (cfg.gen.imatrix_output_path) {
        cli_runtime_monitor_set_phase(g_cli_runtime_monitor, "imatrix");
        rc = ds4_engine_collect_imatrix(engine,
                                        cfg.gen.imatrix_dataset_path,
                                        cfg.gen.imatrix_output_path,
                                        cfg.gen.ctx_size,
                                        cfg.gen.imatrix_max_prompts,
                                        cfg.gen.imatrix_max_tokens);
    } else if (cfg.gen.prompt == NULL) {
        cli_runtime_monitor_set_phase(g_cli_runtime_monitor, "repl");
        rc = run_repl(engine, &cfg);
    } else {
        rc = run_generation(engine, &cfg);
    }
    ds4_engine_close(engine);
    cli_runtime_monitor_stop(&monitor);
    g_cli_runtime_monitor = NULL;
    free(cfg.prompt_owned);
    return rc;
}
