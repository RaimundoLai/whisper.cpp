// crispasr-addon.cpp - CrispASR (Parakeet, Distil-Whisper) N-API bindings
// This file is included by addon.cpp

#include "model-cache.h"
#include "../../third-party/CrispASR/src/parakeet.h"
#include "../../third-party/CrispASR/src/qwen3_tts.h"
#include "whisper.h"
#include "forced_aligner.h"
#include <fstream>
#include <sys/stat.h>

static bool crisp_file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

static std::string crisp_dir_of(const std::string& p) {
    auto sep = p.find_last_of("/\\");
    return (sep == std::string::npos) ? std::string(".") : p.substr(0, sep);
}

static std::string crisp_discover_codec(const std::string& model_path) {
    const std::string dir = crisp_dir_of(model_path);
    static const char* candidates[] = {
        "qwen3-tts-tokenizer-12hz.gguf",
        "qwen3-tts-tokenizer.gguf",
        "qwen3-tts-codec.gguf",
        "snac-24khz.gguf",
        "snac-24khz-q8_0.gguf",
        "snac.gguf",
        "dots-tts-soar-vocoder-q8_0.gguf",
        "dots-tts-soar-vocoder.gguf",
        "dac-44khz.gguf",
        "bigvgan-24khz.gguf",
        "bigvgan-22khz.gguf",
        "vocos-24khz.gguf"
    };
    for (const char* name : candidates) {
        std::string p = dir + "/" + name;
        if (crisp_file_exists(p))
            return p;
    }
    return "";
}

static std::string extract_clean_text_if_json(const std::string& input) {
    if (input.empty()) return "";
    size_t start_idx = 0;
    while (start_idx < input.length() && (input[start_idx] == ' ' || input[start_idx] == '\n' || input[start_idx] == '\r' || input[start_idx] == '\t')) {
        start_idx++;
    }
    if (start_idx >= input.length()) return input;
    if (input[start_idx] != '[' && input[start_idx] != '{') {
        return input;
    }
    
    std::string clean_text = "";
    size_t pos = start_idx;
    while (true) {
        size_t next_content = input.find("\"Content\":", pos);
        if (next_content == std::string::npos) next_content = input.find("\"content\":", pos);
        if (next_content == std::string::npos) next_content = input.find("\"text\":", pos);
        if (next_content == std::string::npos) break;
        
        size_t val_start = input.find("\"", next_content + 9);
        if (val_start == std::string::npos) break;
        val_start++;
        
        size_t val_end = val_start;
        while (val_end < input.length()) {
            if (input[val_end] == '\\') val_end += 2;
            else if (input[val_end] == '"') break;
            else val_end++;
        }
        if (val_end >= input.length()) break;
        
        std::string value = input.substr(val_start, val_end - val_start);
        std::string unescaped = "";
        for (size_t i = 0; i < value.length(); i++) {
            if (value[i] == '\\' && i + 1 < value.length()) {
                char c = value[i+1];
                if (c == '"' || c == '\\' || c == '/') unescaped += c;
                else if (c == 'n') unescaped += '\n';
                else if (c == 't') unescaped += '\t';
                else { unescaped += value[i]; unescaped += c; }
                i++;
            } else {
                unescaped += value[i];
            }
        }
        
        if (!unescaped.empty()) {
            size_t s = 0;
            while (s < unescaped.length() && (unescaped[s] == ' ' || unescaped[s] == '\t')) s++;
            size_t e = unescaped.length();
            while (e > s && (unescaped[e-1] == ' ' || unescaped[e-1] == '\t')) e--;
            std::string trimmed = unescaped.substr(s, e - s);
            
            if (trimmed.length() >= 2 && trimmed.front() == '[' && trimmed.back() == ']') {
                // Ignore bracketed non-speech events
            } else {
                if (!clean_text.empty()) clean_text += " ";
                clean_text += trimmed;
            }
        }
        pos = val_end + 1;
    }
    return clean_text.empty() ? input : clean_text;
}

static void trim_leading_replacement_and_spaces(std::string& str) {
    while (!str.empty()) {
        if (str.front() == ' ' || str.front() == '\t' || str.front() == '\r' || str.front() == '\n') {
            str.erase(str.begin());
        } else if (str.size() >= 3 &&
                   (unsigned char)str[0] == 0xEF &&
                   (unsigned char)str[1] == 0xBF &&
                   (unsigned char)str[2] == 0xBD) {
            str.erase(str.begin(), str.begin() + 3);
        } else {
            break;
        }
    }
}

// ============================================================================
// Parakeet ASR Implementation
// ============================================================================

struct parakeet_addon_result {
    std::string text;
    struct token_data {
        std::string text;
        int64_t start_ms;
        int64_t end_ms;
        float p;
    };
    std::vector<token_data> tokens;
    
    struct word_data {
        std::string text;
        int64_t start_ms;
        int64_t end_ms;
        float p;
    };
    std::vector<word_data> words;
};

class ParakeetWorker : public Napi::AsyncWorker {
public:
    ParakeetWorker(Napi::Function& callback,
                   std::string model_path,
                   std::vector<float> pcmf32,
                   int n_threads,
                   bool use_gpu,
                   bool debug,
                   bool reuse_instance,
                   int64_t auto_release_ms,
                   Napi::Env env)
        : Napi::AsyncWorker(callback),
          m_model_path(std::move(model_path)),
          m_pcmf32(std::move(pcmf32)),
          m_n_threads(n_threads),
          m_use_gpu(use_gpu),
          m_debug(debug),
          m_reuse_instance(reuse_instance),
          m_auto_release_ms(auto_release_ms) {}

    void Execute() override {
        auto& cache = ModelCache::instance();
        std::unique_lock<std::recursive_mutex> type_lock(cache.mutex(ModelType::PARAKEET));

        struct parakeet_context* ctx = nullptr;
        bool owned = false;

        if (m_reuse_instance) {
            ctx = static_cast<struct parakeet_context*>(
                cache.acquire(ModelType::PARAKEET, m_model_path, m_use_gpu));
        }

        if (!ctx) {
            struct parakeet_context_params params = parakeet_context_default_params();
            params.n_threads = m_n_threads;
            params.use_gpu = m_use_gpu;
            params.verbosity = m_debug ? 1 : 0;

            ctx = parakeet_init_from_file(m_model_path.c_str(), params);
            if (!ctx) {
                SetError("Failed to initialize Parakeet context from " + m_model_path);
                return;
            }
            if (m_reuse_instance) {
                cache.store(ModelType::PARAKEET, ctx, m_model_path, m_use_gpu, "", m_auto_release_ms);
            } else {
                owned = true;
            }
        }

        struct parakeet_result* res = parakeet_transcribe_ex(ctx, m_pcmf32.data(), m_pcmf32.size(), 0);
        if (res) {
            m_result.text = res->text ? res->text : "";
            
            for (int i = 0; i < res->n_tokens; i++) {
                parakeet_addon_result::token_data td;
                td.text = res->tokens[i].text;
                td.start_ms = res->tokens[i].t0 * 10;
                td.end_ms = res->tokens[i].t1 * 10;
                td.p = res->tokens[i].p;
                m_result.tokens.push_back(td);
            }
            
            for (int i = 0; i < res->n_words; i++) {
                parakeet_addon_result::word_data wd;
                wd.text = res->words[i].text;
                wd.start_ms = res->words[i].t0 * 10;
                wd.end_ms = res->words[i].t1 * 10;
                wd.p = res->words[i].p;
                m_result.words.push_back(wd);
            }
            
            parakeet_result_free(res);
        } else {
            SetError("Parakeet transcription failed");
        }

        type_lock.unlock();
        if (owned) {
            parakeet_free(ctx);
        } else {
            cache.markIdle(ModelType::PARAKEET);
        }
    }

    void OnOK() override {
        Napi::HandleScope scope(Env());
        Napi::Object result = Napi::Object::New(Env());

        result.Set("text", Napi::String::New(Env(), m_result.text));
        
        Napi::Array tokens = Napi::Array::New(Env(), m_result.tokens.size());
        for (size_t i = 0; i < m_result.tokens.size(); i++) {
            Napi::Object tok = Napi::Object::New(Env());
            tok.Set("word", Napi::String::New(Env(), m_result.tokens[i].text));
            tok.Set("start", Napi::Number::New(Env(), m_result.tokens[i].start_ms / 1000.0));
            tok.Set("end", Napi::Number::New(Env(), m_result.tokens[i].end_ms / 1000.0));
            tok.Set("p", Napi::Number::New(Env(), m_result.tokens[i].p));
            tokens[i] = tok;
        }
        result.Set("tokens", tokens);

        Napi::Array words = Napi::Array::New(Env(), m_result.words.size());
        for (size_t i = 0; i < m_result.words.size(); i++) {
            Napi::Object word = Napi::Object::New(Env());
            word.Set("word", Napi::String::New(Env(), m_result.words[i].text));
            word.Set("start", Napi::Number::New(Env(), m_result.words[i].start_ms / 1000.0));
            word.Set("end", Napi::Number::New(Env(), m_result.words[i].end_ms / 1000.0));
            word.Set("p", Napi::Number::New(Env(), m_result.words[i].p));
            words[i] = word;
        }
        result.Set("words", words);

        Callback().Call({Env().Null(), result});
    }

private:
    std::string m_model_path;
    std::vector<float> m_pcmf32;
    int m_n_threads;
    bool m_use_gpu;
    bool m_debug;
    bool m_reuse_instance;
    int64_t m_auto_release_ms;
    parakeet_addon_result m_result;
};

Napi::Value parakeetASR(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2 || !info[0].IsObject() || !info[1].IsFunction()) {
        Napi::TypeError::New(env, "Usage: parakeetASR(options, callback)").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Object options = info[0].As<Napi::Object>();
    if (!options.Has("model") || !options.Get("model").IsString()) {
        Napi::TypeError::New(env, "options.model (string) is required").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::string model_path = options.Get("model").As<Napi::String>();
    std::vector<float> pcmf32;

    if (options.Has("pcmf32") && options.Get("pcmf32").IsTypedArray()) {
        Napi::Float32Array pcmf32_arr = options.Get("pcmf32").As<Napi::Float32Array>();
        pcmf32.assign(pcmf32_arr.Data(), pcmf32_arr.Data() + pcmf32_arr.ElementLength());
    } else {
        Napi::TypeError::New(env, "options.pcmf32 (Float32Array) is required").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int n_threads = 4;
    if (options.Has("n_threads") && options.Get("n_threads").IsNumber()) {
        n_threads = options.Get("n_threads").As<Napi::Number>().Int32Value();
    }

    bool use_gpu = true;
    if (options.Has("use_gpu") && options.Get("use_gpu").IsBoolean()) {
        use_gpu = options.Get("use_gpu").As<Napi::Boolean>();
    }

    bool debug = false;
    if (options.Has("debug") && options.Get("debug").IsBoolean()) {
        debug = options.Get("debug").As<Napi::Boolean>();
    }

    bool reuse_instance = false;
    if (options.Has("reuse_instance") && options.Get("reuse_instance").IsBoolean()) {
        reuse_instance = options.Get("reuse_instance").As<Napi::Boolean>();
    }

    int64_t auto_release_ms = 0;
    if (options.Has("auto_release_ms") && options.Get("auto_release_ms").IsNumber()) {
        auto_release_ms = options.Get("auto_release_ms").As<Napi::Number>().Int64Value();
    }

    Napi::Function callback = info[1].As<Napi::Function>();
    ParakeetWorker* worker = new ParakeetWorker(callback, model_path, std::move(pcmf32), n_threads, use_gpu, debug, reuse_instance, auto_release_ms, env);
    worker->Queue();

    return env.Undefined();
}


// ============================================================================
// Distil-Whisper Implementation
// ============================================================================

Napi::Value distilWhisper(const Napi::CallbackInfo& info) {
    return whisper(info);
}

// ============================================================================
// Unified CrispASR Implementation (Supports VibeVoice, Voxtral, Parakeet, etc.)
// ============================================================================

struct crispasr_session;
struct crispasr_session_result;

struct crispasr_open_params_v1 {
    int abi_version; // = 1 or 2
    int n_threads;
    int use_gpu;   // 0 = CPU only, non-zero = GPU when available
    int verbosity; // 0 = silent, 1+ = chatty
    int flash_attn;   // 0 = off, non-zero = on (default on)
    int n_gpu_layers; // -1 = max, 0 = CPU-only LLM, >0 = bound
    int reserved[6];  // future-compat padding
};

extern "C" {
    int crispasr_detect_backend_from_gguf(const char* path, char* out_name, int out_cap);
    crispasr_session* crispasr_session_open_with_params(const char* model_path, const char* backend_name, const crispasr_open_params_v1* params);
    const char* crispasr_session_backend(crispasr_session* s);
    void crispasr_session_close(crispasr_session* s);

    crispasr_session_result* crispasr_session_transcribe_lang(crispasr_session* s, const float* pcm, int n_samples, const char* language);
    int crispasr_session_result_n_segments(crispasr_session_result* r);
    const char* crispasr_session_result_segment_text(crispasr_session_result* r, int i);
    int64_t crispasr_session_result_segment_t0(crispasr_session_result* r, int i);
    int64_t crispasr_session_result_segment_t1(crispasr_session_result* r, int i);
    int crispasr_session_result_n_words(crispasr_session_result* r, int i_seg);
    const char* crispasr_session_result_word_text(crispasr_session_result* r, int i_seg, int i_word);
    int64_t crispasr_session_result_word_t0(crispasr_session_result* r, int i_seg, int i_word);
    int64_t crispasr_session_result_word_t1(crispasr_session_result* r, int i_seg, int i_word);
    float crispasr_session_result_word_p(crispasr_session_result* r, int i_seg, int i_word);
    void crispasr_session_result_free(crispasr_session_result* r);

    // ASR setters
    int crispasr_session_set_translate(crispasr_session* s, int enable);
    int crispasr_session_set_source_language(crispasr_session* s, const char* lang);
    int crispasr_session_set_target_language(crispasr_session* s, const char* lang);
    int crispasr_session_set_ask(crispasr_session* s, const char* prompt);
    int crispasr_session_set_hotwords(crispasr_session* s, const char* hotwords, float boost);
    int crispasr_session_set_beam_size(crispasr_session* s, int n);
    int crispasr_session_set_temperature(crispasr_session* s, float temperature, uint64_t seed);
    int crispasr_session_set_top_p(crispasr_session* s, float top_p);
    int crispasr_session_set_top_k(crispasr_session* s, int top_k);
    int crispasr_session_set_min_p(crispasr_session* s, float min_p);
    int crispasr_session_set_repetition_penalty(crispasr_session* s, float r);
    int crispasr_session_set_frequency_penalty(crispasr_session* s, float penalty);
    int crispasr_session_set_best_of(crispasr_session* s, int n);
    int crispasr_session_set_max_new_tokens(crispasr_session* s, int n);
    int crispasr_session_set_punctuation(crispasr_session* s, int enable);
    int crispasr_session_set_punc_model(crispasr_session* s, const char* punc_model);
    int crispasr_session_set_sensitivity(crispasr_session* s, const char* preset_name);
    int crispasr_session_set_return_logits(crispasr_session* s, int enable);
    int crispasr_session_set_alt_n(crispasr_session* s, int n);
    int crispasr_session_set_whisper_decode_extras(crispasr_session* s, int suppress_nst, const char* suppress_regex, int carry_initial_prompt);
    int crispasr_session_set_parakeet_att_context(crispasr_session* s, int left, int right);

    // TTS & Synthesis
    float* crispasr_session_synthesize(crispasr_session* s, const char* text, int* out_n_samples);
    float* crispasr_session_synthesize_raw(crispasr_session* s, const char* text, int* out_n_samples);
    int crispasr_session_accept_marking_responsibility(crispasr_session* s, const char* attestation);
    void crispasr_watermark_embed(float* pcm, int n_samples, float alpha);
    float crispasr_watermark_detect(const float* pcm, int n_samples);
    int crispasr_watermark_load_model(const char* gguf_path);
    int crispasr_audio_load(const char* path, float** out_pcm, int* out_samples, int* out_sample_rate);
    void crispasr_audio_free(float* pcm);
    int crispasr_session_output_sample_rate(crispasr_session* s);
    int crispasr_session_set_codec_path(crispasr_session* s, const char* path);
    int crispasr_session_set_voice(crispasr_session* s, const char* path, const char* ref_text_or_null);
    int crispasr_session_set_speaker_name(crispasr_session* s, const char* name);
    int crispasr_session_set_speaker_id(crispasr_session* s, int id);
    int crispasr_session_set_instruct(crispasr_session* s, const char* instruct);
    int crispasr_session_set_tts_phonemes(crispasr_session* s, const char* phonemes);
    void crispasr_session_set_tts_pad_silence_ms(crispasr_session* s, int ms);
    int crispasr_session_set_tts_seed(crispasr_session* s, uint64_t seed);
    int crispasr_session_set_tts_steps(crispasr_session* s, int steps);
    int crispasr_session_set_tts_num_candidates(crispasr_session* s, int n);
    int crispasr_session_set_cfg_weight(crispasr_session* s, float cfg_weight);
    int crispasr_session_set_length_scale(crispasr_session* s, float scale);
    int crispasr_session_set_tts_noise_temp(crispasr_session* s, float noise_temp);
    int crispasr_session_set_exaggeration(crispasr_session* s, float exaggeration);
    int crispasr_session_set_max_speech_tokens(crispasr_session* s, int n);
    int crispasr_session_set_min_speech_tokens(crispasr_session* s, int n);
    int crispasr_session_set_speaker_identity(crispasr_session* s, const char* identity);
    int crispasr_session_set_g2p_dict(crispasr_session* s, const char* source);
    int crispasr_session_set_tts_reference_language(crispasr_session* s, const char* lang);
    int crispasr_session_is_voice_design(crispasr_session* s);
    int crispasr_session_is_custom_voice(crispasr_session* s);
    int crispasr_session_n_speakers(crispasr_session* s);
    const char* crispasr_session_get_speaker_name(crispasr_session* s, int i);
    void crispasr_pcm_free(float* pcm);
}

struct crispasr_addon_result {
    std::string text;
    
    struct word_data {
        std::string text;
        int64_t start_ms;
        int64_t end_ms;
        float p;
    };

    struct segment_data {
        std::string text;
        int64_t start_ms;
        int64_t end_ms;
        std::vector<word_data> words;
    };
    std::vector<segment_data> segments;
    
    std::string backend;
};

class CrispASRWorker : public Napi::AsyncWorker {
public:
    std::shared_ptr<std::atomic<bool>> m_should_abort;
    std::shared_ptr<std::atomic<bool>> m_was_aborted;

    CrispASRWorker(Napi::Function& callback,
                   std::string model_path,
                   std::string backend_name,
                   std::vector<float> pcmf32,
                   std::string audio_file,
                   int n_threads,
                   bool use_gpu,
                   bool debug,
                   bool flash_attn,
                   int n_gpu_layers,
                   std::string language,
                   bool translate,
                   std::string target_language,
                   std::string context,
                   std::string hotwords,
                   float hotwords_boost,
                   int beam_size,
                   float temperature,
                   int seed,
                   float top_p,
                   float min_p,
                   float repetition_penalty,
                   float frequency_penalty,
                   int best_of,
                   int max_new_tokens,
                   int punctuation,
                   std::string aligner_model_path,
                   std::string vad_model_path,
                   float vad_threshold,
                   int min_speech_ms,
                   int min_silence_ms,
                   int speech_pad_ms,
                   int max_speech_ms,
                   bool reuse_instance,
                   int64_t auto_release_ms,
                   Napi::Function progress_callback,
                   Napi::Env env,
                   std::string source_language = "",
                   std::string punc_model = "",
                   std::string sensitivity = "",
                   int top_k = -1,
                   int return_logits = -1,
                   int alt_n = -1,
                   bool suppress_nst = false,
                   std::string suppress_regex = "",
                   bool carry_initial_prompt = false,
                   int att_context_left = -1,
                   int att_context_right = -1)
        : Napi::AsyncWorker(callback),
          m_model_path(std::move(model_path)),
          m_backend_name(std::move(backend_name)),
          m_pcmf32(std::move(pcmf32)),
          m_audio_file(std::move(audio_file)),
          m_n_threads(n_threads),
          m_use_gpu(use_gpu),
          m_debug(debug),
          m_flash_attn(flash_attn),
          m_n_gpu_layers(n_gpu_layers),
          m_language(std::move(language)),
          m_translate(translate),
          m_target_language(std::move(target_language)),
          m_context(std::move(context)),
          m_hotwords(std::move(hotwords)),
          m_hotwords_boost(hotwords_boost),
          m_beam_size(beam_size),
          m_temperature(temperature),
          m_seed(seed),
          m_top_p(top_p),
          m_min_p(min_p),
          m_repetition_penalty(repetition_penalty),
          m_frequency_penalty(frequency_penalty),
          m_best_of(best_of),
          m_max_new_tokens(max_new_tokens),
          m_punctuation(punctuation),
          m_aligner_model_path(std::move(aligner_model_path)),
          m_vad_model_path(std::move(vad_model_path)),
          m_vad_threshold(vad_threshold),
          m_min_speech_ms(min_speech_ms),
          m_min_silence_ms(min_silence_ms),
          m_speech_pad_ms(speech_pad_ms),
          m_max_speech_ms(max_speech_ms),
          m_reuse_instance(reuse_instance),
          m_auto_release_ms(auto_release_ms),
          env(env),
          m_source_language(std::move(source_language)),
          m_punc_model(std::move(punc_model)),
          m_sensitivity(std::move(sensitivity)),
          m_top_k(top_k),
          m_return_logits(return_logits),
          m_alt_n(alt_n),
          m_suppress_nst(suppress_nst),
          m_suppress_regex(std::move(suppress_regex)),
          m_carry_initial_prompt(carry_initial_prompt),
          m_att_context_left(att_context_left),
          m_att_context_right(att_context_right),
          m_should_abort(std::make_shared<std::atomic<bool>>(false)),
          m_was_aborted(std::make_shared<std::atomic<bool>>(false)) {
        if (!progress_callback.IsEmpty()) {
            tsfn = Napi::ThreadSafeFunction::New(
                env,
                progress_callback,
                "CrispASR Progress Callback",
                0,
                1
            );
        }
    }

    ~CrispASRWorker() {
        if (tsfn) {
            tsfn.Release();
        }
    }

    void OnProgress(int progress) {
        if (tsfn && !m_should_abort->load()) {
            auto abort_flag = m_should_abort;
            auto callback = [abort_flag, progress](Napi::Env env, Napi::Function jsCallback) {
                try {
                    Napi::Value result = jsCallback.Call({Napi::Number::New(env, progress)});
                    if (result.IsBoolean() && !result.As<Napi::Boolean>().Value()) {
                        abort_flag->store(true);
                    }
                } catch (...) {
                }
            };
            tsfn.BlockingCall(callback);
        }
    }

    bool ShouldAbort() const {
        return m_should_abort->load();
    }

    void process_result(crispasr_session_result* res, int64_t offset_ms, const qwen3_asr::alignment_result* align_res = nullptr) {
        int n_segs = crispasr_session_result_n_segments(res);
        std::string full_text = m_result.text;
        
        bool use_aligner_words = (align_res && align_res->success);

        for (int i = 0; i < n_segs; i++) {
            crispasr_addon_result::segment_data sd;
            const char* seg_text = crispasr_session_result_segment_text(res, i);
            sd.text = seg_text ? seg_text : "";
            sd.start_ms = offset_ms + crispasr_session_result_segment_t0(res, i) * 10;
            sd.end_ms = offset_ms + crispasr_session_result_segment_t1(res, i) * 10;

            std::string clean_seg_text = extract_clean_text_if_json(sd.text);
            trim_leading_replacement_and_spaces(clean_seg_text);
            sd.text = clean_seg_text;
            if (!full_text.empty() && !clean_seg_text.empty()) {
                full_text += " ";
            }
            full_text += clean_seg_text;

            if (!use_aligner_words) {
                int n_words = crispasr_session_result_n_words(res, i);
                for (int j = 0; j < n_words; j++) {
                    crispasr_addon_result::word_data wd;
                    const char* w_text = crispasr_session_result_word_text(res, i, j);
                    wd.text = w_text ? w_text : "";
                    
                    int64_t t0 = crispasr_session_result_word_t0(res, i, j);
                    if (t0 < 0) continue; 

                    wd.start_ms = offset_ms + t0 * 10;
                    wd.end_ms = offset_ms + crispasr_session_result_word_t1(res, i, j) * 10;
                    wd.p = crispasr_session_result_word_p(res, i, j);

                    sd.words.push_back(wd);
                }
            }
            
            m_result.segments.push_back(std::move(sd));
        }

        m_result.text = full_text;

        if (use_aligner_words) {
            for (const auto& word : align_res->words) {
                crispasr_addon_result::word_data wd;
                wd.text = word.word;
                wd.start_ms = offset_ms + static_cast<int64_t>(word.start * 1000.0);
                wd.end_ms = offset_ms + static_cast<int64_t>(word.end * 1000.0);
                wd.p = 0.99f;
                
                if (m_result.segments.size() == 1) {
                    m_result.segments.back().words.push_back(wd);
                } else if (m_result.segments.size() > 1) {
                    bool distributed = false;
                    for (auto& sd : m_result.segments) {
                        if (wd.start_ms >= (sd.start_ms - 500) && wd.start_ms <= (sd.end_ms + 500)) {
                            sd.words.push_back(wd);
                            distributed = true;
                            break;
                        }
                    }
                    if (!distributed) {
                        m_result.segments.back().words.push_back(wd);
                    }
                }
            }
        }
    }

    void Execute() override {
        if (m_pcmf32.empty() && !m_audio_file.empty()) {
            std::vector<std::vector<float>> pcmf32s;
            if (!read_audio_data(m_audio_file, m_pcmf32, pcmf32s, false)) {
                SetError("failed to read audio file: " + m_audio_file);
                return;
            }
        }

        auto& cache = ModelCache::instance();
        std::unique_lock<std::recursive_mutex> type_lock(cache.mutex(ModelType::CRISPASR_SESSION));

        crispasr_session* session = nullptr;
        bool owned = false;

        if (m_reuse_instance) {
            session = static_cast<crispasr_session*>(
                cache.acquire(ModelType::CRISPASR_SESSION, m_model_path, m_use_gpu, m_backend_name));
        }

        if (!session) {
            crispasr_open_params_v1 params;
            memset(&params, 0, sizeof(params));
            params.abi_version = 2;
            params.n_threads = m_n_threads;
            params.use_gpu = m_use_gpu ? 1 : 0;
            params.verbosity = m_debug ? 1 : 0;
            params.flash_attn = m_flash_attn ? 1 : 0;
            params.n_gpu_layers = m_n_gpu_layers;

            const char* backend_ptr = m_backend_name.empty() ? nullptr : m_backend_name.c_str();

            session = crispasr_session_open_with_params(m_model_path.c_str(), backend_ptr, &params);
            if (!session) {
                SetError("Failed to open CrispASR session for model " + m_model_path);
                return;
            }
            if (m_reuse_instance) {
                cache.store(ModelType::CRISPASR_SESSION, session, m_model_path, m_use_gpu, m_backend_name, m_auto_release_ms);
            } else {
                owned = true;
            }
        }

        // Apply translate, target language, and context parameters
        crispasr_session_set_translate(session, m_translate ? 1 : 0);
        if (!m_target_language.empty()) {
            crispasr_session_set_target_language(session, m_target_language.c_str());
        }
        if (!m_context.empty()) {
            crispasr_session_set_ask(session, m_context.c_str());
        }
        if (!m_hotwords.empty()) {
            crispasr_session_set_hotwords(session, m_hotwords.c_str(), m_hotwords_boost);
        }
        if (m_beam_size > 1) {
            crispasr_session_set_beam_size(session, m_beam_size);
        }
        if (m_temperature >= 0.0f) {
            crispasr_session_set_temperature(session, m_temperature, m_seed >= 0 ? static_cast<uint64_t>(m_seed) : 0ULL);
        }
        if (m_top_p >= 0.0f) {
            crispasr_session_set_top_p(session, m_top_p);
        }
        if (m_min_p >= 0.0f) {
            crispasr_session_set_min_p(session, m_min_p);
        }
        if (m_repetition_penalty >= 0.0f) {
            crispasr_session_set_repetition_penalty(session, m_repetition_penalty);
        }
        if (m_frequency_penalty >= 0.0f) {
            crispasr_session_set_frequency_penalty(session, m_frequency_penalty);
        }
        if (m_best_of > 0) {
            crispasr_session_set_best_of(session, m_best_of);
        }
        if (m_max_new_tokens > 0) {
            crispasr_session_set_max_new_tokens(session, m_max_new_tokens);
        }
        if (m_punctuation >= 0) {
            crispasr_session_set_punctuation(session, m_punctuation);
        }
        if (!m_source_language.empty()) {
            crispasr_session_set_source_language(session, m_source_language.c_str());
        }
        if (!m_punc_model.empty()) {
            crispasr_session_set_punc_model(session, m_punc_model.c_str());
        }
        if (!m_sensitivity.empty()) {
            crispasr_session_set_sensitivity(session, m_sensitivity.c_str());
        }
        if (m_top_k > 0) {
            crispasr_session_set_top_k(session, m_top_k);
        }
        if (m_return_logits >= 0) {
            crispasr_session_set_return_logits(session, m_return_logits);
        }
        if (m_alt_n > 0) {
            crispasr_session_set_alt_n(session, m_alt_n);
        }
        if (m_suppress_nst || !m_suppress_regex.empty() || m_carry_initial_prompt) {
            crispasr_session_set_whisper_decode_extras(session, m_suppress_nst ? 1 : 0,
                                                       m_suppress_regex.empty() ? nullptr : m_suppress_regex.c_str(),
                                                       m_carry_initial_prompt ? 1 : 0);
        }
        if (m_att_context_left >= 0 || m_att_context_right >= 0) {
            crispasr_session_set_parakeet_att_context(session, m_att_context_left, m_att_context_right);
        }

        // Load ForcedAligner if path provided
        qwen3_asr::ForcedAligner aligner;
        bool use_aligner = !m_aligner_model_path.empty();
        if (use_aligner) {
            if (!aligner.load_model(m_aligner_model_path, m_use_gpu, m_debug)) {
                if (owned) {
                    crispasr_session_close(session);
                } else {
                    cache.markIdle(ModelType::CRISPASR_SESSION);
                }
                SetError("Failed to load aligner model: " + aligner.get_error());
                return;
            }
        }

        m_result.backend = crispasr_session_backend(session);
        const char* lang_ptr = m_language.empty() ? nullptr : m_language.c_str();

        if (m_vad_model_path.empty()) {
            std::vector<float> transcribe_pcm = m_pcmf32;
            if (transcribe_pcm.size() < 32000) {
                transcribe_pcm.resize(32000, 0.0f);
            }
            
            OnProgress(0);

            crispasr_session_result* res = crispasr_session_transcribe_lang(session, transcribe_pcm.data(), transcribe_pcm.size(), lang_ptr);
            if (res) {
                qwen3_asr::alignment_result align_res;
                if (use_aligner) {
                    int n_segs = crispasr_session_result_n_segments(res);
                    std::string total_txt = "";
                    for (int i = 0; i < n_segs; i++) {
                        const char* s_text = crispasr_session_result_segment_text(res, i);
                        if (s_text) {
                            if (!total_txt.empty()) total_txt += " ";
                            total_txt += s_text;
                        }
                    }
                    if (!total_txt.empty()) {
                        std::string detected_lang = m_language.empty() ? "zh" : m_language;
                        std::string clean_txt = extract_clean_text_if_json(total_txt);
                        trim_leading_replacement_and_spaces(clean_txt);
                        align_res = aligner.align(transcribe_pcm.data(), transcribe_pcm.size(), clean_txt, detected_lang);
                    }
                }
                process_result(res, 0, use_aligner ? &align_res : nullptr);
                crispasr_session_result_free(res);
            } else {
                SetError("CrispASR transcription failed");
            }
            
            OnProgress(100);
        } else {
            whisper_vad_context_params vad_ctx_params = whisper_vad_default_context_params();
            vad_ctx_params.n_threads = m_n_threads;
            vad_ctx_params.use_gpu = false;
            
            whisper_vad_context* vctx = whisper_vad_init_from_file_with_params(m_vad_model_path.c_str(), vad_ctx_params);
            if (!vctx) {
                type_lock.unlock();
                if (owned) {
                    crispasr_session_close(session);
                } else {
                    cache.markIdle(ModelType::CRISPASR_SESSION);
                }
                SetError("Failed to initialize whisper VAD context");
                return;
            }

            whisper_vad_params vad_params = whisper_vad_default_params();
            vad_params.threshold = m_vad_threshold;
            vad_params.min_speech_duration_ms = m_min_speech_ms;
            vad_params.min_silence_duration_ms = m_min_silence_ms;
            vad_params.max_speech_duration_s = FLT_MAX;
            vad_params.speech_pad_ms = m_speech_pad_ms;

            whisper_vad_segments* segments = whisper_vad_segments_from_samples(vctx, vad_params, m_pcmf32.data(), m_pcmf32.size());
            if (segments) {
                int n_segments = whisper_vad_segments_n_segments(segments);
                const int64_t max_samples_per_chunk = (static_cast<int64_t>(m_max_speech_ms) * 16000) / 1000;

                for (int i = 0; i < n_segments; i++) {
                    if (ShouldAbort()) {
                        m_was_aborted->store(true);
                        break;
                    }
                    float t0 = whisper_vad_segments_get_segment_t0(segments, i) / 100.0f;
                    float t1 = whisper_vad_segments_get_segment_t1(segments, i) / 100.0f;
                    
                    int64_t seg_start_sample = static_cast<int64_t>(t0 * 16000);
                    int64_t seg_end_sample = static_cast<int64_t>(t1 * 16000);
                    
                    if (seg_start_sample < 0) seg_start_sample = 0;
                    if (seg_end_sample > (int64_t)m_pcmf32.size()) seg_end_sample = m_pcmf32.size();
                    if (seg_end_sample <= seg_start_sample) continue;

                    int64_t chunk_start = seg_start_sample;
                    while (chunk_start < seg_end_sample) {
                        if (ShouldAbort()) {
                            m_was_aborted->store(true);
                            break;
                        }

                        int64_t chunk_end = std::min(chunk_start + max_samples_per_chunk, seg_end_sample);
                        
                        float chunk_t0 = static_cast<float>(chunk_start) / 16000.0f;
                        float chunk_t1 = static_cast<float>(chunk_end) / 16000.0f;

                        int progress = static_cast<int>(((float)chunk_start / m_pcmf32.size()) * 100);
                        OnProgress(progress);

                        std::vector<float> chunk(m_pcmf32.begin() + chunk_start, m_pcmf32.begin() + chunk_end);
                        if (chunk.size() < 32000) {
                            chunk.resize(32000, 0.0f);
                        }

                        crispasr_session_result* res = crispasr_session_transcribe_lang(session, chunk.data(), chunk.size(), lang_ptr);
                        if (res) {
                            qwen3_asr::alignment_result align_res;
                            if (use_aligner) {
                                int n_segs = crispasr_session_result_n_segments(res);
                                std::string total_txt = "";
                                for (int k = 0; k < n_segs; k++) {
                                    const char* s_text = crispasr_session_result_segment_text(res, k);
                                    if (s_text) {
                                        if (!total_txt.empty()) total_txt += " ";
                                        total_txt += s_text;
                                    }
                                }
                                if (!total_txt.empty()) {
                                    std::string detected_lang = m_language.empty() ? "zh" : m_language;
                                    std::string clean_txt = extract_clean_text_if_json(total_txt);
                                    trim_leading_replacement_and_spaces(clean_txt);
                                    align_res = aligner.align(chunk.data(), chunk.size(), clean_txt, detected_lang);
                                }
                            }
                            process_result(res, static_cast<int64_t>(chunk_t0 * 1000), use_aligner ? &align_res : nullptr);
                            crispasr_session_result_free(res);
                        }

                        chunk_start = chunk_end;
                    }
                }
                whisper_vad_free_segments(segments);
                OnProgress(100);
            }
            whisper_vad_free(vctx);
        }

        type_lock.unlock();
        if (owned) {
            crispasr_session_close(session);
        } else {
            cache.markIdle(ModelType::CRISPASR_SESSION);
        }
    }

    void OnOK() override {
        Napi::HandleScope scope(Env());
        Napi::Object result = Napi::Object::New(Env());

        result.Set("text", Napi::String::New(Env(), m_result.text));
        result.Set("backend", Napi::String::New(Env(), m_result.backend));
        result.Set("aborted", Napi::Boolean::New(Env(), m_was_aborted->load()));

        // Set segments
        Napi::Array segments = Napi::Array::New(Env(), m_result.segments.size());
        for (size_t i = 0; i < m_result.segments.size(); i++) {
            Napi::Object s_obj = Napi::Object::New(Env());
            s_obj.Set("text", Napi::String::New(Env(), m_result.segments[i].text));
            s_obj.Set("start", Napi::Number::New(Env(), m_result.segments[i].start_ms));
            s_obj.Set("end", Napi::Number::New(Env(), m_result.segments[i].end_ms));

            Napi::Array s_words = Napi::Array::New(Env(), m_result.segments[i].words.size());
            for (size_t j = 0; j < m_result.segments[i].words.size(); j++) {
                Napi::Object w_obj = Napi::Object::New(Env());
                w_obj.Set("word", Napi::String::New(Env(), m_result.segments[i].words[j].text));
                w_obj.Set("start", Napi::Number::New(Env(), m_result.segments[i].words[j].start_ms / 1000.0));
                w_obj.Set("end", Napi::Number::New(Env(), m_result.segments[i].words[j].end_ms / 1000.0));
                w_obj.Set("p", Napi::Number::New(Env(), m_result.segments[i].words[j].p));
                s_words[j] = w_obj;
            }
            s_obj.Set("words", s_words);
            segments[i] = s_obj;
        }
        result.Set("segments", segments);

        Callback().Call({Env().Null(), result});
    }

private:
    std::string m_model_path;
    std::string m_backend_name;
    std::vector<float> m_pcmf32;
    std::string m_audio_file;
    int m_n_threads;
    bool m_use_gpu;
    bool m_debug;
    bool m_flash_attn;
    int m_n_gpu_layers;
    std::string m_language;
    bool m_translate;
    std::string m_target_language;
    std::string m_context;
    std::string m_hotwords;
    float m_hotwords_boost;
    int m_beam_size;
    float m_temperature;
    int m_seed;
    float m_top_p;
    float m_min_p;
    float m_repetition_penalty;
    float m_frequency_penalty;
    int m_best_of;
    int m_max_new_tokens;
    int m_punctuation;
    std::string m_aligner_model_path;
    std::string m_source_language;
    std::string m_punc_model;
    std::string m_sensitivity;
    int m_top_k = -1;
    int m_return_logits = -1;
    int m_alt_n = -1;
    bool m_suppress_nst = false;
    std::string m_suppress_regex;
    bool m_carry_initial_prompt = false;
    int m_att_context_left = -1;
    int m_att_context_right = -1;
    std::string m_vad_model_path;
    float m_vad_threshold;
    int m_min_speech_ms;
    int m_min_silence_ms;
    int m_speech_pad_ms;
    int m_max_speech_ms;
    Napi::Env env;
    Napi::ThreadSafeFunction tsfn;
    bool m_reuse_instance = false;
    int64_t m_auto_release_ms = 0;
    crispasr_addon_result m_result;
};

Napi::Value crispasrASR(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2 || !info[0].IsObject() || !info[1].IsFunction()) {
        Napi::TypeError::New(env, "Usage: crispasrASR(options, callback)").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Object options = info[0].As<Napi::Object>();
    if (!options.Has("model") || !options.Get("model").IsString()) {
        Napi::TypeError::New(env, "options.model (string) is required").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::string model_path = options.Get("model").As<Napi::String>();
    std::string backend_name = "";
    if (options.Has("backend") && options.Get("backend").IsString()) {
        backend_name = options.Get("backend").As<Napi::String>();
        if (backend_name == "crispasr" || backend_name == "CrispASR") {
            backend_name = "";
        } else if (backend_name == "moss" || backend_name == "moss-transcribe" || backend_name == "moss_transcribe") {
            backend_name = "moss-diarize";
        }
    }

    std::vector<float> pcmf32;
    std::string audio_file = "";
    if (options.Has("pcmf32") && options.Get("pcmf32").IsTypedArray()) {
        Napi::Float32Array pcmf32_arr = options.Get("pcmf32").As<Napi::Float32Array>();
        pcmf32.assign(pcmf32_arr.Data(), pcmf32_arr.Data() + pcmf32_arr.ElementLength());
    } else if (options.Has("file") && options.Get("file").IsString()) {
        audio_file = options.Get("file").As<Napi::String>();
    } else {
        Napi::TypeError::New(env, "options.pcmf32 (Float32Array) or options.file (string) is required").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
    if (options.Has("n_threads") && options.Get("n_threads").IsNumber()) {
        n_threads = options.Get("n_threads").As<Napi::Number>().Int32Value();
    }

    bool use_gpu = true;
    if (options.Has("use_gpu") && options.Get("use_gpu").IsBoolean()) {
        use_gpu = options.Get("use_gpu").As<Napi::Boolean>();
    }

    bool debug = false;
    if (options.Has("debug") && options.Get("debug").IsBoolean()) {
        debug = options.Get("debug").As<Napi::Boolean>();
    }

    bool flash_attn = true;
    if (options.Has("flash_attn") && options.Get("flash_attn").IsBoolean()) {
        flash_attn = options.Get("flash_attn").As<Napi::Boolean>();
    }

    int n_gpu_layers = -1;
    if (options.Has("n_gpu_layers") && options.Get("n_gpu_layers").IsNumber()) {
        n_gpu_layers = options.Get("n_gpu_layers").As<Napi::Number>().Int32Value();
    }

    std::string language = "";
    if (options.Has("language") && options.Get("language").IsString()) {
        language = options.Get("language").As<Napi::String>();
    }

    bool translate = false;
    if (options.Has("translate") && options.Get("translate").IsBoolean()) {
        translate = options.Get("translate").As<Napi::Boolean>();
    }

    std::string target_language = "";
    if (options.Has("target_language") && options.Get("target_language").IsString()) {
        target_language = options.Get("target_language").As<Napi::String>();
    }

    std::string context = "";
    if (options.Has("context") && options.Get("context").IsString()) {
        context = options.Get("context").As<Napi::String>();
    } else if (options.Has("prompt") && options.Get("prompt").IsString()) {
        context = options.Get("prompt").As<Napi::String>();
    } else if (options.Has("ask") && options.Get("ask").IsString()) {
        context = options.Get("ask").As<Napi::String>();
    }

    std::string source_language = "";
    if (options.Has("source_language") && options.Get("source_language").IsString()) {
        source_language = options.Get("source_language").As<Napi::String>();
    } else if (options.Has("sourceLanguage") && options.Get("sourceLanguage").IsString()) {
        source_language = options.Get("sourceLanguage").As<Napi::String>();
    } else if (options.Has("source-language") && options.Get("source-language").IsString()) {
        source_language = options.Get("source-language").As<Napi::String>();
    }

    std::string punc_model = "";
    if (options.Has("punc_model") && options.Get("punc_model").IsString()) {
        punc_model = options.Get("punc_model").As<Napi::String>();
    } else if (options.Has("puncModel") && options.Get("puncModel").IsString()) {
        punc_model = options.Get("puncModel").As<Napi::String>();
    } else if (options.Has("punc-model") && options.Get("punc-model").IsString()) {
        punc_model = options.Get("punc-model").As<Napi::String>();
    }

    std::string sensitivity = "";
    if (options.Has("sensitivity") && options.Get("sensitivity").IsString()) {
        sensitivity = options.Get("sensitivity").As<Napi::String>();
    }

    int top_k = -1;
    if (options.Has("top_k") && options.Get("top_k").IsNumber()) {
        top_k = options.Get("top_k").As<Napi::Number>().Int32Value();
    } else if (options.Has("topK") && options.Get("topK").IsNumber()) {
        top_k = options.Get("topK").As<Napi::Number>().Int32Value();
    } else if (options.Has("top-k") && options.Get("top-k").IsNumber()) {
        top_k = options.Get("top-k").As<Napi::Number>().Int32Value();
    }

    int return_logits = -1;
    if (options.Has("return_logits")) {
        return_logits = options.Get("return_logits").As<Napi::Boolean>() ? 1 : 0;
    } else if (options.Has("returnLogits")) {
        return_logits = options.Get("returnLogits").As<Napi::Boolean>() ? 1 : 0;
    } else if (options.Has("return-logits")) {
        return_logits = options.Get("return-logits").As<Napi::Boolean>() ? 1 : 0;
    }

    int alt_n = -1;
    if (options.Has("alt_n") && options.Get("alt_n").IsNumber()) {
        alt_n = options.Get("alt_n").As<Napi::Number>().Int32Value();
    } else if (options.Has("altN") && options.Get("altN").IsNumber()) {
        alt_n = options.Get("altN").As<Napi::Number>().Int32Value();
    } else if (options.Has("alt-n") && options.Get("alt-n").IsNumber()) {
        alt_n = options.Get("alt-n").As<Napi::Number>().Int32Value();
    }

    bool suppress_nst = false;
    if (options.Has("suppress_nst") && options.Get("suppress_nst").IsBoolean()) {
        suppress_nst = options.Get("suppress_nst").As<Napi::Boolean>();
    } else if (options.Has("suppressNst") && options.Get("suppressNst").IsBoolean()) {
        suppress_nst = options.Get("suppressNst").As<Napi::Boolean>();
    } else if (options.Has("suppress-nst") && options.Get("suppress-nst").IsBoolean()) {
        suppress_nst = options.Get("suppress-nst").As<Napi::Boolean>();
    }

    std::string suppress_regex = "";
    if (options.Has("suppress_regex") && options.Get("suppress_regex").IsString()) {
        suppress_regex = options.Get("suppress_regex").As<Napi::String>();
    } else if (options.Has("suppressRegex") && options.Get("suppressRegex").IsString()) {
        suppress_regex = options.Get("suppressRegex").As<Napi::String>();
    } else if (options.Has("suppress-regex") && options.Get("suppress-regex").IsString()) {
        suppress_regex = options.Get("suppress-regex").As<Napi::String>();
    }

    bool carry_initial_prompt = false;
    if (options.Has("carry_initial_prompt") && options.Get("carry_initial_prompt").IsBoolean()) {
        carry_initial_prompt = options.Get("carry_initial_prompt").As<Napi::Boolean>();
    } else if (options.Has("carryInitialPrompt") && options.Get("carryInitialPrompt").IsBoolean()) {
        carry_initial_prompt = options.Get("carryInitialPrompt").As<Napi::Boolean>();
    } else if (options.Has("carry-initial-prompt") && options.Get("carry-initial-prompt").IsBoolean()) {
        carry_initial_prompt = options.Get("carry-initial-prompt").As<Napi::Boolean>();
    }

    int att_context_left = -1;
    if (options.Has("att_context_left") && options.Get("att_context_left").IsNumber()) {
        att_context_left = options.Get("att_context_left").As<Napi::Number>().Int32Value();
    } else if (options.Has("attContextLeft") && options.Get("attContextLeft").IsNumber()) {
        att_context_left = options.Get("attContextLeft").As<Napi::Number>().Int32Value();
    }

    int att_context_right = -1;
    if (options.Has("att_context_right") && options.Get("att_context_right").IsNumber()) {
        att_context_right = options.Get("att_context_right").As<Napi::Number>().Int32Value();
    } else if (options.Has("attContextRight") && options.Get("attContextRight").IsNumber()) {
        att_context_right = options.Get("attContextRight").As<Napi::Number>().Int32Value();
    }

    std::string hotwords = "";
    if (options.Has("hotwords") && options.Get("hotwords").IsString()) {
        hotwords = options.Get("hotwords").As<Napi::String>();
    }

    float hotwords_boost = 1.5f;
    if (options.Has("hotwords_boost") && options.Get("hotwords_boost").IsNumber()) {
        hotwords_boost = options.Get("hotwords_boost").As<Napi::Number>().FloatValue();
    } else if (options.Has("hotwords-boost") && options.Get("hotwords-boost").IsNumber()) {
        hotwords_boost = options.Get("hotwords-boost").As<Napi::Number>().FloatValue();
    }

    int beam_size = 1;
    if (options.Has("beam_size") && options.Get("beam_size").IsNumber()) {
        beam_size = options.Get("beam_size").As<Napi::Number>().Int32Value();
    } else if (options.Has("beam-size") && options.Get("beam-size").IsNumber()) {
        beam_size = options.Get("beam-size").As<Napi::Number>().Int32Value();
    }

    float temperature = -1.0f;
    if (options.Has("temperature") && options.Get("temperature").IsNumber()) {
        temperature = options.Get("temperature").As<Napi::Number>().FloatValue();
    }

    int seed = -1;
    if (options.Has("seed") && options.Get("seed").IsNumber()) {
        seed = options.Get("seed").As<Napi::Number>().Int32Value();
    }

    float top_p = -1.0f;
    if (options.Has("top_p") && options.Get("top_p").IsNumber()) {
        top_p = options.Get("top_p").As<Napi::Number>().FloatValue();
    } else if (options.Has("top-p") && options.Get("top-p").IsNumber()) {
        top_p = options.Get("top-p").As<Napi::Number>().FloatValue();
    }

    float min_p = -1.0f;
    if (options.Has("min_p") && options.Get("min_p").IsNumber()) {
        min_p = options.Get("min_p").As<Napi::Number>().FloatValue();
    } else if (options.Has("min-p") && options.Get("min-p").IsNumber()) {
        min_p = options.Get("min-p").As<Napi::Number>().FloatValue();
    }

    float repetition_penalty = -1.0f;
    if (options.Has("repetition_penalty") && options.Get("repetition_penalty").IsNumber()) {
        repetition_penalty = options.Get("repetition_penalty").As<Napi::Number>().FloatValue();
    } else if (options.Has("repetition-penalty") && options.Get("repetition-penalty").IsNumber()) {
        repetition_penalty = options.Get("repetition-penalty").As<Napi::Number>().FloatValue();
    }

    float frequency_penalty = -1.0f;
    if (options.Has("frequency_penalty") && options.Get("frequency_penalty").IsNumber()) {
        frequency_penalty = options.Get("frequency_penalty").As<Napi::Number>().FloatValue();
    } else if (options.Has("frequency-penalty") && options.Get("frequency-penalty").IsNumber()) {
        frequency_penalty = options.Get("frequency-penalty").As<Napi::Number>().FloatValue();
    }

    int best_of = -1;
    if (options.Has("best_of") && options.Get("best_of").IsNumber()) {
        best_of = options.Get("best_of").As<Napi::Number>().Int32Value();
    } else if (options.Has("best-of") && options.Get("best-of").IsNumber()) {
        best_of = options.Get("best-of").As<Napi::Number>().Int32Value();
    }

    int max_new_tokens = -1;
    if (options.Has("max_new_tokens") && options.Get("max_new_tokens").IsNumber()) {
        max_new_tokens = options.Get("max_new_tokens").As<Napi::Number>().Int32Value();
    } else if (options.Has("max-new-tokens") && options.Get("max-new-tokens").IsNumber()) {
        max_new_tokens = options.Get("max-new-tokens").As<Napi::Number>().Int32Value();
    }

    int punctuation = -1;
    if (options.Has("punctuation")) {
        punctuation = options.Get("punctuation").As<Napi::Boolean>() ? 1 : 0;
    }

    std::string aligner_model_path = "";
    if (options.Has("aligner_model") && options.Get("aligner_model").IsString()) {
        aligner_model_path = options.Get("aligner_model").As<Napi::String>();
    } else if (options.Has("aligner") && options.Get("aligner").IsString()) {
        aligner_model_path = options.Get("aligner").As<Napi::String>();
    } else if (options.Has("alignerModel") && options.Get("alignerModel").IsString()) {
        aligner_model_path = options.Get("alignerModel").As<Napi::String>();
    }

    // Additional VAD parameters
    std::string vad_model_path = "";
    if (options.Has("vad_model") && options.Get("vad_model").IsString()) {
        vad_model_path = options.Get("vad_model").As<Napi::String>();
    }

    float vad_threshold = 0.3f;
    if (options.Has("vad_threshold") && options.Get("vad_threshold").IsNumber()) {
        vad_threshold = options.Get("vad_threshold").As<Napi::Number>().FloatValue();
    }

    int min_speech_ms = 250;
    if (options.Has("min_speech_duration_ms") && options.Get("min_speech_duration_ms").IsNumber()) {
        min_speech_ms = options.Get("min_speech_duration_ms").As<Napi::Number>().Int32Value();
    }

    int min_silence_ms = 100;
    if (options.Has("min_silence_duration_ms") && options.Get("min_silence_duration_ms").IsNumber()) {
        min_silence_ms = options.Get("min_silence_duration_ms").As<Napi::Number>().Int32Value();
    }

    int speech_pad_ms = 200;
    if (options.Has("speech_pad_ms") && options.Get("speech_pad_ms").IsNumber()) {
        speech_pad_ms = options.Get("speech_pad_ms").As<Napi::Number>().Int32Value();
    }

    int max_speech_ms = 30000;
    if (options.Has("max_speech_duration_ms") && options.Get("max_speech_duration_ms").IsNumber()) {
        max_speech_ms = options.Get("max_speech_duration_ms").As<Napi::Number>().Int32Value();
    }

    bool reuse_instance = false;
    if (options.Has("reuse_instance") && options.Get("reuse_instance").IsBoolean()) {
        reuse_instance = options.Get("reuse_instance").As<Napi::Boolean>();
    }

    int64_t auto_release_ms = 0;
    if (options.Has("auto_release_ms") && options.Get("auto_release_ms").IsNumber()) {
        auto_release_ms = options.Get("auto_release_ms").As<Napi::Number>().Int64Value();
    }

    Napi::Function progress_callback;
    if (options.Has("progress_callback") && options.Get("progress_callback").IsFunction()) {
        progress_callback = options.Get("progress_callback").As<Napi::Function>();
    }

    Napi::Function callback = info[1].As<Napi::Function>();
    CrispASRWorker* worker = new CrispASRWorker(
        callback, model_path, backend_name, std::move(pcmf32), std::move(audio_file), n_threads, use_gpu, debug, 
        flash_attn, n_gpu_layers, language,
        translate, target_language, context, hotwords, hotwords_boost, beam_size,
        temperature, seed, top_p, min_p, repetition_penalty, frequency_penalty, best_of, max_new_tokens, punctuation,
        aligner_model_path,
        vad_model_path, vad_threshold, min_speech_ms, min_silence_ms, speech_pad_ms, max_speech_ms,
        reuse_instance, auto_release_ms,
        progress_callback, env,
        source_language, punc_model, sensitivity, top_k, return_logits, alt_n, suppress_nst, suppress_regex,
        carry_initial_prompt, att_context_left, att_context_right
    );
    worker->Queue();

    return env.Undefined();
}

class CrispASRStream : public Napi::ObjectWrap<CrispASRStream> {
public:
    struct stream_word_item {
        std::string word;
        int64_t start;
        int64_t end;
        float p;
    };

    static Napi::Object Init(Napi::Env env, Napi::Object exports) {
        Napi::Function func = DefineClass(env, "CrispASRStream", {
            InstanceMethod("start", &CrispASRStream::Start),
            InstanceMethod("addAudio", &CrispASRStream::AddAudio),
            InstanceMethod("stop", &CrispASRStream::Stop),
            InstanceMethod("pause", &CrispASRStream::Pause),
            InstanceMethod("resume", &CrispASRStream::Resume),
            InstanceMethod("finish", &CrispASRStream::Finish),
            InstanceMethod("release", &CrispASRStream::Release),
        });
        exports.Set("CrispASRStream", func);
        return exports;
    }

    CrispASRStream(const Napi::CallbackInfo& info) : Napi::ObjectWrap<CrispASRStream>(info) {
        Napi::Env env = info.Env();
        Napi::Object options = info[0].As<Napi::Object>();
        m_model_path = options.Get("model").As<Napi::String>();
        if (options.Has("backend")) {
            m_backend_name = options.Get("backend").As<Napi::String>();
            if (m_backend_name == "crispasr" || m_backend_name == "CrispASR") {
                m_backend_name = "";
            } else if (m_backend_name == "moss" || m_backend_name == "moss-transcribe" || m_backend_name == "moss_transcribe") {
                m_backend_name = "moss-diarize";
            }
        }
        if (options.Has("language")) m_language = options.Get("language").As<Napi::String>();
        if (options.Has("n_threads")) m_n_threads = options.Get("n_threads").As<Napi::Number>().Int32Value();
        if (options.Has("use_gpu")) m_use_gpu = options.Get("use_gpu").As<Napi::Boolean>();
        if (options.Has("flash_attn")) m_flash_attn = options.Get("flash_attn").As<Napi::Boolean>();
        if (options.Has("n_gpu_layers")) m_n_gpu_layers = options.Get("n_gpu_layers").As<Napi::Number>().Int32Value();
        if (options.Has("chunk_size_ms")) m_chunk_size_ms = options.Get("chunk_size_ms").As<Napi::Number>().Int32Value();
        
        if (options.Has("translate")) m_translate = options.Get("translate").As<Napi::Boolean>();
        if (options.Has("target_language")) m_target_language = options.Get("target_language").As<Napi::String>();
        if (options.Has("context")) m_context = options.Get("context").As<Napi::String>();
        else if (options.Has("prompt")) m_context = options.Get("prompt").As<Napi::String>();
        else if (options.Has("ask")) m_context = options.Get("ask").As<Napi::String>();

        if (options.Has("source_language") && options.Get("source_language").IsString()) {
            m_source_language = options.Get("source_language").As<Napi::String>();
        } else if (options.Has("sourceLanguage") && options.Get("sourceLanguage").IsString()) {
            m_source_language = options.Get("sourceLanguage").As<Napi::String>();
        } else if (options.Has("source-language") && options.Get("source-language").IsString()) {
            m_source_language = options.Get("source-language").As<Napi::String>();
        }

        if (options.Has("punc_model") && options.Get("punc_model").IsString()) {
            m_punc_model = options.Get("punc_model").As<Napi::String>();
        } else if (options.Has("puncModel") && options.Get("puncModel").IsString()) {
            m_punc_model = options.Get("puncModel").As<Napi::String>();
        } else if (options.Has("punc-model") && options.Get("punc-model").IsString()) {
            m_punc_model = options.Get("punc-model").As<Napi::String>();
        }

        if (options.Has("sensitivity") && options.Get("sensitivity").IsString()) {
            m_sensitivity = options.Get("sensitivity").As<Napi::String>();
        }

        if (options.Has("top_k") && options.Get("top_k").IsNumber()) {
            m_top_k = options.Get("top_k").As<Napi::Number>().Int32Value();
        } else if (options.Has("topK") && options.Get("topK").IsNumber()) {
            m_top_k = options.Get("topK").As<Napi::Number>().Int32Value();
        } else if (options.Has("top-k") && options.Get("top-k").IsNumber()) {
            m_top_k = options.Get("top-k").As<Napi::Number>().Int32Value();
        }

        if (options.Has("return_logits")) {
            m_return_logits = options.Get("return_logits").As<Napi::Boolean>() ? 1 : 0;
        } else if (options.Has("returnLogits")) {
            m_return_logits = options.Get("returnLogits").As<Napi::Boolean>() ? 1 : 0;
        } else if (options.Has("return-logits")) {
            m_return_logits = options.Get("return-logits").As<Napi::Boolean>() ? 1 : 0;
        }

        if (options.Has("alt_n") && options.Get("alt_n").IsNumber()) {
            m_alt_n = options.Get("alt_n").As<Napi::Number>().Int32Value();
        } else if (options.Has("altN") && options.Get("altN").IsNumber()) {
            m_alt_n = options.Get("altN").As<Napi::Number>().Int32Value();
        } else if (options.Has("alt-n") && options.Get("alt-n").IsNumber()) {
            m_alt_n = options.Get("alt-n").As<Napi::Number>().Int32Value();
        }

        if (options.Has("suppress_nst") && options.Get("suppress_nst").IsBoolean()) {
            m_suppress_nst = options.Get("suppress_nst").As<Napi::Boolean>();
        } else if (options.Has("suppressNst") && options.Get("suppressNst").IsBoolean()) {
            m_suppress_nst = options.Get("suppressNst").As<Napi::Boolean>();
        } else if (options.Has("suppress-nst") && options.Get("suppress-nst").IsBoolean()) {
            m_suppress_nst = options.Get("suppress-nst").As<Napi::Boolean>();
        }

        if (options.Has("suppress_regex") && options.Get("suppress_regex").IsString()) {
            m_suppress_regex = options.Get("suppress_regex").As<Napi::String>();
        } else if (options.Has("suppressRegex") && options.Get("suppressRegex").IsString()) {
            m_suppress_regex = options.Get("suppressRegex").As<Napi::String>();
        } else if (options.Has("suppress-regex") && options.Get("suppress-regex").IsString()) {
            m_suppress_regex = options.Get("suppress-regex").As<Napi::String>();
        }

        if (options.Has("carry_initial_prompt") && options.Get("carry_initial_prompt").IsBoolean()) {
            m_carry_initial_prompt = options.Get("carry_initial_prompt").As<Napi::Boolean>();
        } else if (options.Has("carryInitialPrompt") && options.Get("carryInitialPrompt").IsBoolean()) {
            m_carry_initial_prompt = options.Get("carryInitialPrompt").As<Napi::Boolean>();
        } else if (options.Has("carry-initial-prompt") && options.Get("carry-initial-prompt").IsBoolean()) {
            m_carry_initial_prompt = options.Get("carry-initial-prompt").As<Napi::Boolean>();
        }

        if (options.Has("att_context_left") && options.Get("att_context_left").IsNumber()) {
            m_att_context_left = options.Get("att_context_left").As<Napi::Number>().Int32Value();
        } else if (options.Has("attContextLeft") && options.Get("attContextLeft").IsNumber()) {
            m_att_context_left = options.Get("attContextLeft").As<Napi::Number>().Int32Value();
        }

        if (options.Has("att_context_right") && options.Get("att_context_right").IsNumber()) {
            m_att_context_right = options.Get("att_context_right").As<Napi::Number>().Int32Value();
        } else if (options.Has("attContextRight") && options.Get("attContextRight").IsNumber()) {
            m_att_context_right = options.Get("attContextRight").As<Napi::Number>().Int32Value();
        }
        
        if (options.Has("hotwords") && options.Get("hotwords").IsString()) {
            m_hotwords = options.Get("hotwords").As<Napi::String>();
        }
        if (options.Has("hotwords_boost") && options.Get("hotwords_boost").IsNumber()) {
            m_hotwords_boost = options.Get("hotwords_boost").As<Napi::Number>().FloatValue();
        } else if (options.Has("hotwords-boost") && options.Get("hotwords-boost").IsNumber()) {
            m_hotwords_boost = options.Get("hotwords-boost").As<Napi::Number>().FloatValue();
        }
        if (options.Has("beam_size") && options.Get("beam_size").IsNumber()) {
            m_beam_size = options.Get("beam_size").As<Napi::Number>().Int32Value();
        } else if (options.Has("beam-size") && options.Get("beam-size").IsNumber()) {
            m_beam_size = options.Get("beam-size").As<Napi::Number>().Int32Value();
        }

        if (options.Has("temperature") && options.Get("temperature").IsNumber()) {
            m_temperature = options.Get("temperature").As<Napi::Number>().FloatValue();
        }
        if (options.Has("seed") && options.Get("seed").IsNumber()) {
            m_seed = options.Get("seed").As<Napi::Number>().Int32Value();
        }
        if (options.Has("top_p") && options.Get("top_p").IsNumber()) {
            m_top_p = options.Get("top_p").As<Napi::Number>().FloatValue();
        } else if (options.Has("top-p") && options.Get("top-p").IsNumber()) {
            m_top_p = options.Get("top-p").As<Napi::Number>().FloatValue();
        }
        if (options.Has("min_p") && options.Get("min_p").IsNumber()) {
            m_min_p = options.Get("min_p").As<Napi::Number>().FloatValue();
        } else if (options.Has("min-p") && options.Get("min-p").IsNumber()) {
            m_min_p = options.Get("min-p").As<Napi::Number>().FloatValue();
        }
        if (options.Has("repetition_penalty") && options.Get("repetition_penalty").IsNumber()) {
            m_repetition_penalty = options.Get("repetition_penalty").As<Napi::Number>().FloatValue();
        } else if (options.Has("repetition-penalty") && options.Get("repetition-penalty").IsNumber()) {
            m_repetition_penalty = options.Get("repetition-penalty").As<Napi::Number>().FloatValue();
        }
        if (options.Has("frequency_penalty") && options.Get("frequency_penalty").IsNumber()) {
            m_frequency_penalty = options.Get("frequency_penalty").As<Napi::Number>().FloatValue();
        } else if (options.Has("frequency-penalty") && options.Get("frequency-penalty").IsNumber()) {
            m_frequency_penalty = options.Get("frequency-penalty").As<Napi::Number>().FloatValue();
        }
        if (options.Has("best_of") && options.Get("best_of").IsNumber()) {
            m_best_of = options.Get("best_of").As<Napi::Number>().Int32Value();
        } else if (options.Has("best-of") && options.Get("best-of").IsNumber()) {
            m_best_of = options.Get("best-of").As<Napi::Number>().Int32Value();
        }
        if (options.Has("max_new_tokens") && options.Get("max_new_tokens").IsNumber()) {
            m_max_new_tokens = options.Get("max_new_tokens").As<Napi::Number>().Int32Value();
        } else if (options.Has("max-new-tokens") && options.Get("max-new-tokens").IsNumber()) {
            m_max_new_tokens = options.Get("max-new-tokens").As<Napi::Number>().Int32Value();
        }
        if (options.Has("punctuation")) {
            m_punctuation = options.Get("punctuation").As<Napi::Boolean>() ? 1 : 0;
        }

        if (options.Has("aligner") && options.Get("aligner").IsString()) {
            m_aligner_model_path = options.Get("aligner").As<Napi::String>();
        } else if (options.Has("aligner_model") && options.Get("aligner_model").IsString()) {
            m_aligner_model_path = options.Get("aligner_model").As<Napi::String>();
        } else if (options.Has("alignerModel") && options.Get("alignerModel").IsString()) {
            m_aligner_model_path = options.Get("alignerModel").As<Napi::String>();
        }
        if (options.Has("debug")) m_debug = options.Get("debug").As<Napi::Boolean>();

        if (options.Has("progressive_update")) m_progressive_update = options.Get("progressive_update").As<Napi::Boolean>();
        if (options.Has("progressive_interval_ms")) m_progressive_interval_ms = options.Get("progressive_interval_ms").As<Napi::Number>().Int32Value();
        if (options.Has("progressive_initial_ms")) m_progressive_initial_ms = options.Get("progressive_initial_ms").As<Napi::Number>().Int32Value();
        
        if (options.Has("vad_model") && options.Get("vad_model").IsString())
            m_vad_model_path = options.Get("vad_model").As<Napi::String>();
        if (options.Has("vad_threshold") && options.Get("vad_threshold").IsNumber())
            m_vad_threshold = options.Get("vad_threshold").As<Napi::Number>().FloatValue();
        if (options.Has("min_mute_chunks") && options.Get("min_mute_chunks").IsNumber())
            m_min_mute_chunks = options.Get("min_mute_chunks").As<Napi::Number>().Int32Value();
        if (options.Has("max_nomute_chunks") && options.Get("max_nomute_chunks").IsNumber())
            m_max_nomute_chunks = options.Get("max_nomute_chunks").As<Napi::Number>().Int32Value();
        if (options.Has("use_vad") && options.Get("use_vad").IsBoolean())
            m_use_vad = options.Get("use_vad").As<Napi::Boolean>();
    }

    ~CrispASRStream() {
        m_running = false;
        m_cv.notify_all();
        if (m_worker_thread.joinable()) m_worker_thread.join();
        if (m_session) {
            crispasr_session_close(m_session);
            m_session = nullptr;
        }
        if (m_vctx) {
            whisper_vad_free(m_vctx);
            m_vctx = nullptr;
        }
        m_aligner.reset();
        
        if (m_tsfn) {
            m_tsfn.Abort();
            m_tsfn.Release();
            m_tsfn = nullptr;
        }
    }

    Napi::Value Start(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsFunction()) {
            Napi::TypeError::New(env, "start() requires a callback function").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        if (m_running) {
            Napi::Error::New(env, "Stream is already running").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        Napi::Function callback = info[0].As<Napi::Function>();
        m_tsfn = Napi::ThreadSafeFunction::New(env, callback, "CrispASRStream", 0, 1);
        m_segment_index = 0;
        m_running = true;
        m_worker_thread = std::thread(&CrispASRStream::Worker, this);
        return env.Undefined();
    }

    Napi::Value AddAudio(const Napi::CallbackInfo& info) {
        Napi::Float32Array arr = info[0].As<Napi::Float32Array>();
        std::lock_guard<std::mutex> lock(m_mutex);
        m_audio_buffer.insert(m_audio_buffer.end(), arr.Data(), arr.Data() + arr.ElementLength());
        m_cv.notify_one();
        return info.Env().Undefined();
    }

    Napi::Value Stop(const Napi::CallbackInfo& info) {
        m_running = false;
        m_cv.notify_all();
        if (m_worker_thread.joinable()) m_worker_thread.join();
        
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_session) {
            crispasr_session_close(m_session);
            m_session = nullptr;
        }
        if (m_vctx) {
            whisper_vad_free(m_vctx);
            m_vctx = nullptr;
        }
        m_aligner.reset();
        std::vector<float>().swap(m_audio_buffer);

        if (m_tsfn) {
            m_tsfn.Release();
            m_tsfn = nullptr;
        }
        return info.Env().Undefined();
    }

    Napi::Value Pause(const Napi::CallbackInfo& info) {
        m_paused = true;
        return info.Env().Undefined();
    }

    Napi::Value Resume(const Napi::CallbackInfo& info) {
        m_paused = false;
        return info.Env().Undefined();
    }

    Napi::Value Finish(const Napi::CallbackInfo& info) {
        m_finishing = true;
        m_cv.notify_all();
        return info.Env().Undefined();
    }

    Napi::Value Release(const Napi::CallbackInfo& info) {
        return Stop(info);
    }

private:
    void emit_segment(crispasr_session_result* res, int64_t start_time, int64_t end_time, int segment_index, const std::string& type, const std::vector<stream_word_item>& align_words = {}) {
        int n_segs = crispasr_session_result_n_segments(res);
        std::string text = "";
        std::vector<stream_word_item> words_list = align_words;
        
        for (int i = 0; i < n_segs; i++) {
            const char* seg_text = crispasr_session_result_segment_text(res, i);
            if (seg_text) {
                if (!text.empty()) text += " ";
                text += seg_text;
            }
        }
        trim_leading_replacement_and_spaces(text);

        if (words_list.empty()) {
            for (int i = 0; i < n_segs; i++) {
                int n_words = crispasr_session_result_n_words(res, i);
                for (int j = 0; j < n_words; j++) {
                    stream_word_item wi;
                    const char* w_text = crispasr_session_result_word_text(res, i, j);
                    wi.word = w_text ? w_text : "";
                    wi.start = start_time + crispasr_session_result_word_t0(res, i, j) * 10;
                    wi.end = start_time + crispasr_session_result_word_t1(res, i, j) * 10;
                    wi.p = crispasr_session_result_word_p(res, i, j);
                    words_list.push_back(wi);
                }
            }
        }

        // Clean out empty, whitespace-only, or replacement character artifact words
        std::vector<stream_word_item> filtered_words;
        filtered_words.reserve(words_list.size());
        for (auto& wi : words_list) {
            trim_leading_replacement_and_spaces(wi.word);
            if (!wi.word.empty() && wi.word != "\xEF\xBF\xBD") {
                filtered_words.push_back(std::move(wi));
            }
        }
        words_list = std::move(filtered_words);
        
        if ((!text.empty() || type == "segment") && m_tsfn) {
            auto cb_data = std::make_tuple(start_time, end_time, text, words_list, type);
            auto callback = [cb_data, segment_index](Napi::Env env, Napi::Function cb) {
                Napi::Object result = Napi::Object::New(env);
                result.Set("type", Napi::String::New(env, std::get<4>(cb_data)));
                result.Set("index", Napi::Number::New(env, segment_index));
                result.Set("start", Napi::Number::New(env, std::get<0>(cb_data)));
                result.Set("end", Napi::Number::New(env, std::get<1>(cb_data)));
                result.Set("text", Napi::String::New(env, std::get<2>(cb_data)));
                
                const auto& wl = std::get<3>(cb_data);
                Napi::Array words_arr = Napi::Array::New(env, wl.size());
                for (size_t k = 0; k < wl.size(); k++) {
                    Napi::Object w_obj = Napi::Object::New(env);
                    w_obj.Set("word", Napi::String::New(env, wl[k].word));
                    w_obj.Set("start", Napi::Number::New(env, wl[k].start / 1000.0));
                    w_obj.Set("end", Napi::Number::New(env, wl[k].end / 1000.0));
                    w_obj.Set("p", Napi::Number::New(env, wl[k].p));
                    words_arr[k] = w_obj;
                }
                result.Set("words", words_arr);
                
                cb.Call({env.Null(), result});
            };
            if (type == "progressive") {
                m_tsfn.NonBlockingCall(callback);
            } else {
                m_tsfn.BlockingCall(callback);
            }
        }
    }

    void emit_silence(double t) {
        if (m_tsfn) {
            m_tsfn.BlockingCall([t](Napi::Env env, Napi::Function cb) {
                Napi::Object result = Napi::Object::New(env);
                result.Set("type", Napi::String::New(env, "silence"));
                result.Set("t", Napi::Number::New(env, t));
                cb.Call({env.Null(), result});
            });
        }
    }

    void Worker() {
        crispasr_open_params_v1 params;
        memset(&params, 0, sizeof(params));
        params.abi_version = 2;
        params.n_threads = m_n_threads;
        params.use_gpu = m_use_gpu ? 1 : 0;
        params.verbosity = m_debug ? 1 : 0;
        params.flash_attn = m_flash_attn ? 1 : 0;
        params.n_gpu_layers = m_n_gpu_layers;

        const char* backend_ptr = m_backend_name.empty() ? nullptr : m_backend_name.c_str();
        m_session = crispasr_session_open_with_params(m_model_path.c_str(), backend_ptr, &params);
        if (!m_session) {
            if (m_tsfn) {
                m_tsfn.BlockingCall([](Napi::Env env, Napi::Function cb) {
                    cb.Call({Napi::Error::New(env, "Failed to open CrispASR session").Value(), env.Null()});
                });
            }
            m_running = false;
            return;
        }

        if (!m_running) {
            return;
        }

        crispasr_session_set_translate(m_session, m_translate ? 1 : 0);
        if (!m_target_language.empty()) {
            crispasr_session_set_target_language(m_session, m_target_language.c_str());
        }
        if (!m_context.empty()) {
            crispasr_session_set_ask(m_session, m_context.c_str());
        }
        if (!m_hotwords.empty()) {
            crispasr_session_set_hotwords(m_session, m_hotwords.c_str(), m_hotwords_boost);
        }
        if (m_beam_size > 1) {
            crispasr_session_set_beam_size(m_session, m_beam_size);
        }
        if (m_temperature >= 0.0f) {
            crispasr_session_set_temperature(m_session, m_temperature, m_seed >= 0 ? static_cast<uint64_t>(m_seed) : 0ULL);
        }
        if (m_top_p >= 0.0f) {
            crispasr_session_set_top_p(m_session, m_top_p);
        }
        if (m_min_p >= 0.0f) {
            crispasr_session_set_min_p(m_session, m_min_p);
        }
        if (m_repetition_penalty >= 0.0f) {
            crispasr_session_set_repetition_penalty(m_session, m_repetition_penalty);
        }
        if (m_frequency_penalty >= 0.0f) {
            crispasr_session_set_frequency_penalty(m_session, m_frequency_penalty);
        }
        if (m_best_of > 0) {
            crispasr_session_set_best_of(m_session, m_best_of);
        }
        if (m_max_new_tokens > 0) {
            crispasr_session_set_max_new_tokens(m_session, m_max_new_tokens);
        }
        if (m_punctuation >= 0) {
            crispasr_session_set_punctuation(m_session, m_punctuation);
        }
        if (!m_source_language.empty()) {
            crispasr_session_set_source_language(m_session, m_source_language.c_str());
        }
        if (!m_punc_model.empty()) {
            crispasr_session_set_punc_model(m_session, m_punc_model.c_str());
        }
        if (!m_sensitivity.empty()) {
            crispasr_session_set_sensitivity(m_session, m_sensitivity.c_str());
        }
        if (m_top_k > 0) {
            crispasr_session_set_top_k(m_session, m_top_k);
        }
        if (m_return_logits >= 0) {
            crispasr_session_set_return_logits(m_session, m_return_logits);
        }
        if (m_alt_n > 0) {
            crispasr_session_set_alt_n(m_session, m_alt_n);
        }
        if (m_suppress_nst || !m_suppress_regex.empty() || m_carry_initial_prompt) {
            crispasr_session_set_whisper_decode_extras(m_session, m_suppress_nst ? 1 : 0,
                                                       m_suppress_regex.empty() ? nullptr : m_suppress_regex.c_str(),
                                                       m_carry_initial_prompt ? 1 : 0);
        }
        if (m_att_context_left >= 0 || m_att_context_right >= 0) {
            crispasr_session_set_parakeet_att_context(m_session, m_att_context_left, m_att_context_right);
        }

        if (!m_aligner_model_path.empty()) {
            m_aligner = std::make_unique<qwen3_asr::ForcedAligner>();
            if (!m_aligner->load_model(m_aligner_model_path, m_use_gpu, m_debug)) {
                fprintf(stderr, "[STREAM] Warning: Failed to load Forced Align model\n");
            }
        }

        if (!m_vad_model_path.empty()) {
            whisper_vad_context_params vad_ctx_params = whisper_vad_default_context_params();
            vad_ctx_params.n_threads = m_n_threads;
            vad_ctx_params.use_gpu = false;
            m_vctx = whisper_vad_init_from_file_with_params(m_vad_model_path.c_str(), vad_ctx_params);
        }

        const int sample_rate = 16000;
        const int chunk_samples = (m_chunk_size_ms * sample_rate) / 1000;
        const char* lang_ptr = m_language.empty() ? nullptr : m_language.c_str();
        
        std::vector<float> speech_buffer;
        std::vector<float> pre_speech_cache;
        std::deque<float> accumulated_samples;
        bool in_speech = false;
        int silence_chunk_count = 0;
        int speech_chunk_count = 0;
        int64_t speech_start_sample = 0;
        int64_t total_samples_received = 0;
        
        int64_t last_progressive_sample = 0;
        double last_silence_time = 0.0;

        while (m_running) {
            bool is_finishing_step = false;
            bool timeout = false;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                timeout = !m_cv.wait_for(lock, std::chrono::milliseconds(1000), [this] {
                    return !m_running || !m_audio_buffer.empty() || m_finishing;
                });
                
                if (!m_running) break;
                is_finishing_step = m_finishing;
                
                if (!m_paused) {
                    accumulated_samples.insert(accumulated_samples.end(), m_audio_buffer.begin(), m_audio_buffer.end());
                    m_audio_buffer.clear();
                }
            }
            
            if (timeout && m_running) {
                double current_time_sec = (double)total_samples_received / sample_rate;
                emit_silence(current_time_sec);
                last_silence_time = current_time_sec;
                continue;
            }
            
            size_t processed_samples = 0;
            size_t n_chunks = accumulated_samples.size() / chunk_samples;
            std::vector<bool> chunks_is_speech;
            chunks_is_speech.reserve(n_chunks);
            for (size_t c = 0; c < n_chunks; ++c) {
                size_t offset = c * chunk_samples;
                std::vector<float> vad_chunk(accumulated_samples.begin() + offset, accumulated_samples.begin() + offset + chunk_samples);
                bool is_speech = true;
                if (m_use_vad) {
                    if (!m_vad_model_path.empty() && m_vctx) {
                        whisper_vad_detect_speech(m_vctx, vad_chunk.data(), vad_chunk.size());
                        float speech_prob = 0.0f;
                        int n_probs = whisper_vad_n_probs(m_vctx);
                        if (n_probs > 0) {
                            const float* probs = whisper_vad_probs(m_vctx);
                            for (int k = 0; k < n_probs; k++) if (probs[k] > speech_prob) speech_prob = probs[k];
                        }
                        is_speech = speech_prob >= m_vad_threshold;
                    } else {
                        float energy = 0.0f;
                        for (int j = 0; j < chunk_samples; j++) energy += vad_chunk[j] * vad_chunk[j];
                        energy = std::sqrt(energy / chunk_samples);
                        is_speech = energy > 0.01f;
                    }
                }
                chunks_is_speech.push_back(is_speech);
            }

            auto will_segment_end = [&](size_t start_chunk, bool curr_in_speech, int curr_silence_count, int curr_speech_count) -> bool {
                bool temp_in_speech = curr_in_speech;
                int temp_silence_count = curr_silence_count;
                int temp_speech_count = curr_speech_count;
                
                for (size_t next_c = start_chunk; next_c < chunks_is_speech.size(); ++next_c) {
                    bool next_is_speech = chunks_is_speech[next_c];
                    if (next_is_speech) {
                        if (!temp_in_speech) {
                            temp_in_speech = true;
                        }
                        temp_silence_count = 0;
                        temp_speech_count++;
                        if (temp_speech_count >= m_max_nomute_chunks) {
                            return true;
                        }
                    } else {
                        if (temp_in_speech) {
                            temp_silence_count++;
                            if (temp_silence_count > m_min_mute_chunks) {
                                return true;
                            }
                        }
                    }
                }
                return false;
            };

            for (size_t c = 0; c < n_chunks; ++c) {
                size_t i = c * chunk_samples;
                std::vector<float> vad_chunk(accumulated_samples.begin() + i, accumulated_samples.begin() + i + chunk_samples);
                
                bool is_speech = chunks_is_speech[c];
                
                if (is_speech) {
                    if (!in_speech) {
                        speech_start_sample = total_samples_received + i - pre_speech_cache.size();
                        if (speech_start_sample < 0) {
                            speech_start_sample = 0;
                        }
                        last_progressive_sample = speech_start_sample;
                        m_segment_index++;
                        speech_buffer.insert(speech_buffer.end(), pre_speech_cache.begin(), pre_speech_cache.end());
                        pre_speech_cache.clear();
                    }
                    speech_buffer.insert(speech_buffer.end(), vad_chunk.begin(), vad_chunk.end());
                    in_speech = true;
                    silence_chunk_count = 0;
                    speech_chunk_count++;
                    
                    if (m_progressive_update) {
                        int64_t current_sample = total_samples_received + i + chunk_samples;
                        int64_t elapsed_since_start_ms = ((current_sample - speech_start_sample) * 1000) / sample_rate;
                        
                        if (elapsed_since_start_ms >= m_progressive_initial_ms) {
                            int64_t elapsed_since_last_prog_ms = ((current_sample - last_progressive_sample) * 1000) / sample_rate;
                            if (elapsed_since_last_prog_ms >= m_progressive_interval_ms) {
                                bool has_pending_audio = false;
                                {
                                    std::lock_guard<std::mutex> lock(m_mutex);
                                    has_pending_audio = !m_audio_buffer.empty();
                                }
                                bool skip_progressive = is_finishing_step || 
                                                        has_pending_audio || 
                                                        will_segment_end(c + 1, in_speech, silence_chunk_count, speech_chunk_count);
                                if (!skip_progressive) {
                                    std::vector<float> transcribe_pcm = speech_buffer;
                                    if (transcribe_pcm.size() < 32000 && m_backend_name != "whisper") {
                                        transcribe_pcm.resize(32000, 0.0f);
                                    }
                                    crispasr_session_result* res = crispasr_session_transcribe_lang(m_session, transcribe_pcm.data(), transcribe_pcm.size(), lang_ptr);
                                    if (res) {
                                        int64_t chunk_start_ms = (speech_start_sample * 1000) / sample_rate;
                                        int64_t chunk_end_ms = (current_sample * 1000) / sample_rate;
                                        std::vector<stream_word_item> aligned_words;
                                        if (m_aligner && !m_aligner_model_path.empty()) {
                                            int n_segs = crispasr_session_result_n_segments(res);
                                            std::string total_txt = "";
                                            for (int k = 0; k < n_segs; k++) {
                                                const char* s_text = crispasr_session_result_segment_text(res, k);
                                                if (s_text) {
                                                    if (!total_txt.empty()) total_txt += " ";
                                                    total_txt += s_text;
                                                }
                                            }
                                            if (!total_txt.empty()) {
                                                std::string detected_lang = m_language.empty() ? "zh" : m_language;
                                                std::string clean_txt = extract_clean_text_if_json(total_txt);
                                                trim_leading_replacement_and_spaces(clean_txt);
                                                auto align_res = m_aligner->align(transcribe_pcm.data(), transcribe_pcm.size(), clean_txt, detected_lang);
                                                if (align_res.success) {
                                                    for (const auto& w : align_res.words) {
                                                        stream_word_item wi;
                                                        wi.word = w.word;
                                                        wi.start = chunk_start_ms + static_cast<int64_t>(w.start * 1000.0);
                                                        wi.end = chunk_start_ms + static_cast<int64_t>(w.end * 1000.0);
                                                        wi.p = 0.99f;
                                                        aligned_words.push_back(wi);
                                                    }
                                                }
                                            }
                                        }
                                        emit_segment(res, chunk_start_ms, chunk_end_ms, m_segment_index, "progressive", aligned_words);
                                        crispasr_session_result_free(res);
                                    }
                                    last_progressive_sample = current_sample;
                                }
                            }
                        }
                    }
                    
                    if (speech_chunk_count >= m_max_nomute_chunks) {
                        int64_t current_sample = total_samples_received + i + chunk_samples;
                        std::vector<float> transcribe_pcm = speech_buffer;
                        if (transcribe_pcm.size() < 32000 && m_backend_name != "whisper") {
                            transcribe_pcm.resize(32000, 0.0f);
                        }
                        crispasr_session_result* res = crispasr_session_transcribe_lang(m_session, transcribe_pcm.data(), transcribe_pcm.size(), lang_ptr);
                        if (res) {
                            int64_t chunk_start_ms = (speech_start_sample * 1000) / sample_rate;
                            int64_t chunk_end_ms = (current_sample * 1000) / sample_rate;
                            std::vector<stream_word_item> aligned_words;
                            if (m_aligner && !m_aligner_model_path.empty()) {
                                int n_segs = crispasr_session_result_n_segments(res);
                                std::string total_txt = "";
                                for (int k = 0; k < n_segs; k++) {
                                    const char* s_text = crispasr_session_result_segment_text(res, k);
                                    if (s_text) {
                                        if (!total_txt.empty()) total_txt += " ";
                                        total_txt += s_text;
                                    }
                                }
                                if (!total_txt.empty()) {
                                    std::string detected_lang = m_language.empty() ? "zh" : m_language;
                                    std::string clean_txt = extract_clean_text_if_json(total_txt);
                                    trim_leading_replacement_and_spaces(clean_txt);
                                    auto align_res = m_aligner->align(transcribe_pcm.data(), transcribe_pcm.size(), clean_txt, detected_lang);
                                    if (align_res.success) {
                                        for (const auto& w : align_res.words) {
                                            stream_word_item wi;
                                            wi.word = w.word;
                                            wi.start = chunk_start_ms + static_cast<int64_t>(w.start * 1000.0);
                                            wi.end = chunk_start_ms + static_cast<int64_t>(w.end * 1000.0);
                                            wi.p = 0.99f;
                                            aligned_words.push_back(wi);
                                        }
                                    }
                                }
                            }
                            emit_segment(res, chunk_start_ms, chunk_end_ms, m_segment_index, "segment", aligned_words);
                            crispasr_session_result_free(res);
                        }
                        speech_buffer.clear();
                        speech_chunk_count = 0;
                        speech_start_sample = current_sample;
                        last_progressive_sample = current_sample;
                        m_segment_index++;
                        silence_chunk_count = 0;
                        in_speech = true;
                    }
                } else {
                    if (in_speech) {
                        silence_chunk_count++;
                        if (silence_chunk_count <= m_min_mute_chunks) {
                            speech_buffer.insert(speech_buffer.end(), vad_chunk.begin(), vad_chunk.end());
                        }
                        if (silence_chunk_count > m_min_mute_chunks) {
                            in_speech = false;
                            int64_t speech_end_sample = total_samples_received + i + chunk_samples;
                            int64_t actual_end_sample = speech_end_sample - (silence_chunk_count * chunk_samples);
                            if (actual_end_sample < speech_start_sample) {
                                actual_end_sample = speech_end_sample;
                            }

                            std::vector<float> transcribe_pcm = speech_buffer;
                            size_t silence_samples = std::min(silence_chunk_count, m_min_mute_chunks) * chunk_samples;
                            size_t post_speech_pad_samples = (300 * sample_rate) / 1000;
                            if (silence_samples > post_speech_pad_samples) {
                                size_t trim_samples = silence_samples - post_speech_pad_samples;
                                if (transcribe_pcm.size() > trim_samples) {
                                    transcribe_pcm.resize(transcribe_pcm.size() - trim_samples);
                                }
                            }

                            if (transcribe_pcm.size() < 32000 && m_backend_name != "whisper") {
                                  transcribe_pcm.resize(32000, 0.0f);
                            }
                            crispasr_session_result* res = crispasr_session_transcribe_lang(m_session, transcribe_pcm.data(), transcribe_pcm.size(), lang_ptr);
                            if (res) {
                                int64_t chunk_start_ms = (speech_start_sample * 1000) / sample_rate;
                                int64_t chunk_end_ms = (actual_end_sample * 1000) / sample_rate;
                                std::vector<stream_word_item> aligned_words;
                                if (m_aligner && !m_aligner_model_path.empty()) {
                                    int n_segs = crispasr_session_result_n_segments(res);
                                    std::string total_txt = "";
                                    for (int k = 0; k < n_segs; k++) {
                                        const char* s_text = crispasr_session_result_segment_text(res, k);
                                        if (s_text) {
                                            if (!total_txt.empty()) total_txt += " ";
                                            total_txt += s_text;
                                        }
                                    }
                                    if (!total_txt.empty()) {
                                        std::string detected_lang = m_language.empty() ? "zh" : m_language;
                                        std::string clean_txt = extract_clean_text_if_json(total_txt);
                                        trim_leading_replacement_and_spaces(clean_txt);
                                        auto align_res = m_aligner->align(transcribe_pcm.data(), transcribe_pcm.size(), clean_txt, detected_lang);
                                        if (align_res.success) {
                                            for (const auto& w : align_res.words) {
                                                stream_word_item wi;
                                                wi.word = w.word;
                                                wi.start = chunk_start_ms + static_cast<int64_t>(w.start * 1000.0);
                                                wi.end = chunk_start_ms + static_cast<int64_t>(w.end * 1000.0);
                                                wi.p = 0.99f;
                                                aligned_words.push_back(wi);
                                            }
                                        }
                                    }
                                }
                                emit_segment(res, chunk_start_ms, chunk_end_ms, m_segment_index, "segment", aligned_words);
                                crispasr_session_result_free(res);
                            }
                            speech_buffer.clear();
                            speech_chunk_count = 0;
                            pre_speech_cache.clear();
                        }
                    }
                    
                    if (!in_speech) {
                        pre_speech_cache.insert(pre_speech_cache.end(), vad_chunk.begin(), vad_chunk.end());
                        size_t pre_speech_pad_samples = (300 * sample_rate) / 1000;
                        if (pre_speech_cache.size() > pre_speech_pad_samples) {
                            pre_speech_cache.erase(pre_speech_cache.begin(), pre_speech_cache.end() - pre_speech_pad_samples);
                        }

                        double current_time_sec = (double)(total_samples_received + i + chunk_samples) / sample_rate;
                        if (current_time_sec - last_silence_time >= 1.0) {
                            emit_silence(current_time_sec);
                            last_silence_time = current_time_sec;
                        }
                    }
                }
                processed_samples = i + chunk_samples;
            }
            
            if (processed_samples > 0) {
                total_samples_received += processed_samples;
                accumulated_samples.erase(accumulated_samples.begin(), accumulated_samples.begin() + processed_samples);
            }
            
            if (is_finishing_step) {
                speech_buffer.insert(speech_buffer.end(), accumulated_samples.begin(), accumulated_samples.end());
                int64_t final_sample = total_samples_received + accumulated_samples.size();
                int64_t start = in_speech ? (speech_start_sample * 1000) / sample_rate : (total_samples_received * 1000) / sample_rate;
                if (!speech_buffer.empty()) {
                    if (!in_speech) {
                        m_segment_index++;
                    }
                    std::vector<float> transcribe_pcm = speech_buffer;
                    if (transcribe_pcm.size() < 32000 && m_backend_name != "whisper") {
                        transcribe_pcm.resize(32000, 0.0f);
                    }
                    crispasr_session_result* res = crispasr_session_transcribe_lang(m_session, transcribe_pcm.data(), transcribe_pcm.size(), lang_ptr);
                    if (res) {
                        std::vector<stream_word_item> aligned_words;
                        if (m_aligner && !m_aligner_model_path.empty()) {
                            int n_segs = crispasr_session_result_n_segments(res);
                            std::string total_txt = "";
                            for (int k = 0; k < n_segs; k++) {
                                const char* s_text = crispasr_session_result_segment_text(res, k);
                                if (s_text) {
                                    if (!total_txt.empty()) total_txt += " ";
                                    total_txt += s_text;
                                }
                            }
                            if (!total_txt.empty()) {
                                std::string detected_lang = m_language.empty() ? "zh" : m_language;
                                std::string clean_txt = extract_clean_text_if_json(total_txt);
                                trim_leading_replacement_and_spaces(clean_txt);
                                auto align_res = m_aligner->align(transcribe_pcm.data(), transcribe_pcm.size(), clean_txt, detected_lang);
                                if (align_res.success) {
                                    for (const auto& w : align_res.words) {
                                        stream_word_item wi;
                                        wi.word = w.word;
                                        wi.start = start + static_cast<int64_t>(w.start * 1000.0);
                                        wi.end = start + static_cast<int64_t>(w.end * 1000.0);
                                        wi.p = 0.99f;
                                        aligned_words.push_back(wi);
                                    }
                                }
                            }
                        }
                        emit_segment(res, start, (final_sample * 1000) / sample_rate, m_segment_index, "segment", aligned_words);
                        crispasr_session_result_free(res);
                    }
                }
                break;
            }
        }

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_session) {
                crispasr_session_close(m_session);
                m_session = nullptr;
            }
            m_audio_buffer.clear();
        }

        if (m_tsfn) {
            m_tsfn.BlockingCall([](Napi::Env env, Napi::Function cb) {
                Napi::Object result = Napi::Object::New(env);
                result.Set("type", Napi::String::New(env, "end"));
                cb.Call({env.Null(), result});
            });
        }
    }

    std::string m_model_path;
    std::string m_backend_name;
    std::string m_language;
    bool m_translate = false;
    std::string m_target_language = "";
    std::string m_context = "";
    std::string m_hotwords = "";
    float m_hotwords_boost = 1.5f;
    int m_beam_size = 1;
    float m_temperature = -1.0f;
    int m_seed = -1;
    float m_top_p = -1.0f;
    float m_min_p = -1.0f;
    float m_repetition_penalty = -1.0f;
    float m_frequency_penalty = -1.0f;
    int m_best_of = -1;
    int m_max_new_tokens = -1;
    int m_punctuation = -1;
    std::string m_aligner_model_path = "";
    std::string m_source_language = "";
    std::string m_punc_model = "";
    std::string m_sensitivity = "";
    int m_top_k = -1;
    int m_return_logits = -1;
    int m_alt_n = -1;
    bool m_suppress_nst = false;
    std::string m_suppress_regex = "";
    bool m_carry_initial_prompt = false;
    int m_att_context_left = -1;
    int m_att_context_right = -1;
    bool m_debug = false;
    int m_n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
    bool m_use_gpu = true;
    bool m_flash_attn = true;
    int m_n_gpu_layers = -1;
    int m_chunk_size_ms = 2000;
    bool m_progressive_update = false;
    int m_progressive_interval_ms = 500;
    int m_progressive_initial_ms = 1000;
    int m_segment_index = 0;
    std::string m_vad_model_path;
    float m_vad_threshold = 0.5f;
    int m_min_mute_chunks = 30;
    int m_max_nomute_chunks = 1875;
    bool m_use_vad = true;
    whisper_vad_context* m_vctx = nullptr;
    crispasr_session* m_session = nullptr;
    std::unique_ptr<qwen3_asr::ForcedAligner> m_aligner;
    std::vector<float> m_audio_buffer;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::thread m_worker_thread;
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_paused{false};
    std::atomic<bool> m_finishing{false};
    Napi::ThreadSafeFunction m_tsfn;
};

// ============================================================================
// CrispASR / Qwen3-TTS Implementation
// ============================================================================

#if defined(__APPLE__)
extern "C" {
    void* objc_autoreleasePoolPush(void);
    void  objc_autoreleasePoolPop(void* pool);
}
struct AddonAutoReleaseScope {
    void* pool;
    AddonAutoReleaseScope() : pool(objc_autoreleasePoolPush()) {}
    ~AddonAutoReleaseScope() { objc_autoreleasePoolPop(pool); }
};
#else
struct AddonAutoReleaseScope {};
#endif

class CrispasrTTSWorker : public Napi::AsyncWorker {
public:
    CrispasrTTSWorker(Napi::Function& callback,
                      std::string model_path,
                      std::string codec_model_path,
                      std::string text,
                      std::string voice,
                      std::string ref_text,
                      std::string instruct,
                      std::string language,
                      std::string output_path,
                      int n_threads,
                      bool use_gpu,
                      float temperature,
                      uint64_t seed,
                      bool reuse_instance,
                      int64_t auto_release_ms,
                      Napi::Env env,
                      std::string backend_name = "",
                      bool watermark = false,
                      float speed = 1.0f,
                      int steps = -1,
                      float cfg_scale = -1.0f,
                      float noise_temp = -1.0f,
                      int num_candidates = -1,
                      int min_speech_tokens = -1,
                      int max_speech_tokens = -1,
                      float exaggeration = -1.0f,
                      std::string phonemes = "",
                      int pad_silence_ms = 0,
                      std::string speaker_identity = "",
                      std::string g2p_dict = "",
                      std::string ref_lang = "")
        : Napi::AsyncWorker(callback),
          m_model_path(std::move(model_path)),
          m_codec_model_path(std::move(codec_model_path)),
          m_text(std::move(text)),
          m_voice(std::move(voice)),
          m_ref_text(std::move(ref_text)),
          m_instruct(std::move(instruct)),
          m_language(std::move(language)),
          m_output_path(std::move(output_path)),
          m_n_threads(n_threads),
          m_use_gpu(use_gpu),
          m_temperature(temperature),
          m_seed(seed),
          m_reuse_instance(reuse_instance),
          m_auto_release_ms(auto_release_ms),
          m_backend_name(std::move(backend_name)),
          m_watermark(watermark),
          m_speed(speed),
          m_steps(steps),
          m_cfg_scale(cfg_scale),
          m_noise_temp(noise_temp),
          m_num_candidates(num_candidates),
          m_min_speech_tokens(min_speech_tokens),
          m_max_speech_tokens(max_speech_tokens),
          m_exaggeration(exaggeration),
          m_phonemes(std::move(phonemes)),
          m_pad_silence_ms(pad_silence_ms),
          m_speaker_identity(std::move(speaker_identity)),
          m_g2p_dict(std::move(g2p_dict)),
          m_ref_lang(std::move(ref_lang)) {}

    void Execute() override {
        AddonAutoReleaseScope ar_scope;
        if (m_model_path.empty()) {
            SetError("Model path is required for crispasrTTS");
            return;
        }
        if (m_text.empty()) {
            SetError("Text is required for crispasrTTS");
            return;
        }

        auto& cache = ModelCache::instance();
        std::unique_lock<std::recursive_mutex> type_lock(cache.mutex(ModelType::QWEN3_TTS));

        crispasr_session* session = nullptr;
        bool owned = false;

        std::string cache_tag = m_backend_name.empty() ? m_codec_model_path : (m_backend_name + "|" + m_codec_model_path);
        if (m_reuse_instance) {
            session = static_cast<crispasr_session*>(
                cache.acquire(ModelType::QWEN3_TTS, m_model_path, m_use_gpu, cache_tag));
        }

        if (!session) {
            crispasr_open_params_v1 params;
            memset(&params, 0, sizeof(params));
            params.abi_version = 2;
            params.n_threads = m_n_threads;
            params.use_gpu = m_use_gpu ? 1 : 0;
            params.verbosity = 0;
            params.flash_attn = 1;
            params.n_gpu_layers = -1;

            const char* backend_ptr = m_backend_name.empty() ? nullptr : m_backend_name.c_str();
            session = crispasr_session_open_with_params(m_model_path.c_str(), backend_ptr, &params);
            if (!session) {
                SetError("Failed to initialize CrispASR TTS session from " + m_model_path);
                return;
            }

            std::string codec_path = m_codec_model_path;
            if (codec_path.empty()) {
                codec_path = crisp_discover_codec(m_model_path);
            }
            if (!codec_path.empty()) {
                crispasr_session_set_codec_path(session, codec_path.c_str());
            }

            if (m_reuse_instance) {
                cache.store(ModelType::QWEN3_TTS, session, m_model_path, m_use_gpu, cache_tag, m_auto_release_ms);
            } else {
                owned = true;
            }
        }

        if (!m_codec_model_path.empty()) {
            crispasr_session_set_codec_path(session, m_codec_model_path.c_str());
        }

        if (m_temperature >= 0.0f) {
            crispasr_session_set_temperature(session, m_temperature, m_seed);
        }
        crispasr_session_set_tts_seed(session, m_seed);

        if (!m_language.empty()) {
            crispasr_session_set_target_language(session, m_language.c_str());
        }
        if (!m_ref_lang.empty()) {
            crispasr_session_set_tts_reference_language(session, m_ref_lang.c_str());
        }

        if (!m_voice.empty() || !m_ref_text.empty()) {
            crispasr_session_set_voice(session, m_voice.c_str(), m_ref_text.empty() ? nullptr : m_ref_text.c_str());
        }

        if (!m_instruct.empty()) {
            crispasr_session_set_instruct(session, m_instruct.c_str());
        }

        if (!m_phonemes.empty()) {
            crispasr_session_set_tts_phonemes(session, m_phonemes.c_str());
        }

        if (m_pad_silence_ms > 0) {
            crispasr_session_set_tts_pad_silence_ms(session, m_pad_silence_ms);
        }

        if (m_steps > 0) {
            crispasr_session_set_tts_steps(session, m_steps);
        }

        if (m_cfg_scale >= 0.0f) {
            crispasr_session_set_cfg_weight(session, m_cfg_scale);
        }

        if (m_noise_temp >= 0.0f) {
            crispasr_session_set_tts_noise_temp(session, m_noise_temp);
        }

        if (m_num_candidates > 0) {
            crispasr_session_set_tts_num_candidates(session, m_num_candidates);
        }

        if (m_speed > 0.0f && m_speed != 1.0f) {
            crispasr_session_set_length_scale(session, 1.0f / m_speed);
        }

        if (m_min_speech_tokens >= 0) {
            crispasr_session_set_min_speech_tokens(session, m_min_speech_tokens);
        }
        if (m_max_speech_tokens > 0) {
            crispasr_session_set_max_speech_tokens(session, m_max_speech_tokens);
        }

        if (m_exaggeration >= 0.0f) {
            crispasr_session_set_exaggeration(session, m_exaggeration);
        }

        if (!m_speaker_identity.empty()) {
            crispasr_session_set_speaker_identity(session, m_speaker_identity.c_str());
        }

        if (!m_g2p_dict.empty()) {
            crispasr_session_set_g2p_dict(session, m_g2p_dict.c_str());
        }

        int n_samples = 0;
        float* pcm = nullptr;

        if (m_watermark) {
            pcm = crispasr_session_synthesize(session, m_text.c_str(), &n_samples);
        } else {
            crispasr_session_accept_marking_responsibility(session, "Caller accepted marking responsibility via addon options");
            pcm = crispasr_session_synthesize_raw(session, m_text.c_str(), &n_samples);
            if (!pcm) {
                pcm = crispasr_session_synthesize(session, m_text.c_str(), &n_samples);
            }
        }

        if (!pcm || n_samples <= 0) {
            type_lock.unlock();
            if (owned) {
                crispasr_session_close(session);
            } else {
                cache.markIdle(ModelType::QWEN3_TTS);
            }
            SetError("CrispASR TTS synthesis failed or returned empty audio");
            return;
        }

        m_pcm_samples.assign(pcm, pcm + n_samples);
        crispasr_pcm_free(pcm);

        int native_sr = crispasr_session_output_sample_rate(session);
        if (native_sr <= 0) {
            native_sr = 24000;
        }
        m_sample_rate = native_sr;
        const char* b = crispasr_session_backend(session);
        m_backend = b ? b : "";

        type_lock.unlock();
        if (owned) {
            crispasr_session_close(session);
        } else {
            cache.markIdle(ModelType::QWEN3_TTS);
        }

        if (!m_output_path.empty()) {
            std::ofstream fout(m_output_path, std::ios::binary);
            if (fout.is_open()) {
                uint32_t sample_rate = static_cast<uint32_t>(m_sample_rate);
                uint16_t num_channels = 1;
                uint16_t bits_per_sample = 16;
                uint32_t byte_rate = sample_rate * num_channels * (bits_per_sample / 8);
                uint16_t block_align = num_channels * (bits_per_sample / 8);
                uint32_t data_size = n_samples * sizeof(int16_t);
                uint32_t chunk_size = 36 + data_size;

                fout.write("RIFF", 4);
                fout.write(reinterpret_cast<const char*>(&chunk_size), 4);
                fout.write("WAVE", 4);
                fout.write("fmt ", 4);
                uint32_t subchunk1_size = 16;
                uint16_t audio_format = 1;
                fout.write(reinterpret_cast<const char*>(&subchunk1_size), 4);
                fout.write(reinterpret_cast<const char*>(&audio_format), 2);
                fout.write(reinterpret_cast<const char*>(&num_channels), 2);
                fout.write(reinterpret_cast<const char*>(&sample_rate), 4);
                fout.write(reinterpret_cast<const char*>(&byte_rate), 4);
                fout.write(reinterpret_cast<const char*>(&block_align), 2);
                fout.write(reinterpret_cast<const char*>(&bits_per_sample), 2);
                fout.write("data", 4);
                fout.write(reinterpret_cast<const char*>(&data_size), 4);

                for (int i = 0; i < n_samples; ++i) {
                    float s = std::max(-1.0f, std::min(1.0f, m_pcm_samples[i]));
                    int16_t val = (s < 0) ? static_cast<int16_t>(s * 32768.0f) : static_cast<int16_t>(s * 32767.0f);
                    fout.write(reinterpret_cast<const char*>(&val), sizeof(val));
                }
                fout.close();
            }
        }
    }

    void OnOK() override {
        try {
            Napi::HandleScope scope(Env());
            Napi::Object result = Napi::Object::New(Env());

            result.Set("sampleRate", Napi::Number::New(Env(), m_sample_rate));
            result.Set("n_samples", Napi::Number::New(Env(), m_pcm_samples.size()));
            result.Set("duration", Napi::Number::New(Env(), static_cast<double>(m_pcm_samples.size()) / static_cast<double>(m_sample_rate)));
            result.Set("watermarked", Napi::Boolean::New(Env(), m_watermark));
            result.Set("backend", Napi::String::New(Env(), m_backend));

            if (!m_output_path.empty()) {
                result.Set("path", Napi::String::New(Env(), m_output_path));
            } else {
                Napi::Float32Array buffer = Napi::Float32Array::New(Env(), m_pcm_samples.size());
                for (size_t i = 0; i < m_pcm_samples.size(); ++i) {
                    buffer[i] = m_pcm_samples[i];
                }
                result.Set("buffer", buffer);
            }

            Callback().Call({Env().Null(), result});
        } catch (const Napi::Error& e) {
            fprintf(stderr, "[CrispasrTTSWorker] Caught Napi::Error in OnOK during teardown: %s\n", e.what());
        } catch (const std::exception& e) {
            fprintf(stderr, "[CrispasrTTSWorker] Caught std::exception in OnOK: %s\n", e.what());
        } catch (...) {
            fprintf(stderr, "[CrispasrTTSWorker] Suppressed unknown exception in OnOK\n");
        }
    }

    void OnError(const Napi::Error& e) override {
        try {
            Napi::HandleScope scope(Env());
            Callback().Call({e.Value(), Env().Undefined()});
        } catch (...) {
            fprintf(stderr, "[CrispasrTTSWorker] Suppressed error callback in OnError during teardown\n");
        }
    }

private:
    std::string m_model_path;
    std::string m_codec_model_path;
    std::string m_text;
    std::string m_voice;
    std::string m_ref_text;
    std::string m_instruct;
    std::string m_language;
    std::string m_output_path;
    int m_n_threads;
    bool m_use_gpu;
    float m_temperature;
    uint64_t m_seed;
    bool m_reuse_instance;
    int64_t m_auto_release_ms;
    std::string m_backend_name;
    bool m_watermark = false;
    float m_speed = 1.0f;
    int m_steps = -1;
    float m_cfg_scale = -1.0f;
    float m_noise_temp = -1.0f;
    int m_num_candidates = -1;
    int m_min_speech_tokens = -1;
    int m_max_speech_tokens = -1;
    float m_exaggeration = -1.0f;
    std::string m_phonemes;
    int m_pad_silence_ms = 0;
    std::string m_speaker_identity;
    std::string m_g2p_dict;
    std::string m_ref_lang;
    int m_sample_rate = 24000;
    std::string m_backend;
    std::vector<float> m_pcm_samples;
};

Napi::Value crispasrTTS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2 || !info[0].IsObject() || !info[1].IsFunction()) {
        Napi::TypeError::New(env, "Usage: crispasrTTS(options, callback)").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Object options = info[0].As<Napi::Object>();
    Napi::Function callback = info[1].As<Napi::Function>();

    std::string model_path = (options.Has("model") && options.Get("model").IsString())
                                 ? options.Get("model").As<Napi::String>().Utf8Value()
                                 : std::string("");
    std::string codec_model_path;
    if (options.Has("codec_model") && options.Get("codec_model").IsString()) {
        codec_model_path = options.Get("codec_model").As<Napi::String>().Utf8Value();
    } else if (options.Has("codecModel") && options.Get("codecModel").IsString()) {
        codec_model_path = options.Get("codecModel").As<Napi::String>().Utf8Value();
    }

    std::string text = (options.Has("text") && options.Get("text").IsString())
                           ? options.Get("text").As<Napi::String>().Utf8Value()
                           : std::string("");
    std::string voice = (options.Has("voice") && options.Get("voice").IsString())
                            ? options.Get("voice").As<Napi::String>().Utf8Value()
                            : std::string("");
    std::string ref_text;
    if (options.Has("ref_text") && options.Get("ref_text").IsString()) {
        ref_text = options.Get("ref_text").As<Napi::String>().Utf8Value();
    } else if (options.Has("refText") && options.Get("refText").IsString()) {
        ref_text = options.Get("refText").As<Napi::String>().Utf8Value();
    }

    std::string instruct = (options.Has("instruct") && options.Get("instruct").IsString())
                               ? options.Get("instruct").As<Napi::String>().Utf8Value()
                               : std::string("");
    std::string language = (options.Has("language") && options.Get("language").IsString())
                               ? options.Get("language").As<Napi::String>().Utf8Value()
                               : std::string("zh");

    std::string output_path;
    if (options.Has("output_path") && options.Get("output_path").IsString()) {
        output_path = options.Get("output_path").As<Napi::String>().Utf8Value();
    } else if (options.Has("outputPath") && options.Get("outputPath").IsString()) {
        output_path = options.Get("outputPath").As<Napi::String>().Utf8Value();
    }

    int n_threads = (options.Has("n_threads") && options.Get("n_threads").IsNumber())
                        ? options.Get("n_threads").As<Napi::Number>().Int32Value()
                        : std::min(4, static_cast<int32_t>(std::thread::hardware_concurrency()));
    bool use_gpu = (options.Has("use_gpu") && options.Get("use_gpu").IsBoolean())
                       ? options.Get("use_gpu").As<Napi::Boolean>().Value()
                       : true;
    float temperature = (options.Has("temperature") && options.Get("temperature").IsNumber())
                            ? options.Get("temperature").As<Napi::Number>().FloatValue()
                            : 0.9f;
    uint64_t seed = (options.Has("seed") && options.Get("seed").IsNumber())
                        ? options.Get("seed").As<Napi::Number>().Int64Value()
                        : 42;

    bool reuse_instance = false;
    if (options.Has("reuse_instance") && options.Get("reuse_instance").IsBoolean()) {
        reuse_instance = options.Get("reuse_instance").As<Napi::Boolean>();
    }

    int64_t auto_release_ms = 0;
    if (options.Has("auto_release_ms") && options.Get("auto_release_ms").IsNumber()) {
        auto_release_ms = options.Get("auto_release_ms").As<Napi::Number>().Int64Value();
    }

    std::string backend_name = "";
    if (options.Has("backend") && options.Get("backend").IsString()) {
        backend_name = options.Get("backend").As<Napi::String>().Utf8Value();
        if (backend_name == "crispasr" || backend_name == "CrispASR") {
            backend_name = "";
        }
    }

    bool watermark = false;
    if (options.Has("watermark") && options.Get("watermark").IsBoolean()) {
        watermark = options.Get("watermark").As<Napi::Boolean>().Value();
    } else if (options.Has("enable_watermark") && options.Get("enable_watermark").IsBoolean()) {
        watermark = options.Get("enable_watermark").As<Napi::Boolean>().Value();
    } else if (options.Has("enableWatermark") && options.Get("enableWatermark").IsBoolean()) {
        watermark = options.Get("enableWatermark").As<Napi::Boolean>().Value();
    }

    float speed = 1.0f;
    if (options.Has("speed") && options.Get("speed").IsNumber()) {
        speed = options.Get("speed").As<Napi::Number>().FloatValue();
    } else if (options.Has("rate") && options.Get("rate").IsNumber()) {
        speed = options.Get("rate").As<Napi::Number>().FloatValue();
    }

    int steps = -1;
    if (options.Has("steps") && options.Get("steps").IsNumber()) {
        steps = options.Get("steps").As<Napi::Number>().Int32Value();
    } else if (options.Has("num_steps") && options.Get("num_steps").IsNumber()) {
        steps = options.Get("num_steps").As<Napi::Number>().Int32Value();
    } else if (options.Has("numSteps") && options.Get("numSteps").IsNumber()) {
        steps = options.Get("numSteps").As<Napi::Number>().Int32Value();
    } else if (options.Has("tts_steps") && options.Get("tts_steps").IsNumber()) {
        steps = options.Get("tts_steps").As<Napi::Number>().Int32Value();
    } else if (options.Has("ttsSteps") && options.Get("ttsSteps").IsNumber()) {
        steps = options.Get("ttsSteps").As<Napi::Number>().Int32Value();
    }

    float cfg_scale = -1.0f;
    if (options.Has("cfg_scale") && options.Get("cfg_scale").IsNumber()) {
        cfg_scale = options.Get("cfg_scale").As<Napi::Number>().FloatValue();
    } else if (options.Has("cfgScale") && options.Get("cfgScale").IsNumber()) {
        cfg_scale = options.Get("cfgScale").As<Napi::Number>().FloatValue();
    } else if (options.Has("cfg_weight") && options.Get("cfg_weight").IsNumber()) {
        cfg_scale = options.Get("cfg_weight").As<Napi::Number>().FloatValue();
    } else if (options.Has("cfgWeight") && options.Get("cfgWeight").IsNumber()) {
        cfg_scale = options.Get("cfgWeight").As<Napi::Number>().FloatValue();
    } else if (options.Has("tts_cfg_scale") && options.Get("tts_cfg_scale").IsNumber()) {
        cfg_scale = options.Get("tts_cfg_scale").As<Napi::Number>().FloatValue();
    }

    float noise_temp = -1.0f;
    if (options.Has("noise_temp") && options.Get("noise_temp").IsNumber()) {
        noise_temp = options.Get("noise_temp").As<Napi::Number>().FloatValue();
    } else if (options.Has("noiseTemp") && options.Get("noiseTemp").IsNumber()) {
        noise_temp = options.Get("noiseTemp").As<Napi::Number>().FloatValue();
    } else if (options.Has("noise_temperature") && options.Get("noise_temperature").IsNumber()) {
        noise_temp = options.Get("noise_temperature").As<Napi::Number>().FloatValue();
    } else if (options.Has("tts_noise_temp") && options.Get("tts_noise_temp").IsNumber()) {
        noise_temp = options.Get("tts_noise_temp").As<Napi::Number>().FloatValue();
    }

    int num_candidates = -1;
    if (options.Has("num_candidates") && options.Get("num_candidates").IsNumber()) {
        num_candidates = options.Get("num_candidates").As<Napi::Number>().Int32Value();
    } else if (options.Has("numCandidates") && options.Get("numCandidates").IsNumber()) {
        num_candidates = options.Get("numCandidates").As<Napi::Number>().Int32Value();
    } else if (options.Has("tts_num_candidates") && options.Get("tts_num_candidates").IsNumber()) {
        num_candidates = options.Get("tts_num_candidates").As<Napi::Number>().Int32Value();
    }

    int min_speech_tokens = -1;
    if (options.Has("min_speech_tokens") && options.Get("min_speech_tokens").IsNumber()) {
        min_speech_tokens = options.Get("min_speech_tokens").As<Napi::Number>().Int32Value();
    } else if (options.Has("minSpeechTokens") && options.Get("minSpeechTokens").IsNumber()) {
        min_speech_tokens = options.Get("minSpeechTokens").As<Napi::Number>().Int32Value();
    } else if (options.Has("tts_min_speech_tokens") && options.Get("tts_min_speech_tokens").IsNumber()) {
        min_speech_tokens = options.Get("tts_min_speech_tokens").As<Napi::Number>().Int32Value();
    }

    int max_speech_tokens = -1;
    if (options.Has("max_speech_tokens") && options.Get("max_speech_tokens").IsNumber()) {
        max_speech_tokens = options.Get("max_speech_tokens").As<Napi::Number>().Int32Value();
    } else if (options.Has("maxSpeechTokens") && options.Get("maxSpeechTokens").IsNumber()) {
        max_speech_tokens = options.Get("maxSpeechTokens").As<Napi::Number>().Int32Value();
    } else if (options.Has("tts_max_speech_tokens") && options.Get("tts_max_speech_tokens").IsNumber()) {
        max_speech_tokens = options.Get("tts_max_speech_tokens").As<Napi::Number>().Int32Value();
    }

    float exaggeration = -1.0f;
    if (options.Has("exaggeration") && options.Get("exaggeration").IsNumber()) {
        exaggeration = options.Get("exaggeration").As<Napi::Number>().FloatValue();
    } else if (options.Has("tts_exaggeration") && options.Get("tts_exaggeration").IsNumber()) {
        exaggeration = options.Get("tts_exaggeration").As<Napi::Number>().FloatValue();
    }

    std::string phonemes = "";
    if (options.Has("phonemes") && options.Get("phonemes").IsString()) {
        phonemes = options.Get("phonemes").As<Napi::String>().Utf8Value();
    } else if (options.Has("tts_phonemes") && options.Get("tts_phonemes").IsString()) {
        phonemes = options.Get("tts_phonemes").As<Napi::String>().Utf8Value();
    }

    int pad_silence_ms = 0;
    if (options.Has("pad_silence_ms") && options.Get("pad_silence_ms").IsNumber()) {
        pad_silence_ms = options.Get("pad_silence_ms").As<Napi::Number>().Int32Value();
    } else if (options.Has("padSilenceMs") && options.Get("padSilenceMs").IsNumber()) {
        pad_silence_ms = options.Get("padSilenceMs").As<Napi::Number>().Int32Value();
    } else if (options.Has("tts_pad_silence_ms") && options.Get("tts_pad_silence_ms").IsNumber()) {
        pad_silence_ms = options.Get("tts_pad_silence_ms").As<Napi::Number>().Int32Value();
    }

    std::string speaker_identity = "";
    if (options.Has("speaker_identity") && options.Get("speaker_identity").IsString()) {
        speaker_identity = options.Get("speaker_identity").As<Napi::String>().Utf8Value();
    } else if (options.Has("speakerIdentity") && options.Get("speakerIdentity").IsString()) {
        speaker_identity = options.Get("speakerIdentity").As<Napi::String>().Utf8Value();
    } else if (options.Has("tts_speaker_identity") && options.Get("tts_speaker_identity").IsString()) {
        speaker_identity = options.Get("tts_speaker_identity").As<Napi::String>().Utf8Value();
    }

    std::string g2p_dict = "";
    if (options.Has("g2p_dict") && options.Get("g2p_dict").IsString()) {
        g2p_dict = options.Get("g2p_dict").As<Napi::String>().Utf8Value();
    } else if (options.Has("g2pDict") && options.Get("g2pDict").IsString()) {
        g2p_dict = options.Get("g2pDict").As<Napi::String>().Utf8Value();
    }

    std::string ref_lang = "";
    if (options.Has("ref_language") && options.Get("ref_language").IsString()) {
        ref_lang = options.Get("ref_language").As<Napi::String>().Utf8Value();
    } else if (options.Has("refLanguage") && options.Get("refLanguage").IsString()) {
        ref_lang = options.Get("refLanguage").As<Napi::String>().Utf8Value();
    }

    CrispasrTTSWorker* worker = new CrispasrTTSWorker(
        callback, model_path, codec_model_path, text, voice, ref_text, instruct, language,
        output_path, n_threads, use_gpu, temperature, seed, reuse_instance, auto_release_ms, env,
        backend_name, watermark, speed, steps, cfg_scale, noise_temp, num_candidates,
        min_speech_tokens, max_speech_tokens, exaggeration, phonemes, pad_silence_ms,
        speaker_identity, g2p_dict, ref_lang);
    worker->Queue();
    return env.Undefined();
}

// ============================================================================
// Standalone Audio Watermarking Interface (for Sherpa-ONNX, Kokoro, etc.)
// ============================================================================

struct WavAudioData {
    std::vector<float> samples;
    int sample_rate = 24000;
    int channels = 1;
    bool success = false;
    std::string error;
};

static WavAudioData parse_wav_memory(const uint8_t* data, size_t size) {
    WavAudioData result;
    if (!data || size < 44) {
        result.error = "Data too small for WAV header";
        return result;
    }
    if (std::memcmp(data, "RIFF", 4) != 0 || std::memcmp(data + 8, "WAVE", 4) != 0) {
        result.error = "Not a valid RIFF/WAVE header";
        return result;
    }

    size_t pos = 12;
    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    const uint8_t* pcm_data = nullptr;
    uint32_t pcm_data_size = 0;

    while (pos + 8 <= size) {
        char chunk_id[5] = {0};
        std::memcpy(chunk_id, data + pos, 4);
        uint32_t chunk_size = *reinterpret_cast<const uint32_t*>(data + pos + 4);
        pos += 8;

        if (std::strcmp(chunk_id, "fmt ") == 0) {
            if (chunk_size < 16 || pos + 16 > size) {
                result.error = "Corrupt fmt chunk";
                return result;
            }
            audio_format = *reinterpret_cast<const uint16_t*>(data + pos);
            num_channels = *reinterpret_cast<const uint16_t*>(data + pos + 2);
            sample_rate = *reinterpret_cast<const uint32_t*>(data + pos + 4);
            bits_per_sample = *reinterpret_cast<const uint16_t*>(data + pos + 14);
            pos += chunk_size;
        } else if (std::strcmp(chunk_id, "data") == 0) {
            pcm_data = data + pos;
            pcm_data_size = std::min(chunk_size, static_cast<uint32_t>(size - pos));
            pos += chunk_size;
            break;
        } else {
            pos += chunk_size;
        }
        if (chunk_size % 2 != 0) {
            pos++;
        }
    }

    if (!pcm_data || pcm_data_size == 0 || num_channels == 0 || sample_rate == 0) {
        result.error = "Missing fmt or data chunk in WAV";
        return result;
    }

    result.sample_rate = static_cast<int>(sample_rate);
    result.channels = static_cast<int>(num_channels);

    if (audio_format == 1 && bits_per_sample == 16) {
        size_t total_samples = pcm_data_size / (sizeof(int16_t) * num_channels);
        result.samples.resize(total_samples);
        const int16_t* src = reinterpret_cast<const int16_t*>(pcm_data);
        for (size_t i = 0; i < total_samples; ++i) {
            float sum = 0.0f;
            for (int c = 0; c < num_channels; ++c) {
                sum += static_cast<float>(src[i * num_channels + c]) / 32768.0f;
            }
            result.samples[i] = sum / num_channels;
        }
        result.success = true;
    } else if (audio_format == 3 && bits_per_sample == 32) {
        size_t total_samples = pcm_data_size / (sizeof(float) * num_channels);
        result.samples.resize(total_samples);
        const float* src = reinterpret_cast<const float*>(pcm_data);
        for (size_t i = 0; i < total_samples; ++i) {
            float sum = 0.0f;
            for (int c = 0; c < num_channels; ++c) {
                sum += src[i * num_channels + c];
            }
            result.samples[i] = sum / num_channels;
        }
        result.success = true;
    } else if (audio_format == 1 && bits_per_sample == 24) {
        size_t total_samples = pcm_data_size / (3 * num_channels);
        result.samples.resize(total_samples);
        for (size_t i = 0; i < total_samples; ++i) {
            float sum = 0.0f;
            for (int c = 0; c < num_channels; ++c) {
                const uint8_t* b = pcm_data + (i * num_channels + c) * 3;
                int32_t val = (b[0]) | (b[1] << 8) | (b[2] << 16);
                if (val & 0x800000) val |= ~0xFFFFFF;
                sum += static_cast<float>(val) / 8388608.0f;
            }
            result.samples[i] = sum / num_channels;
        }
        result.success = true;
    } else {
        result.error = "Unsupported WAV format: format=" + std::to_string(audio_format) + ", bits=" + std::to_string(bits_per_sample);
        return result;
    }
    return result;
}

static WavAudioData read_wav_file(const std::string& path) {
    WavAudioData result;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        result.error = "Failed to open file: " + path;
        return result;
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        result.error = "Failed to read file: " + path;
        return result;
    }
    return parse_wav_memory(buffer.data(), buffer.size());
}

static std::vector<uint8_t> encode_wav_memory(const float* pcm, size_t n_samples, int sample_rate) {
    uint32_t sr = static_cast<uint32_t>(sample_rate > 0 ? sample_rate : 24000);
    uint16_t num_channels = 1;
    uint16_t bits_per_sample = 16;
    uint32_t byte_rate = sr * num_channels * (bits_per_sample / 8);
    uint16_t block_align = num_channels * (bits_per_sample / 8);
    uint32_t data_size = static_cast<uint32_t>(n_samples * sizeof(int16_t));
    uint32_t chunk_size = 36 + data_size;

    std::vector<uint8_t> out(44 + data_size);
    uint8_t* p = out.data();

    std::memcpy(p, "RIFF", 4);
    std::memcpy(p + 4, &chunk_size, 4);
    std::memcpy(p + 8, "WAVE", 4);
    std::memcpy(p + 12, "fmt ", 4);
    uint32_t subchunk1_size = 16;
    uint16_t audio_format = 1;
    std::memcpy(p + 16, &subchunk1_size, 4);
    std::memcpy(p + 20, &audio_format, 2);
    std::memcpy(p + 22, &num_channels, 2);
    std::memcpy(p + 24, &sr, 4);
    std::memcpy(p + 28, &byte_rate, 4);
    std::memcpy(p + 32, &block_align, 2);
    std::memcpy(p + 34, &bits_per_sample, 2);
    std::memcpy(p + 36, "data", 4);
    std::memcpy(p + 40, &data_size, 4);

    int16_t* dst = reinterpret_cast<int16_t*>(p + 44);
    for (size_t i = 0; i < n_samples; ++i) {
        float s = std::max(-1.0f, std::min(1.0f, pcm[i]));
        dst[i] = (s < 0) ? static_cast<int16_t>(s * 32768.0f) : static_cast<int16_t>(s * 32767.0f);
    }
    return out;
}

static bool write_wav_file(const std::string& path, const float* pcm, size_t n_samples, int sample_rate) {
    auto bytes = encode_wav_memory(pcm, n_samples, sample_rate);
    std::ofstream fout(path, std::ios::binary);
    if (!fout.is_open()) return false;
    fout.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    return fout.good();
}

class WatermarkWorker : public Napi::AsyncWorker {
public:
    WatermarkWorker(Napi::Function& callback,
                    std::string input_path,
                    std::string output_path,
                    std::vector<float> pcm_input,
                    std::vector<uint8_t> wav_buffer_input,
                    int sample_rate,
                    float alpha,
                    std::string watermark_model)
        : Napi::AsyncWorker(callback),
          m_input_path(input_path),
          m_output_path(output_path),
          m_pcm(std::move(pcm_input)),
          m_wav_buffer(std::move(wav_buffer_input)),
          m_sample_rate(sample_rate),
          m_alpha(alpha),
          m_watermark_model(watermark_model) {}

    WatermarkWorker(Napi::Env env,
                    std::string input_path,
                    std::string output_path,
                    std::vector<float> pcm_input,
                    std::vector<uint8_t> wav_buffer_input,
                    int sample_rate,
                    float alpha,
                    std::string watermark_model)
        : Napi::AsyncWorker(env),
          m_deferred(new Napi::Promise::Deferred(env)),
          m_input_path(input_path),
          m_output_path(output_path),
          m_pcm(std::move(pcm_input)),
          m_wav_buffer(std::move(wav_buffer_input)),
          m_sample_rate(sample_rate),
          m_alpha(alpha),
          m_watermark_model(watermark_model) {}

    Napi::Promise GetPromise() {
        return m_deferred->Promise();
    }

    void Execute() override {
        if (m_pcm.empty()) {
            if (!m_wav_buffer.empty()) {
                auto parsed = parse_wav_memory(m_wav_buffer.data(), m_wav_buffer.size());
                if (parsed.success) {
                    m_pcm = std::move(parsed.samples);
                    m_sample_rate = parsed.sample_rate;
                } else {
                    SetError("Failed to parse WAV buffer: " + parsed.error);
                    return;
                }
            } else if (!m_input_path.empty()) {
                auto loaded = read_wav_file(m_input_path);
                if (loaded.success) {
                    m_pcm = std::move(loaded.samples);
                    m_sample_rate = loaded.sample_rate;
                } else {
                    float* pcm = nullptr;
                    int n_samples = 0;
                    int sr = 0;
                    if (crispasr_audio_load(m_input_path.c_str(), &pcm, &n_samples, &sr) == 0 && pcm && n_samples > 0) {
                        m_pcm.assign(pcm, pcm + n_samples);
                        m_sample_rate = (sr > 0) ? sr : 16000;
                        crispasr_audio_free(pcm);
                    } else {
                        SetError("Failed to load audio from: " + m_input_path + " (" + loaded.error + ")");
                        return;
                    }
                }
            }
        }

        if (m_pcm.empty()) {
            SetError("No audio samples provided or audio is empty");
            return;
        }

        if (!m_watermark_model.empty()) {
            crispasr_watermark_load_model(m_watermark_model.c_str());
        }

        crispasr_watermark_embed(m_pcm.data(), static_cast<int>(m_pcm.size()), m_alpha);

        std::string target_path = m_output_path;
        if (target_path.empty() && !m_input_path.empty()) {
            target_path = m_input_path;
        }

        if (!target_path.empty()) {
            if (!write_wav_file(target_path, m_pcm.data(), m_pcm.size(), m_sample_rate)) {
                SetError("Failed to write watermarked audio to: " + target_path);
                return;
            }
            m_final_path = target_path;
        } else {
            m_out_wav_bytes = encode_wav_memory(m_pcm.data(), m_pcm.size(), m_sample_rate);
        }
    }

    void OnOK() override {
        Napi::HandleScope scope(Env());
        Napi::Object res = Napi::Object::New(Env());
        res.Set("success", Napi::Boolean::New(Env(), true));
        res.Set("sampleRate", Napi::Number::New(Env(), m_sample_rate));
        res.Set("n_samples", Napi::Number::New(Env(), m_pcm.size()));
        res.Set("duration", Napi::Number::New(Env(), static_cast<double>(m_pcm.size()) / static_cast<double>(m_sample_rate > 0 ? m_sample_rate : 24000)));
        res.Set("watermarked", Napi::Boolean::New(Env(), true));

        if (!m_final_path.empty()) {
            res.Set("path", Napi::String::New(Env(), m_final_path));
        }
        if (!m_out_wav_bytes.empty()) {
            res.Set("buffer", Napi::Buffer<uint8_t>::Copy(Env(), m_out_wav_bytes.data(), m_out_wav_bytes.size()));
        }

        if (m_deferred) {
            m_deferred->Resolve(res);
        } else {
            Callback().Call({Env().Null(), res});
        }
    }

    void OnError(const Napi::Error& e) override {
        Napi::HandleScope scope(Env());
        if (m_deferred) {
            m_deferred->Reject(e.Value());
        } else {
            Callback().Call({e.Value(), Env().Undefined()});
        }
    }

private:
    std::string m_input_path;
    std::string m_output_path;
    std::string m_final_path;
    std::vector<float> m_pcm;
    std::vector<uint8_t> m_wav_buffer;
    std::vector<uint8_t> m_out_wav_bytes;
    int m_sample_rate = 24000;
    float m_alpha = -1.0f;
    std::string m_watermark_model;
    std::unique_ptr<Napi::Promise::Deferred> m_deferred;
};

class WatermarkDetectWorker : public Napi::AsyncWorker {
public:
    WatermarkDetectWorker(Napi::Function& callback,
                          std::string input_path,
                          std::vector<float> pcm_input,
                          std::vector<uint8_t> wav_buffer_input,
                          int sample_rate,
                          std::string watermark_model)
        : Napi::AsyncWorker(callback),
          m_input_path(input_path),
          m_pcm(std::move(pcm_input)),
          m_wav_buffer(std::move(wav_buffer_input)),
          m_sample_rate(sample_rate),
          m_watermark_model(watermark_model) {}

    WatermarkDetectWorker(Napi::Env env,
                          std::string input_path,
                          std::vector<float> pcm_input,
                          std::vector<uint8_t> wav_buffer_input,
                          int sample_rate,
                          std::string watermark_model)
        : Napi::AsyncWorker(env),
          m_deferred(new Napi::Promise::Deferred(env)),
          m_input_path(input_path),
          m_pcm(std::move(pcm_input)),
          m_wav_buffer(std::move(wav_buffer_input)),
          m_sample_rate(sample_rate),
          m_watermark_model(watermark_model) {}

    Napi::Promise GetPromise() {
        return m_deferred->Promise();
    }

    void Execute() override {
        if (m_pcm.empty()) {
            if (!m_wav_buffer.empty()) {
                auto parsed = parse_wav_memory(m_wav_buffer.data(), m_wav_buffer.size());
                if (parsed.success) {
                    m_pcm = std::move(parsed.samples);
                    m_sample_rate = parsed.sample_rate;
                } else {
                    SetError("Failed to parse WAV buffer: " + parsed.error);
                    return;
                }
            } else if (!m_input_path.empty()) {
                auto loaded = read_wav_file(m_input_path);
                if (loaded.success) {
                    m_pcm = std::move(loaded.samples);
                    m_sample_rate = loaded.sample_rate;
                } else {
                    float* pcm = nullptr;
                    int n_samples = 0;
                    int sr = 0;
                    if (crispasr_audio_load(m_input_path.c_str(), &pcm, &n_samples, &sr) == 0 && pcm && n_samples > 0) {
                        m_pcm.assign(pcm, pcm + n_samples);
                        m_sample_rate = (sr > 0) ? sr : 16000;
                        crispasr_audio_free(pcm);
                    } else {
                        SetError("Failed to load audio from: " + m_input_path + " (" + loaded.error + ")");
                        return;
                    }
                }
            }
        }

        if (m_pcm.empty()) {
            SetError("No audio samples provided or audio is empty");
            return;
        }

        if (!m_watermark_model.empty()) {
            crispasr_watermark_load_model(m_watermark_model.c_str());
        }

        m_score = crispasr_watermark_detect(m_pcm.data(), static_cast<int>(m_pcm.size()));
    }

    void OnOK() override {
        Napi::HandleScope scope(Env());
        Napi::Object res = Napi::Object::New(Env());
        res.Set("success", Napi::Boolean::New(Env(), true));
        res.Set("score", Napi::Number::New(Env(), m_score));
        res.Set("detected", Napi::Boolean::New(Env(), m_score >= 0.65f));
        res.Set("threshold", Napi::Number::New(Env(), 0.65f));
        res.Set("sampleRate", Napi::Number::New(Env(), m_sample_rate));
        res.Set("n_samples", Napi::Number::New(Env(), m_pcm.size()));
        res.Set("duration", Napi::Number::New(Env(), static_cast<double>(m_pcm.size()) / static_cast<double>(m_sample_rate > 0 ? m_sample_rate : 24000)));

        if (m_deferred) {
            m_deferred->Resolve(res);
        } else {
            Callback().Call({Env().Null(), res});
        }
    }

    void OnError(const Napi::Error& e) override {
        Napi::HandleScope scope(Env());
        if (m_deferred) {
            m_deferred->Reject(e.Value());
        } else {
            Callback().Call({e.Value(), Env().Undefined()});
        }
    }

private:
    std::string m_input_path;
    std::vector<float> m_pcm;
    std::vector<uint8_t> m_wav_buffer;
    int m_sample_rate = 24000;
    std::string m_watermark_model;
    float m_score = 0.0f;
    std::unique_ptr<Napi::Promise::Deferred> m_deferred;
};

Napi::Value applyWatermark(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Usage: applyWatermark(options, [callback])").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::string input_path = "";
    std::string output_path = "";
    std::vector<float> pcm_input;
    std::vector<uint8_t> wav_buffer_input;
    int sample_rate = 24000;
    float alpha = -1.0f;
    std::string watermark_model = "";

    Napi::Function callback;
    bool has_callback = false;

    if (info.Length() >= 2 && info[1].IsFunction()) {
        callback = info[1].As<Napi::Function>();
        has_callback = true;
    } else if (info[0].IsFunction()) {
        callback = info[0].As<Napi::Function>();
        has_callback = true;
    }

    if (info[0].IsString()) {
        input_path = info[0].As<Napi::String>().Utf8Value();
    } else if (info[0].IsObject()) {
        Napi::Object opts = info[0].As<Napi::Object>();
        if (opts.Has("path") && opts.Get("path").IsString()) {
            input_path = opts.Get("path").As<Napi::String>().Utf8Value();
        } else if (opts.Has("input_path") && opts.Get("input_path").IsString()) {
            input_path = opts.Get("input_path").As<Napi::String>().Utf8Value();
        } else if (opts.Has("filePath") && opts.Get("filePath").IsString()) {
            input_path = opts.Get("filePath").As<Napi::String>().Utf8Value();
        } else if (opts.Has("file") && opts.Get("file").IsString()) {
            input_path = opts.Get("file").As<Napi::String>().Utf8Value();
        }

        if (opts.Has("output_path") && opts.Get("output_path").IsString()) {
            output_path = opts.Get("output_path").As<Napi::String>().Utf8Value();
        } else if (opts.Has("outputPath") && opts.Get("outputPath").IsString()) {
            output_path = opts.Get("outputPath").As<Napi::String>().Utf8Value();
        }

        if (opts.Has("sample_rate") && opts.Get("sample_rate").IsNumber()) {
            sample_rate = opts.Get("sample_rate").As<Napi::Number>().Int32Value();
        } else if (opts.Has("sampleRate") && opts.Get("sampleRate").IsNumber()) {
            sample_rate = opts.Get("sampleRate").As<Napi::Number>().Int32Value();
        }

        if (opts.Has("alpha") && opts.Get("alpha").IsNumber()) {
            alpha = opts.Get("alpha").As<Napi::Number>().FloatValue();
        }

        if (opts.Has("watermark_model") && opts.Get("watermark_model").IsString()) {
            watermark_model = opts.Get("watermark_model").As<Napi::String>().Utf8Value();
        } else if (opts.Has("watermarkModel") && opts.Get("watermarkModel").IsString()) {
            watermark_model = opts.Get("watermarkModel").As<Napi::String>().Utf8Value();
        }

        if (opts.Has("buffer")) {
            Napi::Value bufVal = opts.Get("buffer");
            if (bufVal.IsBuffer()) {
                Napi::Buffer<uint8_t> buf = bufVal.As<Napi::Buffer<uint8_t>>();
                wav_buffer_input.assign(buf.Data(), buf.Data() + buf.Length());
            } else if (bufVal.IsTypedArray()) {
                Napi::TypedArray typedArray = bufVal.As<Napi::TypedArray>();
                if (typedArray.TypedArrayType() == napi_float32_array) {
                    Napi::Float32Array fArray = bufVal.As<Napi::Float32Array>();
                    pcm_input.assign(fArray.Data(), fArray.Data() + fArray.ElementLength());
                }
            }
        }
    }

    if (has_callback) {
        WatermarkWorker* worker = new WatermarkWorker(callback, input_path, output_path, std::move(pcm_input), std::move(wav_buffer_input), sample_rate, alpha, watermark_model);
        worker->Queue();
        return env.Undefined();
    } else {
        WatermarkWorker* worker = new WatermarkWorker(env, input_path, output_path, std::move(pcm_input), std::move(wav_buffer_input), sample_rate, alpha, watermark_model);
        auto promise = worker->GetPromise();
        worker->Queue();
        return promise;
    }
}

Napi::Value detectWatermark(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Usage: detectWatermark(options, [callback])").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::string input_path = "";
    std::vector<float> pcm_input;
    std::vector<uint8_t> wav_buffer_input;
    int sample_rate = 24000;
    std::string watermark_model = "";

    Napi::Function callback;
    bool has_callback = false;

    if (info.Length() >= 2 && info[1].IsFunction()) {
        callback = info[1].As<Napi::Function>();
        has_callback = true;
    } else if (info[0].IsFunction()) {
        callback = info[0].As<Napi::Function>();
        has_callback = true;
    }

    if (info[0].IsString()) {
        input_path = info[0].As<Napi::String>().Utf8Value();
    } else if (info[0].IsObject()) {
        Napi::Object opts = info[0].As<Napi::Object>();
        if (opts.Has("path") && opts.Get("path").IsString()) {
            input_path = opts.Get("path").As<Napi::String>().Utf8Value();
        } else if (opts.Has("input_path") && opts.Get("input_path").IsString()) {
            input_path = opts.Get("input_path").As<Napi::String>().Utf8Value();
        } else if (opts.Has("filePath") && opts.Get("filePath").IsString()) {
            input_path = opts.Get("filePath").As<Napi::String>().Utf8Value();
        } else if (opts.Has("file") && opts.Get("file").IsString()) {
            input_path = opts.Get("file").As<Napi::String>().Utf8Value();
        }

        if (opts.Has("sample_rate") && opts.Get("sample_rate").IsNumber()) {
            sample_rate = opts.Get("sample_rate").As<Napi::Number>().Int32Value();
        } else if (opts.Has("sampleRate") && opts.Get("sampleRate").IsNumber()) {
            sample_rate = opts.Get("sampleRate").As<Napi::Number>().Int32Value();
        }

        if (opts.Has("watermark_model") && opts.Get("watermark_model").IsString()) {
            watermark_model = opts.Get("watermark_model").As<Napi::String>().Utf8Value();
        } else if (opts.Has("watermarkModel") && opts.Get("watermarkModel").IsString()) {
            watermark_model = opts.Get("watermarkModel").As<Napi::String>().Utf8Value();
        }

        if (opts.Has("buffer")) {
            Napi::Value bufVal = opts.Get("buffer");
            if (bufVal.IsBuffer()) {
                Napi::Buffer<uint8_t> buf = bufVal.As<Napi::Buffer<uint8_t>>();
                wav_buffer_input.assign(buf.Data(), buf.Data() + buf.Length());
            } else if (bufVal.IsTypedArray()) {
                Napi::TypedArray typedArray = bufVal.As<Napi::TypedArray>();
                if (typedArray.TypedArrayType() == napi_float32_array) {
                    Napi::Float32Array fArray = bufVal.As<Napi::Float32Array>();
                    pcm_input.assign(fArray.Data(), fArray.Data() + fArray.ElementLength());
                }
            }
        }
    }

    if (has_callback) {
        WatermarkDetectWorker* worker = new WatermarkDetectWorker(callback, input_path, std::move(pcm_input), std::move(wav_buffer_input), sample_rate, watermark_model);
        worker->Queue();
        return env.Undefined();
    } else {
        WatermarkDetectWorker* worker = new WatermarkDetectWorker(env, input_path, std::move(pcm_input), std::move(wav_buffer_input), sample_rate, watermark_model);
        auto promise = worker->GetPromise();
        worker->Queue();
        return promise;
    }
}

Napi::Value loadWatermarkModel(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Usage: loadWatermarkModel(modelPath)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    std::string path = info[0].As<Napi::String>().Utf8Value();
    int rc = crispasr_watermark_load_model(path.c_str());
    return Napi::Boolean::New(env, rc == 0);
}

void InitCrispASR(Napi::Env env, Napi::Object exports) {
    printf("InitCrispASR: Initializing CrispASR exports...\n");
    exports.Set("parakeetASR", Napi::Function::New(env, parakeetASR));
    exports.Set("crispasrASR", Napi::Function::New(env, crispasrASR));
    exports.Set("distilWhisper", Napi::Function::New(env, distilWhisper));
    exports.Set("crispasrTTS", Napi::Function::New(env, crispasrTTS));
    exports.Set("qwen3TTS", Napi::Function::New(env, crispasrTTS));
    exports.Set("tts", Napi::Function::New(env, crispasrTTS));

    // Standalone Watermark APIs
    exports.Set("applyWatermark", Napi::Function::New(env, applyWatermark));
    exports.Set("watermarkAudio", Napi::Function::New(env, applyWatermark));
    exports.Set("embedWatermark", Napi::Function::New(env, applyWatermark));
    exports.Set("detectWatermark", Napi::Function::New(env, detectWatermark));
    exports.Set("loadWatermarkModel", Napi::Function::New(env, loadWatermarkModel));

    CrispASRStream::Init(env, exports);
}