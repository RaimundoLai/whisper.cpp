// crispasr-addon.cpp - CrispASR (Parakeet, Distil-Whisper) N-API bindings
// This file is included by addon.cpp



#include "../../third-party/CrispASR/src/parakeet.h"
#include "whisper.h"

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
                   Napi::Env env)
        : Napi::AsyncWorker(callback),
          m_model_path(std::move(model_path)),
          m_pcmf32(std::move(pcmf32)),
          m_n_threads(n_threads),
          m_use_gpu(use_gpu),
          m_debug(debug) {}

    void Execute() override {
        struct parakeet_context_params params = parakeet_context_default_params();
        params.n_threads = m_n_threads;
        params.use_gpu = m_use_gpu;
        params.verbosity = m_debug ? 1 : 0;

        struct parakeet_context* ctx = parakeet_init_from_file(m_model_path.c_str(), params);
        if (!ctx) {
            SetError("Failed to initialize Parakeet context from " + m_model_path);
            return;
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

        parakeet_free(ctx);
    }

    void OnOK() override {
        Napi::HandleScope scope(Env());
        Napi::Object result = Napi::Object::New(Env());

        result.Set("text", Napi::String::New(Env(), m_result.text));
        
        Napi::Array tokens = Napi::Array::New(Env(), m_result.tokens.size());
        for (size_t i = 0; i < m_result.tokens.size(); i++) {
            Napi::Object tok = Napi::Object::New(Env());
            tok.Set("text", Napi::String::New(Env(), m_result.tokens[i].text));
            tok.Set("start", Napi::Number::New(Env(), m_result.tokens[i].start_ms));
            tok.Set("end", Napi::Number::New(Env(), m_result.tokens[i].end_ms));
            tok.Set("p", Napi::Number::New(Env(), m_result.tokens[i].p));
            tokens[i] = tok;
        }
        result.Set("tokens", tokens);

        Napi::Array words = Napi::Array::New(Env(), m_result.words.size());
        for (size_t i = 0; i < m_result.words.size(); i++) {
            Napi::Object word = Napi::Object::New(Env());
            word.Set("text", Napi::String::New(Env(), m_result.words[i].text));
            word.Set("start", Napi::Number::New(Env(), m_result.words[i].start_ms));
            word.Set("end", Napi::Number::New(Env(), m_result.words[i].end_ms));
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

    Napi::Function callback = info[1].As<Napi::Function>();
    ParakeetWorker* worker = new ParakeetWorker(callback, model_path, std::move(pcmf32), n_threads, use_gpu, debug, env);
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
    crispasr_session* crispasr_session_open_with_params(const char* model_path, const char* backend_name, const crispasr_open_params_v1* params);
    const char* crispasr_session_backend(crispasr_session* s);
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
    void crispasr_session_close(crispasr_session* s);
}

struct crispasr_addon_result {
    std::string text;
    
    struct word_data {
        std::string text;
        int64_t start_ms;
        int64_t end_ms;
        float p;
    };
    std::vector<word_data> words;

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
                   int n_threads,
                   bool use_gpu,
                   bool debug,
                   std::string language,
                   std::string vad_model_path,
                   float vad_threshold,
                   int min_speech_ms,
                   int min_silence_ms,
                   int speech_pad_ms,
                   int max_speech_ms,
                   Napi::Function progress_callback,
                   Napi::Env env)
        : Napi::AsyncWorker(callback),
          m_model_path(std::move(model_path)),
          m_backend_name(std::move(backend_name)),
          m_pcmf32(std::move(pcmf32)),
          m_n_threads(n_threads),
          m_use_gpu(use_gpu),
          m_debug(debug),
          m_language(std::move(language)),
          m_vad_model_path(std::move(vad_model_path)),
          m_vad_threshold(vad_threshold),
          m_min_speech_ms(min_speech_ms),
          m_min_silence_ms(min_silence_ms),
          m_speech_pad_ms(speech_pad_ms),
          m_max_speech_ms(max_speech_ms),
          env(env),
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

    void process_result(crispasr_session_result* res, int64_t offset_ms) {
        int n_segs = crispasr_session_result_n_segments(res);
        std::string full_text = m_result.text;

        for (int i = 0; i < n_segs; i++) {
            crispasr_addon_result::segment_data sd;
            const char* seg_text = crispasr_session_result_segment_text(res, i);
            sd.text = seg_text ? seg_text : "";
            sd.start_ms = offset_ms + crispasr_session_result_segment_t0(res, i) * 10;
            sd.end_ms = offset_ms + crispasr_session_result_segment_t1(res, i) * 10;

            if (!full_text.empty() && !sd.text.empty()) {
                full_text += " ";
            }
            full_text += sd.text;

            int n_words = crispasr_session_result_n_words(res, i);
            for (int j = 0; j < n_words; j++) {
                crispasr_addon_result::word_data wd;
                const char* w_text = crispasr_session_result_word_text(res, i, j);
                wd.text = w_text ? w_text : "";
                wd.start_ms = offset_ms + crispasr_session_result_word_t0(res, i, j) * 10;
                wd.end_ms = offset_ms + crispasr_session_result_word_t1(res, i, j) * 10;
                wd.p = crispasr_session_result_word_p(res, i, j);

                sd.words.push_back(wd);
                m_result.words.push_back(wd);
            }

            m_result.segments.push_back(std::move(sd));
        }

        m_result.text = full_text;
    }

    void Execute() override {
        crispasr_open_params_v1 params;
        memset(&params, 0, sizeof(params));
        params.abi_version = 2;
        params.n_threads = m_n_threads;
        params.use_gpu = m_use_gpu ? 1 : 0;
        params.verbosity = m_debug ? 1 : 0;
        params.flash_attn = 1;
        params.n_gpu_layers = -1;

        const char* backend_ptr = m_backend_name.empty() ? nullptr : m_backend_name.c_str();

        crispasr_session* session = crispasr_session_open_with_params(m_model_path.c_str(), backend_ptr, &params);
        if (!session) {
            SetError("Failed to open CrispASR session for model " + m_model_path);
            return;
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
                process_result(res, 0);
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
                crispasr_session_close(session);
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
                            process_result(res, static_cast<int64_t>(chunk_t0 * 1000));
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

        crispasr_session_close(session);
    }

    void OnOK() override {
        Napi::HandleScope scope(Env());
        Napi::Object result = Napi::Object::New(Env());

        result.Set("text", Napi::String::New(Env(), m_result.text));
        result.Set("backend", Napi::String::New(Env(), m_result.backend));
        result.Set("aborted", Napi::Boolean::New(Env(), m_was_aborted->load()));

        // Set flat words
        Napi::Array words = Napi::Array::New(Env(), m_result.words.size());
        for (size_t i = 0; i < m_result.words.size(); i++) {
            Napi::Object w_obj = Napi::Object::New(Env());
            w_obj.Set("text", Napi::String::New(Env(), m_result.words[i].text));
            w_obj.Set("start", Napi::Number::New(Env(), m_result.words[i].start_ms));
            w_obj.Set("end", Napi::Number::New(Env(), m_result.words[i].end_ms));
            w_obj.Set("p", Napi::Number::New(Env(), m_result.words[i].p));
            words[i] = w_obj;
        }
        result.Set("words", words);

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
                w_obj.Set("text", Napi::String::New(Env(), m_result.segments[i].words[j].text));
                w_obj.Set("start", Napi::Number::New(Env(), m_result.segments[i].words[j].start_ms));
                w_obj.Set("end", Napi::Number::New(Env(), m_result.segments[i].words[j].end_ms));
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
    int m_n_threads;
    bool m_use_gpu;
    bool m_debug;
    std::string m_language;
    std::string m_vad_model_path;
    float m_vad_threshold;
    int m_min_speech_ms;
    int m_min_silence_ms;
    int m_speech_pad_ms;
    int m_max_speech_ms;
    Napi::Env env;
    Napi::ThreadSafeFunction tsfn;
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
    }

    std::vector<float> pcmf32;
    if (options.Has("pcmf32") && options.Get("pcmf32").IsTypedArray()) {
        Napi::Float32Array pcmf32_arr = options.Get("pcmf32").As<Napi::Float32Array>();
        pcmf32.assign(pcmf32_arr.Data(), pcmf32_arr.Data() + pcmf32_arr.ElementLength());
    } else if (options.Has("file") && options.Get("file").IsString()) {
        std::string audio_file = options.Get("file").As<Napi::String>();
        std::vector<std::vector<float>> pcmf32s;
        if (!read_audio_data(audio_file, pcmf32, pcmf32s, false)) {
            Napi::Error::New(env, "failed to read audio file: " + audio_file).ThrowAsJavaScriptException();
            return env.Undefined();
        }
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

    std::string language = "";
    if (options.Has("language") && options.Get("language").IsString()) {
        language = options.Get("language").As<Napi::String>();
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

    Napi::Function progress_callback;
    if (options.Has("progress_callback") && options.Get("progress_callback").IsFunction()) {
        progress_callback = options.Get("progress_callback").As<Napi::Function>();
    }

    Napi::Function callback = info[1].As<Napi::Function>();
    CrispASRWorker* worker = new CrispASRWorker(
        callback, model_path, backend_name, std::move(pcmf32), n_threads, use_gpu, debug, language,
        vad_model_path, vad_threshold, min_speech_ms, min_silence_ms, speech_pad_ms, max_speech_ms,
        progress_callback, env
    );
    worker->Queue();

    return env.Undefined();
}

class CrispASRStream : public Napi::ObjectWrap<CrispASRStream> {
public:
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
        if (options.Has("backend")) m_backend_name = options.Get("backend").As<Napi::String>();
        if (options.Has("language")) m_language = options.Get("language").As<Napi::String>();
        if (options.Has("n_threads")) m_n_threads = options.Get("n_threads").As<Napi::Number>().Int32Value();
        if (options.Has("use_gpu")) m_use_gpu = options.Get("use_gpu").As<Napi::Boolean>();
        if (options.Has("chunk_size_ms")) m_chunk_size_ms = options.Get("chunk_size_ms").As<Napi::Number>().Int32Value();
        
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
    }

    Napi::Value Start(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        Napi::Function callback = info[0].As<Napi::Function>();
        m_tsfn = Napi::ThreadSafeFunction::New(env, callback, "CrispASRStream", 0, 1);
        
        crispasr_open_params_v1 params;
        memset(&params, 0, sizeof(params));
        params.abi_version = 2;
        params.n_threads = m_n_threads;
        params.use_gpu = m_use_gpu ? 1 : 0;
        params.verbosity = 0;
        params.flash_attn = 1;
        params.n_gpu_layers = -1;

        const char* backend_ptr = m_backend_name.empty() ? nullptr : m_backend_name.c_str();
        m_session = crispasr_session_open_with_params(m_model_path.c_str(), backend_ptr, &params);
        if (!m_session) {
            Napi::Error::New(env, "Failed to open CrispASR session").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        
        m_segment_index = 0;
        if (!m_vad_model_path.empty()) {
            whisper_vad_context_params vad_ctx_params = whisper_vad_default_context_params();
            vad_ctx_params.n_threads = m_n_threads;
            vad_ctx_params.use_gpu = false;
            m_vctx = whisper_vad_init_from_file_with_params(m_vad_model_path.c_str(), vad_ctx_params);
        }
        
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
        m_audio_buffer.clear();
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
    void emit_segment(crispasr_session_result* res, int64_t start_time, int64_t end_time, int segment_index, const std::string& type) {
        int n_segs = crispasr_session_result_n_segments(res);
        std::string text = "";
        
        struct word_item {
            std::string word;
            int64_t start;
            int64_t end;
            float p;
        };
        std::vector<word_item> words_list;
        
        for (int i = 0; i < n_segs; i++) {
            const char* seg_text = crispasr_session_result_segment_text(res, i);
            if (seg_text) {
                if (!text.empty()) text += " ";
                text += seg_text;
            }
        }

        bool should_extract_words = true;
        if (m_backend_name.find("voxtral") != std::string::npos ||
            m_backend_name.find("vibevoice") != std::string::npos) {
            should_extract_words = false;
        }

        if (should_extract_words) {
            for (int i = 0; i < n_segs; i++) {
                int n_words = crispasr_session_result_n_words(res, i);
                for (int j = 0; j < n_words; j++) {
                    word_item wi;
                    const char* w_text = crispasr_session_result_word_text(res, i, j);
                    wi.word = w_text ? w_text : "";
                    wi.start = start_time + crispasr_session_result_word_t0(res, i, j) * 10;
                    wi.end = start_time + crispasr_session_result_word_t1(res, i, j) * 10;
                    wi.p = crispasr_session_result_word_p(res, i, j);
                    words_list.push_back(wi);
                }
            }
        }
        
        if (!text.empty() || type == "segment") {
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
        const int sample_rate = 16000;
        const int chunk_samples = (m_chunk_size_ms * sample_rate) / 1000;
        const char* lang_ptr = m_language.empty() ? nullptr : m_language.c_str();
        
        std::vector<float> speech_buffer;
        std::vector<float> pre_speech_cache;
        std::vector<float> accumulated_samples;
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
                                        emit_segment(res, chunk_start_ms, chunk_end_ms, m_segment_index, "progressive");
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
                            emit_segment(res, chunk_start_ms, chunk_end_ms, m_segment_index, "segment");
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
                                emit_segment(res, chunk_start_ms, chunk_end_ms, m_segment_index, "segment");
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
                        emit_segment(res, start, (final_sample * 1000) / sample_rate, m_segment_index, "segment");
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
    int m_n_threads = 4;
    bool m_use_gpu = true;
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
    std::vector<float> m_audio_buffer;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::thread m_worker_thread;
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_paused{false};
    std::atomic<bool> m_finishing{false};
    Napi::ThreadSafeFunction m_tsfn;
};

void InitCrispASR(Napi::Env env, Napi::Object exports) {
    printf("InitCrispASR: Initializing CrispASR exports...\n");
    exports.Set("parakeetASR", Napi::Function::New(env, parakeetASR));
    exports.Set("crispasrASR", Napi::Function::New(env, crispasrASR));
    exports.Set("distilWhisper", Napi::Function::New(env, distilWhisper));
    CrispASRStream::Init(env, exports);
}
