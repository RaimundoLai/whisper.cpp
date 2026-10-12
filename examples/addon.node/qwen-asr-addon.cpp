// qwen-asr-addon.cpp - Qwen3 ASR N-API bindings
// This file is included by addon.cpp

#include "model-cache.h"
#include "qwen3_asr.h"
#include "forced_aligner.h"
#include "stream-segmenter.h"

// ============================================================================
// Qwen3 ASR Transcription Worker
// ============================================================================



struct qwen_asr_addon_result {
    std::string full_text;
    struct timing_info {
        int64_t mel_ms = 0;
        int64_t encode_ms = 0;
        int64_t decode_ms = 0;
        int64_t total_ms = 0;
    } timing;
    std::vector<int32_t> all_tokens;
    
    struct segment_data {
        int64_t start_ms;
        int64_t end_ms;
        std::string text;
        std::string language;
        std::vector<int32_t> tokens;
        std::vector<qwen3_asr::aligned_word> words;
    };
    std::vector<segment_data> segments;
};

class QwenASRWorker : public Napi::AsyncWorker {
public:
    // Abort flags
    std::shared_ptr<std::atomic<bool>> m_should_abort;
    std::shared_ptr<std::atomic<bool>> m_was_aborted;

    QwenASRWorker(Napi::Function& callback,
                  std::string model_path,
                  std::string aligner_model_path,
                  std::string vad_model_path,
                  std::vector<float> pcmf32,
                  std::string language,
                  std::vector<std::string> hotwords,
                  int n_threads,
                  int max_tokens,
                  bool debug,
                  bool use_gpu,
                  float vad_threshold,
                  int min_speech_ms,
                  int max_speech_ms,
                  int min_silence_ms,
                  int speech_pad_ms,
                  bool reuse_instance,
                  int64_t auto_release_ms,
                  Napi::Function progress_callback,
                  Napi::Env env)
        : Napi::AsyncWorker(callback),
          m_model_path(std::move(model_path)),
          m_aligner_model_path(std::move(aligner_model_path)),
          m_vad_model_path(std::move(vad_model_path)),
          m_pcmf32(std::move(pcmf32)),
          m_language(std::move(language)),
          m_hotwords(std::move(hotwords)),
          m_n_threads(n_threads),
          m_max_tokens(max_tokens),
          m_debug(debug),
          m_use_gpu(use_gpu),
          m_vad_threshold(vad_threshold),
          m_min_speech_ms(min_speech_ms),
          m_max_speech_ms(max_speech_ms),
          m_min_silence_ms(min_silence_ms),
          m_speech_pad_ms(speech_pad_ms),
          m_reuse_instance(reuse_instance),
          m_auto_release_ms(auto_release_ms),
          env(env),
          m_should_abort(std::make_shared<std::atomic<bool>>(false)),
          m_was_aborted(std::make_shared<std::atomic<bool>>(false)) {
        if (!progress_callback.IsEmpty()) {
            tsfn = Napi::ThreadSafeFunction::New(
                env,
                progress_callback,
                "QwenASR Progress Callback",
                0,
                1
            );
        }
    }

    ~QwenASRWorker() {
        if (tsfn) {
            tsfn.Release();
        }
    }

    void Execute() override {
        auto& cache = ModelCache::instance();

        std::unique_lock<std::recursive_mutex> asr_lock(cache.mutex(ModelType::QWEN3_ASR));
        qwen3_asr::Qwen3ASR* asr_ptr = nullptr;
        bool asr_owned = false;

        if (m_reuse_instance) {
            asr_ptr = static_cast<qwen3_asr::Qwen3ASR*>(
                cache.acquire(ModelType::QWEN3_ASR, m_model_path, m_use_gpu));
        }

        if (!asr_ptr) {
            asr_ptr = new qwen3_asr::Qwen3ASR();
            if (!asr_ptr->load_model(m_model_path, m_use_gpu, m_debug)) {
                SetError("Failed to load model: " + asr_ptr->get_error());
                delete asr_ptr;
                return;
            }
            if (m_reuse_instance) {
                cache.store(ModelType::QWEN3_ASR, asr_ptr, m_model_path, m_use_gpu, "", m_auto_release_ms);
            } else {
                asr_owned = true;
            }
        }

        qwen3_asr::Qwen3ASR& asr = *asr_ptr;

        std::unique_lock<std::recursive_mutex> aligner_lock(cache.mutex(ModelType::QWEN3_ALIGNER));
        qwen3_asr::ForcedAligner* aligner_ptr = nullptr;
        bool aligner_owned = false;
        bool use_aligner = !m_aligner_model_path.empty();

        if (use_aligner) {
            if (m_reuse_instance) {
                aligner_ptr = static_cast<qwen3_asr::ForcedAligner*>(
                    cache.acquire(ModelType::QWEN3_ALIGNER, m_aligner_model_path, m_use_gpu));
            }
            if (!aligner_ptr) {
                aligner_ptr = new qwen3_asr::ForcedAligner();
                if (!aligner_ptr->load_model(m_aligner_model_path, m_use_gpu, m_debug)) {
                    asr_lock.unlock();
                    aligner_lock.unlock();
                    if (asr_owned) { delete asr_ptr; }
                    else { cache.markIdle(ModelType::QWEN3_ASR); }
                    SetError("Failed to load aligner model: " + aligner_ptr->get_error());
                    delete aligner_ptr;
                    return;
                }
                if (m_reuse_instance) {
                    cache.store(ModelType::QWEN3_ALIGNER, aligner_ptr, m_aligner_model_path, m_use_gpu, "", m_auto_release_ms);
                } else {
                    aligner_owned = true;
                }
            }
        }

        qwen3_asr::ForcedAligner& aligner = *aligner_ptr;

        auto abort_flag = m_should_abort;
        auto was_aborted = m_was_aborted;
        asr.set_progress_callback([this, abort_flag, was_aborted](int tokens_generated, int max_tokens) {
            if (abort_flag->load()) {
                was_aborted->store(true);
                return;
            }
            float chunk_progress = (max_tokens > 0) ? ((float)tokens_generated / max_tokens) : 0.0f;
            int global_progress = static_cast<int>((m_progress_base + chunk_progress * m_progress_scale) * 100);
            if (global_progress > 100) global_progress = 100;
            
            static int last_reported = -1;
            if (global_progress != last_reported) {
                last_reported = global_progress;
                OnProgress(global_progress);
            }
        });

        qwen3_asr::transcribe_params tp;
        tp.max_tokens = m_max_tokens;
        tp.language = m_language;
        tp.hotwords = m_hotwords;
        tp.n_threads = m_n_threads;
        tp.print_progress = m_debug;
        tp.print_timing = m_debug;

        if (m_vad_model_path.empty()) {
            m_progress_base = 0.0f;
            m_progress_scale = 1.0f;
            auto res = asr.transcribe(m_pcmf32.data(), m_pcmf32.size(), tp);
            if (!res.success) {
                asr_lock.unlock();
                aligner_lock.unlock();
                if (asr_owned) { delete asr_ptr; } else { cache.markIdle(ModelType::QWEN3_ASR); }
                if (use_aligner) { if (aligner_owned) { delete aligner_ptr; } else { cache.markIdle(ModelType::QWEN3_ALIGNER); } }
                SetError("Transcription failed: " + res.error_msg);
                return;
            }
            m_result.full_text = res.text;
            m_result.timing.mel_ms = res.t_mel_ms;
            m_result.timing.encode_ms = res.t_encode_ms;
            m_result.timing.decode_ms = res.t_decode_ms;
            m_result.timing.total_ms = res.t_total_ms;
            m_result.all_tokens = res.tokens;
            
            qwen_asr_addon_result::segment_data seg;
            seg.start_ms = 0;
            seg.end_ms = (m_pcmf32.size() * 1000) / 16000;
            seg.tokens = res.tokens;
            seg.language = res.language;
            seg.text = res.text;

            if (use_aligner && !seg.text.empty()) {
                auto align_res = aligner.align(m_pcmf32.data(), m_pcmf32.size(), seg.text, m_language);
                if (align_res.success) {
                    seg.words = align_res.words;
                }
            }
            m_result.segments.push_back(seg);
        } else {
            whisper_vad_context_params vad_ctx_params = whisper_vad_default_context_params();
            vad_ctx_params.n_threads = m_n_threads;
            vad_ctx_params.use_gpu = false;
            
            whisper_vad_context* vctx = whisper_vad_init_from_file_with_params(m_vad_model_path.c_str(), vad_ctx_params);
            if (!vctx) {
                asr_lock.unlock();
                aligner_lock.unlock();
                if (asr_owned) { delete asr_ptr; } else { cache.markIdle(ModelType::QWEN3_ASR); }
                if (use_aligner) { if (aligner_owned) { delete aligner_ptr; } else { cache.markIdle(ModelType::QWEN3_ALIGNER); } }
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

                        m_progress_base = (float)chunk_start / m_pcmf32.size();
                        m_progress_scale = (float)(chunk_end - chunk_start) / m_pcmf32.size();

                        std::vector<float> chunk(m_pcmf32.begin() + chunk_start, m_pcmf32.begin() + chunk_end);
                        auto res = asr.transcribe(chunk.data(), chunk.size(), tp);
                        if (res.success) {
                            m_result.full_text += res.text;
                            m_result.timing.mel_ms += res.t_mel_ms;
                            m_result.timing.encode_ms += res.t_encode_ms;
                            m_result.timing.decode_ms += res.t_decode_ms;
                            m_result.timing.total_ms += res.t_total_ms;
                            m_result.all_tokens.insert(m_result.all_tokens.end(), res.tokens.begin(), res.tokens.end());
                            
                            qwen_asr_addon_result::segment_data seg;
                            seg.start_ms = static_cast<int64_t>(chunk_t0 * 1000);
                            seg.end_ms = static_cast<int64_t>(chunk_t1 * 1000);
                            seg.tokens = res.tokens;
                            seg.language = res.language;
                            seg.text = res.text;

                            if (use_aligner && !seg.text.empty()) {
                                auto align_res = aligner.align(chunk.data(), chunk.size(), seg.text, m_language);
                                if (align_res.success) {
                                    for (auto& w : align_res.words) {
                                        w.start += chunk_t0;
                                        w.end += chunk_t0;
                                    }
                                    seg.words = align_res.words;
                                }
                            }

                            m_result.segments.push_back(seg);
                        }

                        chunk_start = chunk_end;
                    }
                }
                whisper_vad_free_segments(segments);
                OnProgress(100);
            }
            whisper_vad_free(vctx);
        }

        asr_lock.unlock();
        aligner_lock.unlock();
        if (asr_owned) { delete asr_ptr; } else { cache.markIdle(ModelType::QWEN3_ASR); }
        if (use_aligner) {
            if (aligner_owned) { delete aligner_ptr; } else { cache.markIdle(ModelType::QWEN3_ALIGNER); }
        }
    }

    void OnOK() override {
        Napi::HandleScope scope(Env());
        Napi::Object result = Napi::Object::New(Env());

        result.Set("text", Napi::String::New(Env(), m_result.full_text));

        // Timing info
        Napi::Object timing = Napi::Object::New(Env());
        timing.Set("mel_ms", Napi::Number::New(Env(), m_result.timing.mel_ms));
        timing.Set("encode_ms", Napi::Number::New(Env(), m_result.timing.encode_ms));
        timing.Set("decode_ms", Napi::Number::New(Env(), m_result.timing.decode_ms));
        timing.Set("total_ms", Napi::Number::New(Env(), m_result.timing.total_ms));
        result.Set("timing", timing);

        // Tokens array
        Napi::Array tokens = Napi::Array::New(Env(), m_result.all_tokens.size());
        for (size_t i = 0; i < m_result.all_tokens.size(); i++) {
            tokens[i] = Napi::Number::New(Env(), m_result.all_tokens[i]);
        }
        result.Set("tokens", tokens);

        // Segments array
        Napi::Array segments = Napi::Array::New(Env(), m_result.segments.size());
        for (size_t i = 0; i < m_result.segments.size(); i++) {
            Napi::Object seg = Napi::Object::New(Env());
            seg.Set("start", Napi::Number::New(Env(), m_result.segments[i].start_ms));
            seg.Set("end", Napi::Number::New(Env(), m_result.segments[i].end_ms));
            seg.Set("text", Napi::String::New(Env(), m_result.segments[i].text));
            seg.Set("language", Napi::String::New(Env(), m_result.segments[i].language));
            
            if (!m_result.segments[i].words.empty()) {
                Napi::Array words = Napi::Array::New(Env(), m_result.segments[i].words.size());
                for (size_t j = 0; j < m_result.segments[i].words.size(); j++) {
                    Napi::Object word = Napi::Object::New(Env());
                    word.Set("word", Napi::String::New(Env(), m_result.segments[i].words[j].word));
                    word.Set("start", Napi::Number::New(Env(), m_result.segments[i].words[j].start));
                    word.Set("end", Napi::Number::New(Env(), m_result.segments[i].words[j].end));
                    words[j] = word;
                }
                seg.Set("words", words);
            }

            segments[i] = seg;
        }
        result.Set("segments", segments);

        result.Set("aborted", Napi::Boolean::New(Env(), m_was_aborted->load()));

        Callback().Call({Env().Null(), result});
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
                    // Continue on error
                }
            };
            tsfn.BlockingCall(callback);
        }
    }

    bool ShouldAbort() const {
        return m_should_abort->load();
    }

private:
    std::string m_model_path;
    std::string m_aligner_model_path;
    std::string m_vad_model_path;
    std::vector<float> m_pcmf32;
    std::string m_language;
    std::vector<std::string> m_hotwords;
    int m_n_threads;
    int m_max_tokens;
    bool m_debug;
    bool m_use_gpu;
    
    float m_vad_threshold;
    int m_min_speech_ms;
    int m_max_speech_ms;
    int m_min_silence_ms;
    int m_speech_pad_ms;
    
    float m_progress_base = 0.0f;
    float m_progress_scale = 1.0f;
    bool m_reuse_instance = false;
    int64_t m_auto_release_ms = 0;

    Napi::Env env;
    Napi::ThreadSafeFunction tsfn;
    qwen_asr_addon_result m_result;
};

// ============================================================================
// Qwen3 ASR Transcription Entry Point
// ============================================================================

Napi::Value qwenASR(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2 || !info[0].IsObject() || !info[1].IsFunction()) {
        Napi::TypeError::New(env, "Usage: qwenASR(options, callback)").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Object options = info[0].As<Napi::Object>();

    // Required: model path
    if (!options.Has("model") || !options.Get("model").IsString()) {
        Napi::TypeError::New(env, "options.model (string) is required").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    std::string model_path = options.Get("model").As<Napi::String>();

    // Audio input: pcmf32 (Float32Array) or file path
    std::vector<float> pcmf32;

    if (options.Has("file") && options.Get("file").IsString()) {
        std::string file_path = options.Get("file").As<Napi::String>();
        int sample_rate = 0;
        if (!qwen3_asr::load_audio_file(file_path, pcmf32, sample_rate)) {
            Napi::TypeError::New(env, "Failed to load audio file: " + file_path).ThrowAsJavaScriptException();
            return env.Undefined();
        }
        if (sample_rate != 16000) {
            Napi::TypeError::New(env, "Audio must be 16kHz, got: " + std::to_string(sample_rate) + " Hz").ThrowAsJavaScriptException();
            return env.Undefined();
        }
    } else if (options.Has("pcmf32") && options.Get("pcmf32").IsTypedArray()) {
        Napi::Float32Array pcmf32_arr = options.Get("pcmf32").As<Napi::Float32Array>();
        pcmf32.reserve(pcmf32_arr.ElementLength());
        for (size_t i = 0; i < pcmf32_arr.ElementLength(); i++) {
            pcmf32.push_back(pcmf32_arr[i]);
        }
    } else {
        Napi::TypeError::New(env, "Either options.file (WAV path) or options.pcmf32 (Float32Array) is required").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Optional parameters
    std::string language = "";
    if (options.Has("language") && options.Get("language").IsString()) {
        language = options.Get("language").As<Napi::String>();
    }

    std::vector<std::string> hotwords;
    if (options.Has("hotwords") && options.Get("hotwords").IsArray()) {
        Napi::Array values = options.Get("hotwords").As<Napi::Array>();
        for (uint32_t i = 0; i < values.Length() && hotwords.size() < 128; ++i) {
            Napi::Value value = values.Get(i);
            if (value.IsString()) {
                std::string hotword = value.As<Napi::String>().Utf8Value();
                if (!hotword.empty() && hotword.size() <= 256) {
                    hotwords.push_back(std::move(hotword));
                }
            }
        }
    }

    int n_threads = std::min(4, static_cast<int>(std::thread::hardware_concurrency()));
    if (options.Has("n_threads") && options.Get("n_threads").IsNumber()) {
        n_threads = options.Get("n_threads").As<Napi::Number>().Int32Value();
    }

    int max_tokens = 1024;
    if (options.Has("max_tokens") && options.Get("max_tokens").IsNumber()) {
        max_tokens = options.Get("max_tokens").As<Napi::Number>().Int32Value();
    }

    bool debug = false;
    if (options.Has("debug") && options.Get("debug").IsBoolean()) {
        debug = options.Get("debug").As<Napi::Boolean>();
    }

    bool use_gpu = true;
    if (options.Has("use_gpu") && options.Get("use_gpu").IsBoolean()) {
        use_gpu = options.Get("use_gpu").As<Napi::Boolean>();
    }

    std::string aligner_model_path = "";
    if (options.Has("aligner_model") && options.Get("aligner_model").IsString()) {
        aligner_model_path = options.Get("aligner_model").As<Napi::String>();
    }

    std::string vad_model_path = "";
    if (options.Has("vad_model") && options.Get("vad_model").IsString()) {
        vad_model_path = options.Get("vad_model").As<Napi::String>();
    }

    float vad_threshold = 0.5f;
    if (options.Has("vad_threshold") && options.Get("vad_threshold").IsNumber()) {
        vad_threshold = options.Get("vad_threshold").As<Napi::Number>().FloatValue();
    }

    int min_speech_ms = 250;
    if (options.Has("min_speech_ms") && options.Get("min_speech_ms").IsNumber()) {
        min_speech_ms = options.Get("min_speech_ms").As<Napi::Number>().Int32Value();
    }

    int max_speech_ms = 15000;
    if (options.Has("max_speech_ms") && options.Get("max_speech_ms").IsNumber()) {
        max_speech_ms = options.Get("max_speech_ms").As<Napi::Number>().Int32Value();
    }

    int min_silence_ms = 100;
    if (options.Has("min_silence_ms") && options.Get("min_silence_ms").IsNumber()) {
        min_silence_ms = options.Get("min_silence_ms").As<Napi::Number>().Int32Value();
    }

    int speech_pad_ms = 400;
    if (options.Has("speech_pad_ms") && options.Get("speech_pad_ms").IsNumber()) {
        speech_pad_ms = options.Get("speech_pad_ms").As<Napi::Number>().Int32Value();
    }

    bool reuse_instance = false;
    if (options.Has("reuse_instance") && options.Get("reuse_instance").IsBoolean()) {
        reuse_instance = options.Get("reuse_instance").As<Napi::Boolean>();
    }

    int64_t auto_release_ms = 0;
    if (options.Has("auto_release_ms") && options.Get("auto_release_ms").IsNumber()) {
        auto_release_ms = options.Get("auto_release_ms").As<Napi::Number>().Int64Value();
    }

    // Progress callback (optional)
    Napi::Function progress_callback;
    if (options.Has("progress_callback") && options.Get("progress_callback").IsFunction()) {
        progress_callback = options.Get("progress_callback").As<Napi::Function>();
    }

    Napi::Function callback = info[1].As<Napi::Function>();
    QwenASRWorker* worker = new QwenASRWorker(
        callback, model_path, aligner_model_path, vad_model_path, std::move(pcmf32),
        language, std::move(hotwords), n_threads, max_tokens, debug, use_gpu,
        vad_threshold, min_speech_ms, max_speech_ms, min_silence_ms, speech_pad_ms,
        reuse_instance, auto_release_ms,
        progress_callback, env);
    worker->Queue();

    return env.Undefined();
}


class QwenASRAlignWorker : public Napi::AsyncWorker {
public:
    std::shared_ptr<std::atomic<bool>> m_should_abort;
    std::shared_ptr<std::atomic<bool>> m_was_aborted;

    QwenASRAlignWorker(Napi::Function& callback,
                       std::string model_path,
                       std::vector<float> pcmf32,
                       std::string text,
                       std::string language,
                       bool debug,
                       bool use_gpu,
                       bool reuse_instance,
                       int64_t auto_release_ms)
        : Napi::AsyncWorker(callback),
          m_model_path(std::move(model_path)),
          m_pcmf32(std::move(pcmf32)),
          m_text(std::move(text)),
          m_language(std::move(language)),
          m_debug(debug),
          m_use_gpu(use_gpu),
          m_reuse_instance(reuse_instance),
          m_auto_release_ms(auto_release_ms),
          m_should_abort(std::make_shared<std::atomic<bool>>(false)),
          m_was_aborted(std::make_shared<std::atomic<bool>>(false)) {}

    void Execute() override {
        auto& cache = ModelCache::instance();
        std::unique_lock<std::recursive_mutex> lock(cache.mutex(ModelType::QWEN3_ALIGNER));

        qwen3_asr::ForcedAligner* aligner_ptr = nullptr;
        bool owned = false;

        if (m_reuse_instance) {
            aligner_ptr = static_cast<qwen3_asr::ForcedAligner*>(
                cache.acquire(ModelType::QWEN3_ALIGNER, m_model_path, m_use_gpu));
        }

        if (!aligner_ptr) {
            aligner_ptr = new qwen3_asr::ForcedAligner();
            if (!aligner_ptr->load_model(m_model_path, m_use_gpu, m_debug)) {
                SetError("Failed to load aligner model: " + aligner_ptr->get_error());
                delete aligner_ptr;
                return;
            }
            if (m_reuse_instance) {
                cache.store(ModelType::QWEN3_ALIGNER, aligner_ptr, m_model_path, m_use_gpu, "", m_auto_release_ms);
            } else {
                owned = true;
            }
        }

        m_result = aligner_ptr->align(m_pcmf32.data(), m_pcmf32.size(), m_text, m_language);

        lock.unlock();
        if (owned) {
            delete aligner_ptr;
        } else {
            cache.markIdle(ModelType::QWEN3_ALIGNER);
        }

        if (!m_result.success) {
            SetError("Alignment failed: " + m_result.error_msg);
            return;
        }
    }

    void OnOK() override {
        Napi::HandleScope scope(Env());
        Napi::Object result = Napi::Object::New(Env());

        // Words array with timestamps
        Napi::Array words = Napi::Array::New(Env(), m_result.words.size());
        for (size_t i = 0; i < m_result.words.size(); i++) {
            Napi::Object word = Napi::Object::New(Env());
            word.Set("word", Napi::String::New(Env(), m_result.words[i].word));
            word.Set("start", Napi::Number::New(Env(), m_result.words[i].start));
            word.Set("end", Napi::Number::New(Env(), m_result.words[i].end));
            words[i] = word;
        }
        result.Set("words", words);

        // Timing info
        Napi::Object timing = Napi::Object::New(Env());
        timing.Set("mel_ms", Napi::Number::New(Env(), m_result.t_mel_ms));
        timing.Set("encode_ms", Napi::Number::New(Env(), m_result.t_encode_ms));
        timing.Set("decode_ms", Napi::Number::New(Env(), m_result.t_decode_ms));
        timing.Set("total_ms", Napi::Number::New(Env(), m_result.t_total_ms));
        result.Set("timing", timing);

        result.Set("aborted", Napi::Boolean::New(Env(), m_was_aborted->load()));

        Callback().Call({Env().Null(), result});
    }

private:
    std::string m_model_path;
    std::vector<float> m_pcmf32;
    std::string m_text;
    std::string m_language;
    bool m_debug;
    bool m_use_gpu;
    bool m_reuse_instance;
    int64_t m_auto_release_ms;
    qwen3_asr::alignment_result m_result;
};

// ============================================================================
// Qwen3 Forced Alignment Entry Point
// ============================================================================

Napi::Value qwenASRAlign(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2 || !info[0].IsObject() || !info[1].IsFunction()) {
        Napi::TypeError::New(env, "Usage: qwenASRAlign(options, callback)").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Object options = info[0].As<Napi::Object>();

    // Required: model path
    if (!options.Has("model") || !options.Get("model").IsString()) {
        Napi::TypeError::New(env, "options.model (string) is required").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    std::string model_path = options.Get("model").As<Napi::String>();

    // Required: text for alignment
    if (!options.Has("text") || !options.Get("text").IsString()) {
        Napi::TypeError::New(env, "options.text (string) is required").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    std::string text = options.Get("text").As<Napi::String>();

    // Audio input: pcmf32 (Float32Array) or file path
    std::vector<float> pcmf32;

    if (options.Has("file") && options.Get("file").IsString()) {
        std::string file_path = options.Get("file").As<Napi::String>();
        int sample_rate = 0;
        if (!qwen3_asr::load_audio_file(file_path, pcmf32, sample_rate)) {
            Napi::TypeError::New(env, "Failed to load audio file: " + file_path).ThrowAsJavaScriptException();
            return env.Undefined();
        }
        if (sample_rate != 16000) {
            Napi::TypeError::New(env, "Audio must be 16kHz, got: " + std::to_string(sample_rate) + " Hz").ThrowAsJavaScriptException();
            return env.Undefined();
        }
    } else if (options.Has("pcmf32") && options.Get("pcmf32").IsTypedArray()) {
        Napi::Float32Array pcmf32_arr = options.Get("pcmf32").As<Napi::Float32Array>();
        pcmf32.reserve(pcmf32_arr.ElementLength());
        for (size_t i = 0; i < pcmf32_arr.ElementLength(); i++) {
            pcmf32.push_back(pcmf32_arr[i]);
        }
    } else {
        Napi::TypeError::New(env, "Either options.file (WAV path) or options.pcmf32 (Float32Array) is required").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Optional parameters
    std::string language = "";
    if (options.Has("language") && options.Get("language").IsString()) {
        language = options.Get("language").As<Napi::String>();
    }

    bool debug = false;
    if (options.Has("debug") && options.Get("debug").IsBoolean()) {
        debug = options.Get("debug").As<Napi::Boolean>();
    }

    bool use_gpu = true;
    if (options.Has("use_gpu") && options.Get("use_gpu").IsBoolean()) {
        use_gpu = options.Get("use_gpu").As<Napi::Boolean>();
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
    QwenASRAlignWorker* worker = new QwenASRAlignWorker(
        callback, model_path, std::move(pcmf32),
        text, language, debug, use_gpu, reuse_instance, auto_release_ms);
    worker->Queue();

    return env.Undefined();
}
// ============================================================================
// Qwen3 ASR Stream Implementation
// ============================================================================

class QwenASRStream : public Napi::ObjectWrap<QwenASRStream> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports) {
        Napi::HandleScope scope(env);
        Napi::Function func = DefineClass(env, "QwenASRStream", {
            InstanceMethod("start", &QwenASRStream::Start),
            InstanceMethod("addAudio", &QwenASRStream::AddAudio),
            InstanceMethod("stop", &QwenASRStream::Stop),
            InstanceMethod("pause", &QwenASRStream::Pause),
            InstanceMethod("resume", &QwenASRStream::Resume),
            InstanceMethod("finish", &QwenASRStream::Finish),
            InstanceMethod("release", &QwenASRStream::Release)
        });
        exports.Set("QwenASRStream", func);
        return exports;
    }

    QwenASRStream(const Napi::CallbackInfo& info) : Napi::ObjectWrap<QwenASRStream>(info) {
        Napi::Env env = info.Env();
        
        if (info.Length() < 1 || !info[0].IsObject()) {
            Napi::TypeError::New(env, "QwenASRStream requires an options object").ThrowAsJavaScriptException();
            return;
        }
        
        Napi::Object options = info[0].As<Napi::Object>();
        
        // Required: model path
        if (!options.Has("model") || !options.Get("model").IsString()) {
            Napi::TypeError::New(env, "options.model (string) is required").ThrowAsJavaScriptException();
            return;
        }
        m_model_path = options.Get("model").As<Napi::String>();
        
        // Optional variables extraction
        if (options.Has("aligner") && options.Get("aligner").IsString()) 
            m_aligner_model_path = options.Get("aligner").As<Napi::String>();
            
        if (options.Has("language") && options.Get("language").IsString()) 
            m_language = options.Get("language").As<Napi::String>();

        if (options.Has("hotwords") && options.Get("hotwords").IsArray()) {
            Napi::Array values = options.Get("hotwords").As<Napi::Array>();
            for (uint32_t i = 0; i < values.Length() && m_hotwords.size() < 128; ++i) {
                Napi::Value value = values.Get(i);
                if (value.IsString()) {
                    std::string hotword = value.As<Napi::String>().Utf8Value();
                    if (!hotword.empty() && hotword.size() <= 256) {
                        m_hotwords.push_back(std::move(hotword));
                    }
                }
            }
        }

        if (options.Has("n_threads") && options.Get("n_threads").IsNumber()) 
            m_n_threads = options.Get("n_threads").As<Napi::Number>().Int32Value();
            
        if (options.Has("max_tokens") && options.Get("max_tokens").IsNumber()) 
            m_max_tokens = options.Get("max_tokens").As<Napi::Number>().Int32Value();
            
        if (options.Has("chunk_size_ms") && options.Get("chunk_size_ms").IsNumber()) 
            m_chunk_size_ms = options.Get("chunk_size_ms").As<Napi::Number>().Int32Value();
            
        if (options.Has("progressive_update") && options.Get("progressive_update").IsBoolean()) 
            m_progressive_update = options.Get("progressive_update").As<Napi::Boolean>();
        if (options.Has("progressive_window_ms") && options.Get("progressive_window_ms").IsNumber()) 
            m_progressive_window_ms = options.Get("progressive_window_ms").As<Napi::Number>().Int32Value();
        if (options.Has("progressive_initial_ms") && options.Get("progressive_initial_ms").IsNumber()) 
            m_progressive_initial_ms = options.Get("progressive_initial_ms").As<Napi::Number>().Int32Value();
        if (options.Has("progressive_window_tokens") && options.Get("progressive_window_tokens").IsNumber()) 
            m_progressive_window_tokens = options.Get("progressive_window_tokens").As<Napi::Number>().Int32Value();
            
        if (options.Has("min_mute_chunks") && options.Get("min_mute_chunks").IsNumber()) 
            m_min_mute_chunks = options.Get("min_mute_chunks").As<Napi::Number>().Int32Value();
            
        if (options.Has("max_nomute_chunks") && options.Get("max_nomute_chunks").IsNumber()) 
            m_max_nomute_chunks = options.Get("max_nomute_chunks").As<Napi::Number>().Int32Value();
            
        if (options.Has("vad_threshold") && options.Get("vad_threshold").IsNumber()) 
            m_vad_threshold = options.Get("vad_threshold").As<Napi::Number>().FloatValue();
            
        if (options.Has("vad_model") && options.Get("vad_model").IsString()) 
            m_vad_model_path = options.Get("vad_model").As<Napi::String>();
            
        if (options.Has("debug") && options.Get("debug").IsBoolean()) 
            m_debug = options.Get("debug").As<Napi::Boolean>();
            
        if (options.Has("use_gpu") && options.Get("use_gpu").IsBoolean()) 
            m_use_gpu = options.Get("use_gpu").As<Napi::Boolean>();

        stream_seg::parse_options(m_seg, options);
    }
    
    ~QwenASRStream() {
        if (m_state.load() != StreamState::IDLE) {
            m_state = StreamState::STOPPING;
            m_cv.notify_one();
            if (m_worker_thread.joinable()) m_worker_thread.join();
        }
        if (m_tsfn_callback) m_tsfn_callback.Release();
        
        m_asr.reset();
        m_aligner.reset();
        if (m_vctx) {
            whisper_vad_free(m_vctx);
            m_vctx = nullptr;
        }
    }
    
    Napi::Value Start(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (m_state.load() != StreamState::IDLE) {
            Napi::Error::New(env, "Stream already running").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        if (info.Length() < 1 || !info[0].IsFunction()) {
            Napi::TypeError::New(env, "start() requires a callback").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        
        Napi::Function callback = info[0].As<Napi::Function>();
        m_tsfn_callback = Napi::ThreadSafeFunction::New(env, callback, "QwenASRStreamCallback", 0, 1);
        m_segment_index = 0;
        m_state = StreamState::RUNNING;
        m_worker_thread = std::thread(&QwenASRStream::StreamWorker, this);
        return env.Undefined();
    }
    
    Napi::Value AddAudio(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (m_state.load() != StreamState::RUNNING && m_state.load() != StreamState::PAUSED) return env.Undefined();
        if (info.Length() < 1 || !info[0].IsTypedArray()) return env.Undefined();
        
        Napi::Float32Array arr = info[0].As<Napi::Float32Array>();
        float* data = arr.Data();
        size_t length = arr.ElementLength();
        
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_audio_buffer.insert(m_audio_buffer.end(), data, data + length);
        }
        m_cv.notify_one();
        return env.Undefined();
    }
    
    Napi::Value Pause(const Napi::CallbackInfo& info) {
        if (m_state.load() == StreamState::RUNNING) m_state = StreamState::PAUSED;
        return info.Env().Undefined();
    }
    
    Napi::Value Resume(const Napi::CallbackInfo& info) {
        if (m_state.load() == StreamState::PAUSED) {
            m_state = StreamState::RUNNING;
            m_cv.notify_one();
        }
        return info.Env().Undefined();
    }
    
    Napi::Value Stop(const Napi::CallbackInfo& info) {
        if (m_state.load() != StreamState::IDLE) {
            m_state = StreamState::STOPPING;
            m_cv.notify_one();
            if (m_worker_thread.joinable()) m_worker_thread.join();
            
            // Release memory immediately on main JS thread
            m_asr.reset();
            m_aligner.reset();
            if (m_vctx) {
                whisper_vad_free(m_vctx);
                m_vctx = nullptr;
            }
        }
        return info.Env().Undefined();
    }
    
    Napi::Value Finish(const Napi::CallbackInfo& info) {
        if (m_state.load() == StreamState::RUNNING || m_state.load() == StreamState::PAUSED) {
            m_state = StreamState::FINISHING;
            m_cv.notify_one();
        }
        return info.Env().Undefined();
    }
    
    // Explicitly release resources. Useful after Finish() completes.
    Napi::Value Release(const Napi::CallbackInfo& info) {
        if (m_state.load() != StreamState::IDLE) {
            m_state = StreamState::STOPPING;
            m_cv.notify_one();
            if (m_worker_thread.joinable()) m_worker_thread.join();
        }
        
        m_asr.reset();
        m_aligner.reset();
        if (m_vctx) {
            whisper_vad_free(m_vctx);
            m_vctx = nullptr;
        }
        return info.Env().Undefined();
    }

private:
    void StreamWorker() {
        m_asr = std::make_unique<qwen3_asr::Qwen3ASR>();
        if (!m_asr->load_model(m_model_path, m_use_gpu, m_debug)) {
            if (m_tsfn_callback) {
                m_tsfn_callback.BlockingCall([](Napi::Env env, Napi::Function cb) {
                    cb.Call({Napi::Error::New(env, "Failed to load ASR model").Value(), env.Null()});
                });
            }
            m_state = StreamState::IDLE;
            return;
        }
        
        if (m_state.load() == StreamState::STOPPING || m_state.load() == StreamState::IDLE) {
            return;
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
        
        qwen3_asr::transcribe_params tp;
        tp.max_tokens = m_max_tokens;
        tp.language = m_language;
        tp.hotwords = m_hotwords;
        tp.n_threads = m_n_threads;
        tp.print_progress = false;
        tp.print_timing = false;
        
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

        const size_t post_speech_pad_samples = (300 * sample_rate) / 1000;
        stream_seg::TailCache<stream_seg::Hypothesis> tail_cache;
        bool tail_attempted = false;
        std::string prev_hyp_text;
        bool skip_progressive_initial = false;

        auto silence_threshold = [&](int speech_chunks) {
            return stream_seg::silence_threshold(m_seg, m_min_mute_chunks, m_chunk_size_ms, speech_chunks);
        };

        // Emits a stable sentence prefix as a final segment and drops its audio.
        auto try_early_commit = [&](const stream_seg::Hypothesis& hyp) -> bool {
            auto plan = stream_seg::plan_commit(hyp, prev_hyp_text, m_seg);
            if (!plan) return false;
            const int64_t cut_sample = (plan->cut_ms * sample_rate) / 1000;
            const int64_t offset = cut_sample - speech_start_sample;
            if (offset <= 0 || offset >= static_cast<int64_t>(speech_buffer.size())) return false;

            stream_seg::Hypothesis head, rest;
            stream_seg::split_hypothesis(hyp, *plan, head, rest);
            emitHypothesis(head, m_segment_index, false);
            m_segment_index++;

            speech_buffer.erase(speech_buffer.begin(), speech_buffer.begin() + offset);
            speech_start_sample = cut_sample;
            speech_chunk_count = std::max(0, speech_chunk_count - static_cast<int>(offset / chunk_samples));
            if (!rest.text.empty()) {
                emitHypothesis(rest, m_segment_index, true);
            }
            prev_hyp_text = rest.text;
            skip_progressive_initial = true;
            tail_cache.reset();
            tail_attempted = false;
            return true;
        };
        
        while (true) {
            bool is_finishing = false;
            bool timeout = false;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                timeout = !m_cv.wait_for(lock, std::chrono::milliseconds(1000), [this] {
                    StreamState s = m_state.load();
                    return s == StreamState::STOPPING || s == StreamState::FINISHING || (s == StreamState::RUNNING && !m_audio_buffer.empty());
                });
                if (m_state.load() == StreamState::STOPPING) break;
                if (m_state.load() == StreamState::PAUSED) continue;
                is_finishing = (m_state.load() == StreamState::FINISHING);
                
                accumulated_samples.insert(accumulated_samples.end(), m_audio_buffer.begin(), m_audio_buffer.end());
                m_audio_buffer.clear();
            }
            
            if (timeout && m_state.load() == StreamState::RUNNING) {
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
                if (m_vad_model_path.empty()) {
                    float energy = 0.0f;
                    for (int j = 0; j < chunk_samples; j++) energy += vad_chunk[j] * vad_chunk[j];
                    energy = std::sqrt(energy / chunk_samples);
                    is_speech = energy > 0.01f;
                } else if (m_vctx) {
                    whisper_vad_detect_speech(m_vctx, vad_chunk.data(), vad_chunk.size());
                    float speech_prob = 0.0f;
                    int n_probs = whisper_vad_n_probs(m_vctx);
                    if (n_probs > 0) {
                        const float* probs = whisper_vad_probs(m_vctx);
                        for (int k = 0; k < n_probs; k++) if (probs[k] > speech_prob) speech_prob = probs[k];
                    }
                    is_speech = speech_prob >= m_vad_threshold;
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
                            if (temp_silence_count > silence_threshold(temp_speech_count)) {
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
                        prev_hyp_text.clear();
                        skip_progressive_initial = false;
                    }
                    speech_buffer.insert(speech_buffer.end(), vad_chunk.begin(), vad_chunk.end());
                    in_speech = true;
                    silence_chunk_count = 0;
                    speech_chunk_count++;
                    // New speech invalidates any speculative tail decode.
                    tail_cache.reset();
                    tail_attempted = false;
                    
                    if (m_progressive_update) {
                        int64_t current_sample = total_samples_received + i + chunk_samples;
                        int64_t elapsed_since_start_ms = ((current_sample - speech_start_sample) * 1000) / sample_rate;
                        
                        if (skip_progressive_initial || elapsed_since_start_ms >= m_progressive_initial_ms) {
                            int64_t elapsed_since_last_prog_ms = ((current_sample - last_progressive_sample) * 1000) / sample_rate;
                            if (elapsed_since_last_prog_ms >= m_progressive_window_ms) {
                                bool has_pending_audio = false;
                                {
                                    std::lock_guard<std::mutex> lock(m_mutex);
                                    has_pending_audio = !m_audio_buffer.empty();
                                }
                                bool skip_progressive = is_finishing || 
                                                        has_pending_audio || 
                                                        will_segment_end(c + 1, in_speech, silence_chunk_count, speech_chunk_count);
                                if (!skip_progressive) {
                                    int64_t chunk_start_ms = (speech_start_sample * 1000) / sample_rate;
                                    int64_t chunk_end_ms = (current_sample * 1000) / sample_rate;
                                    
                                    stream_seg::Hypothesis hyp;
                                    if (decodeHypothesis(speech_buffer, tp, chunk_start_ms, chunk_end_ms, hyp)) {
                                        if (!try_early_commit(hyp)) {
                                            emitHypothesis(hyp, m_segment_index, true);
                                            prev_hyp_text = hyp.text;
                                        }
                                    }
                                    last_progressive_sample = current_sample;
                                }
                            }
                        }
                    }
                    
                    // Split if speech exceeds maximum duration without silence
                    if (speech_chunk_count >= m_max_nomute_chunks) {
                        int64_t current_sample = total_samples_received + i + chunk_samples;
                        int64_t start_time_ms = (speech_start_sample * 1000) / sample_rate;
                        int64_t end_time_ms = (current_sample * 1000) / sample_rate;
                        
                        processAndOutput(speech_buffer, tp, start_time_ms, end_time_ms, m_segment_index, false);
                        
                        speech_buffer.clear();
                        speech_chunk_count = 0;
                        speech_start_sample = current_sample;
                        last_progressive_sample = current_sample;
                        m_segment_index++;
                        silence_chunk_count = 0;
                        in_speech = true;
                        prev_hyp_text.clear();
                        skip_progressive_initial = false;
                    }
                } else {
                    if (in_speech) {
                        silence_chunk_count++;
                        const int silence_limit = silence_threshold(speech_chunk_count);
                        if (silence_chunk_count <= silence_limit) {
                            speech_buffer.insert(speech_buffer.end(), vad_chunk.begin(), vad_chunk.end());
                        }

                        // Builds the exact window the final pass decodes: speech plus at
                        // most post_speech_pad_samples of trailing silence.
                        auto build_final_window = [&](int held_silence_chunks, std::vector<float>& pcm, int64_t& start_ms, int64_t& end_ms) {
                            int64_t speech_end_sample = total_samples_received + i + chunk_samples;
                            int64_t actual_end_sample = speech_end_sample - (silence_chunk_count * chunk_samples);
                            if (actual_end_sample < speech_start_sample) {
                                actual_end_sample = speech_end_sample;
                            }
                            pcm = speech_buffer;
                            size_t trim_samples = stream_seg::trailing_trim(
                                static_cast<size_t>(held_silence_chunks) * chunk_samples, post_speech_pad_samples);
                            if (trim_samples > 0 && pcm.size() > trim_samples) {
                                pcm.resize(pcm.size() - trim_samples);
                            }
                            start_ms = (speech_start_sample * 1000) / sample_rate;
                            end_ms = (actual_end_sample * 1000) / sample_rate;
                        };

                        if (silence_chunk_count > silence_limit) {
                            in_speech = false;
                            std::vector<float> transcribe_pcm;
                            int64_t start_ms = 0;
                            int64_t end_ms = 0;
                            build_final_window(std::min(silence_chunk_count, silence_limit), transcribe_pcm, start_ms, end_ms);

                            if (m_debug) {
                                fprintf(stderr, "[STREAM] final %lld-%lld ms: speculative tail %s\n", (long long)start_ms, (long long)end_ms,
                                        tail_cache.matches(transcribe_pcm.size(), start_ms, end_ms) ? "reused" : (tail_cache.valid ? "mismatch" : "not available"));
                            }
                            if (tail_cache.matches(transcribe_pcm.size(), start_ms, end_ms)) {
                                emitHypothesis(tail_cache.result, m_segment_index, false);
                            } else {
                                processAndOutput(transcribe_pcm, tp, start_ms, end_ms, m_segment_index, false);
                            }
                            speech_buffer.clear();
                            speech_chunk_count = 0;
                            pre_speech_cache.clear();
                            tail_cache.reset();
                            tail_attempted = false;
                            prev_hyp_text.clear();
                        } else if (m_seg.speculative_final && !tail_attempted &&
                                   static_cast<int64_t>(silence_chunk_count) * m_chunk_size_ms >= m_seg.tail_delay_ms) {
                            // Pending audio is not a reason to skip: at worst the speculative decode is wasted.
                            bool skip_tail = is_finishing ||
                                             will_segment_end(c + 1, in_speech, silence_chunk_count, speech_chunk_count);
                            if (!skip_tail) {
                                tail_attempted = true;
                                std::vector<float> tail_pcm;
                                int64_t start_ms = 0;
                                int64_t end_ms = 0;
                                build_final_window(silence_chunk_count, tail_pcm, start_ms, end_ms);
                                stream_seg::Hypothesis hyp;
                                if (decodeHypothesis(tail_pcm, tp, start_ms, end_ms, hyp)) {
                                    tail_cache.valid = true;
                                    tail_cache.pcm_size = tail_pcm.size();
                                    tail_cache.start_ms = start_ms;
                                    tail_cache.end_ms = end_ms;
                                    tail_cache.result = hyp;
                                    if (m_progressive_update) {
                                        emitHypothesis(hyp, m_segment_index, true);
                                    }
                                }
                            }
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
            
            if (is_finishing) {
                speech_buffer.insert(speech_buffer.end(), accumulated_samples.begin(), accumulated_samples.end());
                int64_t final_sample = total_samples_received + accumulated_samples.size();
                int64_t start = in_speech ? (speech_start_sample * 1000) / sample_rate : (total_samples_received * 1000) / sample_rate;
                if (!speech_buffer.empty()) {
                    if (!in_speech) {
                        m_segment_index++;
                    }
                    processAndOutput(speech_buffer, tp, start, (final_sample * 1000) / sample_rate, m_segment_index, false);
                }
                tail_cache.reset();
                break;
            }
        }
        
        if (m_tsfn_callback) {
            m_tsfn_callback.BlockingCall([](Napi::Env env, Napi::Function jsCallback) {
                Napi::Object result = Napi::Object::New(env);
                result.Set("type", "end");
                jsCallback.Call({env.Null(), result});
            });
        }
    }

    void emit_silence(double t) {
        if (m_tsfn_callback) {
            m_tsfn_callback.BlockingCall([t](Napi::Env env, Napi::Function jsCallback) {
                Napi::Object result = Napi::Object::New(env);
                result.Set("type", Napi::String::New(env, "silence"));
                result.Set("t", Napi::Number::New(env, t));
                jsCallback.Call({env.Null(), result});
            });
        }
    }

    // Transcribes (and aligns) audio into an engine-agnostic hypothesis.
    // Word times are absolute milliseconds.
    bool decodeHypothesis(std::vector<float>& audio, const qwen3_asr::transcribe_params& tp, int64_t start_ms, int64_t end_ms, stream_seg::Hypothesis& out) {
        if (!m_asr) return false;

        auto res = m_asr->transcribe(audio.data(), audio.size(), tp);
        if (!res.success) {
            fprintf(stderr, "[STREAM] processAndOutput transcribe failed: %s\n", res.error_msg.c_str());
            return false;
        }

        out = stream_seg::Hypothesis{};
        out.start_ms = start_ms;
        out.end_ms = end_ms;
        out.text = res.text;
        out.language = res.language;

        if (m_aligner && !m_aligner_model_path.empty() && !out.text.empty()) {
            std::string detected_lang = m_language.empty() ? out.language : m_language;
            qwen3_asr::alignment_result align_res = m_aligner->align(audio.data(), audio.size(), out.text, detected_lang);
            if (align_res.success) {
                out.words.reserve(align_res.words.size());
                for (const auto& w : align_res.words) {
                    stream_seg::Word word;
                    word.text = w.word;
                    word.t0_ms = start_ms + static_cast<int64_t>(std::llround(w.start * 1000.0));
                    word.t1_ms = start_ms + static_cast<int64_t>(std::llround(w.end * 1000.0));
                    out.words.push_back(std::move(word));
                }
                out.words_timed = true;
            } else if (m_debug) {
                fprintf(stderr, "[STREAM] Warning: Alignment failed for segment: %s\n", align_res.error_msg.c_str());
            }
        }
        return true;
    }

    void emitHypothesis(const stream_seg::Hypothesis& hyp, int segment_index, bool is_progressive) {
        if (!m_tsfn_callback) return;
        auto callback = [hyp, is_progressive, segment_index](Napi::Env env, Napi::Function jsCallback) {
            Napi::Object result = Napi::Object::New(env);
            result.Set("type", Napi::String::New(env, is_progressive ? "progressive" : "segment"));
            result.Set("index", Napi::Number::New(env, segment_index));
            result.Set("start", Napi::Number::New(env, hyp.start_ms));
            result.Set("end", Napi::Number::New(env, hyp.end_ms));
            result.Set("text", Napi::String::New(env, hyp.text));
            result.Set("language", Napi::String::New(env, hyp.language));

            if (hyp.words_timed) {
                Napi::Array wordsArr = Napi::Array::New(env, hyp.words.size());
                for (size_t i = 0; i < hyp.words.size(); i++) {
                    Napi::Object word = Napi::Object::New(env);
                    word.Set("word", Napi::String::New(env, hyp.words[i].text));
                    word.Set("start", Napi::Number::New(env, hyp.words[i].t0_ms / 1000.0));
                    word.Set("end", Napi::Number::New(env, hyp.words[i].t1_ms / 1000.0));
                    wordsArr[i] = word;
                }
                result.Set("words", wordsArr);
            }

            jsCallback.Call({env.Null(), result});
        };
        if (is_progressive) {
            m_tsfn_callback.NonBlockingCall(callback);
        } else {
            m_tsfn_callback.BlockingCall(callback);
        }
    }

    void processAndOutput(std::vector<float>& audio, const qwen3_asr::transcribe_params& tp, int64_t start_ms, int64_t end_ms, int segment_index, bool is_progressive = false) {
        stream_seg::Hypothesis hyp;
        if (decodeHypothesis(audio, tp, start_ms, end_ms, hyp)) {
            emitHypothesis(hyp, segment_index, is_progressive);
        }
    }
    // ASR State
    int m_segment_index = 0;
    std::unique_ptr<qwen3_asr::Qwen3ASR> m_asr;
    std::unique_ptr<qwen3_asr::ForcedAligner> m_aligner;
    whisper_vad_context* m_vctx = nullptr;
    
    // Config
    std::string m_model_path;
    bool m_progressive_update = false;
    int m_progressive_window_ms = 500;
    int m_progressive_initial_ms = 5000;
    int m_progressive_window_tokens = 3;
    std::string m_aligner_model_path;
    std::string m_vad_model_path;
    std::string m_language = "";
    std::vector<std::string> m_hotwords;
    int m_n_threads = 4;
    int m_max_tokens = 1024;
    int m_chunk_size_ms = 32;
    stream_seg::Options m_seg;
    int m_min_mute_chunks = 30; // ~1 sec
    int m_max_nomute_chunks = 1875; // ~60 sec
    float m_vad_threshold = 0.5f;
    bool m_debug = false;
    bool m_use_gpu = true;
    
    // Threading
    std::atomic<StreamState> m_state{StreamState::IDLE};
    std::thread m_worker_thread;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::vector<float> m_audio_buffer;
    Napi::ThreadSafeFunction m_tsfn_callback;
    
    // Helper to detect language
    static std::string detect_language(const std::string& text) {
        // very basic lang detect for fallback if needed inside cpp aligner proxy
        int score_han = 0, score_latin = 0;
        for (char c : text) {
            if ((c & 0x80) == 0 && isalpha(c)) score_latin++;
            else if ((c & 0x80) != 0) score_han++;
        }
        if (score_han > score_latin) return "zh";
        return "en";
    }
};

