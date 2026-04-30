#include "napi.h"
#include "whisper.h"

#include <string>
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <chrono>
#include <cmath>

// --- VADStream Class Definition ---
// Standalone VAD stream class for real-time voice activity detection

enum class VADStreamState {
    IDLE,
    RUNNING,
    PAUSED,
    STOPPING,
    FINISHING
};

struct VADSessionState {
    int64_t session_start_ms;
    int64_t pause_ms;
    int64_t resume_ms;
    int64_t total_paused_ms;
    bool is_paused;
};

class VADStream : public Napi::ObjectWrap<VADStream> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    VADStream(const Napi::CallbackInfo& info);
    ~VADStream();

private:
    // N-API Methods
    Napi::Value Start(const Napi::CallbackInfo& info);
    Napi::Value AddAudio(const Napi::CallbackInfo& info);
    Napi::Value Stop(const Napi::CallbackInfo& info);
    Napi::Value Pause(const Napi::CallbackInfo& info);
    Napi::Value Resume(const Napi::CallbackInfo& info);
    Napi::Value Finish(const Napi::CallbackInfo& info);
    Napi::Value GetSessionState(const Napi::CallbackInfo& info);

    // Internal worker
    void VADWorker();

    // Helper to get current timestamp in ms
    int64_t GetCurrentTimeMs() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
    }

    // VAD context
    whisper_vad_context* m_vad_ctx = nullptr;
    whisper_vad_params m_vad_params;

    // Model path and settings
    std::string m_model_path;
    int m_n_threads = 4;
    bool m_use_gpu = false;

    // Audio buffer
    std::deque<float> m_audio_buffer;
    int64_t m_n_samples_total = 0;  // Total samples received

    // Streaming VAD state
    bool m_in_speech = false;
    int64_t m_speech_start_sample = 0;
    std::vector<float> m_speech_audio;  // Audio during speech
    std::vector<float> m_prev_buffer;   // Lookback buffer for speech padding
    int m_silence_frames = 0;  // Consecutive silence frames
    
    // Streaming parameters
    float m_max_speech_duration_s = 15.0f;  // Force segment end after this duration
    int m_min_silence_duration_ms_stream = 500;  // Silence duration for streaming (longer than batch)
    
    // Progressive parameters
    bool m_progressive_update = false;
    int m_progressive_interval_ms = 500;
    int m_progressive_initial_ms = 5000;

    // Thread control
    std::thread m_worker_thread;
    std::atomic<VADStreamState> m_state;
    std::mutex m_mutex;
    std::condition_variable m_cv;

    // Callback
    Napi::ThreadSafeFunction m_tsfn_callback;

    // Session state
    VADSessionState m_session_state;
};

// --- Implementation ---

Napi::Object VADStream::Init(Napi::Env env, Napi::Object exports) {
    Napi::HandleScope scope(env);
    Napi::Function func = DefineClass(env, "VADStream", {
        InstanceMethod("start", &VADStream::Start),
        InstanceMethod("addAudio", &VADStream::AddAudio),
        InstanceMethod("stop", &VADStream::Stop),
        InstanceMethod("pause", &VADStream::Pause),
        InstanceMethod("resume", &VADStream::Resume),
        InstanceMethod("finish", &VADStream::Finish),
        InstanceMethod("getSessionState", &VADStream::GetSessionState),
    });
    exports.Set("VADStream", func);
    return exports;
}

VADStream::VADStream(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<VADStream>(info), m_state(VADStreamState::IDLE) {
    Napi::Env env = info.Env();
    
    if (info.Length() < 1 || !info[0].IsObject()) {
        Napi::TypeError::New(env, "Constructor requires an options object").ThrowAsJavaScriptException();
        return;
    }
    
    Napi::Object options = info[0].As<Napi::Object>();
    
    // Required: model path
    if (options.Has("model") && options.Get("model").IsString()) {
        m_model_path = options.Get("model").As<Napi::String>();
    } else {
        Napi::TypeError::New(env, "Constructor options must include a 'model' path").ThrowAsJavaScriptException();
        return;
    }
    
    // Optional: n_threads
    if (options.Has("n_threads") && options.Get("n_threads").IsNumber()) {
        m_n_threads = options.Get("n_threads").As<Napi::Number>();
    }
    
    // Optional: use_gpu
    if (options.Has("use_gpu") && options.Get("use_gpu").IsBoolean()) {
        m_use_gpu = options.Get("use_gpu").As<Napi::Boolean>();
    }
    
    // Initialize VAD params with defaults
    m_vad_params = whisper_vad_default_params();
    
    // Optional VAD parameters
    if (options.Has("threshold") && options.Get("threshold").IsNumber()) {
        m_vad_params.threshold = options.Get("threshold").As<Napi::Number>().FloatValue();
    }
    if (options.Has("min_speech_duration_ms") && options.Get("min_speech_duration_ms").IsNumber()) {
        m_vad_params.min_speech_duration_ms = options.Get("min_speech_duration_ms").As<Napi::Number>();
    }
    if (options.Has("min_silence_duration_ms") && options.Get("min_silence_duration_ms").IsNumber()) {
        m_vad_params.min_silence_duration_ms = options.Get("min_silence_duration_ms").As<Napi::Number>();
    }
    if (options.Has("speech_pad_ms") && options.Get("speech_pad_ms").IsNumber()) {
        m_vad_params.speech_pad_ms = options.Get("speech_pad_ms").As<Napi::Number>();
    }
    
    // Streaming-specific parameters
    if (options.Has("max_speech_duration_s") && options.Get("max_speech_duration_s").IsNumber()) {
        m_max_speech_duration_s = options.Get("max_speech_duration_s").As<Napi::Number>().FloatValue();
    }
    // For streaming, use longer silence duration (default 500ms) unless specified
    if (options.Has("min_silence_duration_ms") && options.Get("min_silence_duration_ms").IsNumber()) {
        m_min_silence_duration_ms_stream = options.Get("min_silence_duration_ms").As<Napi::Number>();
    }
    
    // Progressive parameters
    if (options.Has("progressive_update") && options.Get("progressive_update").IsBoolean()) {
        m_progressive_update = options.Get("progressive_update").As<Napi::Boolean>();
    }
    if (options.Has("progressive_interval_ms") && options.Get("progressive_interval_ms").IsNumber()) {
        m_progressive_interval_ms = options.Get("progressive_interval_ms").As<Napi::Number>().Int32Value();
    }
    if (options.Has("progressive_initial_ms") && options.Get("progressive_initial_ms").IsNumber()) {
        m_progressive_initial_ms = options.Get("progressive_initial_ms").As<Napi::Number>().Int32Value();
    }
    
    // Initialize session state
    m_session_state = {0, 0, 0, 0, false};
}

VADStream::~VADStream() {
    VADStreamState current_state = m_state.load();
    if (current_state != VADStreamState::IDLE) {
        m_state = VADStreamState::STOPPING;
        m_cv.notify_one();
        if (m_worker_thread.joinable()) {
            m_worker_thread.join();
        }
    }
    if (m_tsfn_callback) {
        m_tsfn_callback.Release();
    }
    if (m_vad_ctx) {
        whisper_vad_free(m_vad_ctx);
    }
}

Napi::Value VADStream::Start(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (info.Length() < 1 || !info[0].IsFunction()) {
        Napi::TypeError::New(env, "start() requires a callback function").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    
    if (m_state.load() != VADStreamState::IDLE) {
        Napi::Error::New(env, "VADStream has already been started").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    
    // Initialize VAD context
    struct whisper_vad_context_params ctx_params = whisper_vad_default_context_params();
    ctx_params.n_threads = m_n_threads;
    ctx_params.use_gpu = m_use_gpu;
    
    m_vad_ctx = whisper_vad_init_from_file_with_params(m_model_path.c_str(), ctx_params);
    if (m_vad_ctx == nullptr) {
        Napi::Error::New(env, "Failed to initialize VAD context from model").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    
    // Reset state
    m_audio_buffer.clear();
    m_speech_audio.clear();
    m_prev_buffer.clear();
    m_n_samples_total = 0;
    m_in_speech = false;
    m_speech_start_sample = 0;
    m_silence_frames = 0;
    
    // Initialize session state
    m_session_state.session_start_ms = GetCurrentTimeMs();
    m_session_state.pause_ms = 0;
    m_session_state.resume_ms = 0;
    m_session_state.total_paused_ms = 0;
    m_session_state.is_paused = false;
    
    // Setup callback
    Napi::Function callback = info[0].As<Napi::Function>();
    m_tsfn_callback = Napi::ThreadSafeFunction::New(env, callback, "VADStreamCallback", 0, 1);
    
    // Start worker thread
    m_state = VADStreamState::RUNNING;
    m_worker_thread = std::thread(&VADStream::VADWorker, this);
    
    return env.Undefined();
}

Napi::Value VADStream::AddAudio(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    VADStreamState current_state = m_state.load();
    
    if (current_state != VADStreamState::RUNNING && current_state != VADStreamState::PAUSED) {
        return env.Undefined();
    }
    
    if (info.Length() < 1 || !info[0].IsTypedArray()) {
        Napi::TypeError::New(env, "addAudio() requires a Float32Array").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    
    // Limit buffer size to prevent memory issues (max 60 seconds)
    if (m_audio_buffer.size() > WHISPER_SAMPLE_RATE * 60) {
        fprintf(stderr, "Warning: VADStream audio buffer is too large, dropping new audio.\n");
        return env.Undefined();
    }
    
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

Napi::Value VADStream::Stop(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    VADStreamState current_state = m_state.load();
    
    if (current_state == VADStreamState::IDLE || current_state == VADStreamState::STOPPING) {
        return env.Undefined();
    }
    
    m_state = VADStreamState::STOPPING;
    m_cv.notify_one();
    
    if (m_worker_thread.joinable()) {
        m_worker_thread.join();
    }
    
    if (m_tsfn_callback) {
        m_tsfn_callback.Abort();
        m_tsfn_callback.Release();
        m_tsfn_callback = nullptr;
    }
    
    if (m_vad_ctx) {
        whisper_vad_free(m_vad_ctx);
        m_vad_ctx = nullptr;
    }
    
    m_state = VADStreamState::IDLE;
    return env.Undefined();
}

Napi::Value VADStream::Pause(const Napi::CallbackInfo& info) {
    if (m_state.load() == VADStreamState::RUNNING) {
        m_state = VADStreamState::PAUSED;
        m_session_state.pause_ms = GetCurrentTimeMs();
        m_session_state.is_paused = true;
    }
    return info.Env().Undefined();
}

Napi::Value VADStream::Resume(const Napi::CallbackInfo& info) {
    if (m_state.load() == VADStreamState::PAUSED) {
        int64_t now = GetCurrentTimeMs();
        m_session_state.resume_ms = now;
        m_session_state.total_paused_ms += (now - m_session_state.pause_ms);
        m_session_state.is_paused = false;
        m_state = VADStreamState::RUNNING;
        m_cv.notify_one();
    }
    return info.Env().Undefined();
}

Napi::Value VADStream::GetSessionState(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Object result = Napi::Object::New(env);
    
    result.Set("startMs", Napi::Number::New(env, m_session_state.session_start_ms));
    result.Set("pauseMs", Napi::Number::New(env, m_session_state.pause_ms));
    result.Set("resumeMs", Napi::Number::New(env, m_session_state.resume_ms));
    result.Set("totalPausedMs", Napi::Number::New(env, m_session_state.total_paused_ms));
    result.Set("isPaused", Napi::Boolean::New(env, m_session_state.is_paused));
    
    return result;
}

void VADStream::VADWorker() {
    // Silero VAD processes 512 samples (32ms) at a time at 16kHz
    // We'll accumulate chunks and process them
    const size_t chunk_samples = 512;  // 32ms at 16kHz
    const size_t process_buffer_size = chunk_samples * 8;  // Process 256ms at a time
    
    int64_t last_progressive_sample = 0;
    
    // Calculate silence/speech thresholds in frames
    // Use streaming-specific silence duration (longer for better results)
    const int min_speech_frames = (m_vad_params.min_speech_duration_ms * WHISPER_SAMPLE_RATE) / (1000 * chunk_samples);
    const int min_silence_frames = (m_min_silence_duration_ms_stream * WHISPER_SAMPLE_RATE) / (1000 * chunk_samples);
    const size_t max_speech_samples = (size_t)(m_max_speech_duration_s * WHISPER_SAMPLE_RATE);
    
    std::vector<float> process_buffer;
    int speech_frames = 0;
    
    // Helper lambda to emit voice segment
    auto emit_segment = [this](bool is_progressive = false) {
        if (!m_speech_audio.empty() && m_tsfn_callback) {
            float start_s = (float)m_speech_start_sample / WHISPER_SAMPLE_RATE;
            float end_s = (float)(m_speech_start_sample + m_speech_audio.size()) / WHISPER_SAMPLE_RATE;
            
            auto audio_copy = m_speech_audio;
            auto callback = [start_s, end_s, audio_copy, is_progressive](Napi::Env env, Napi::Function jsCallback) {
                Napi::Object result = Napi::Object::New(env);
                result.Set("type", Napi::String::New(env, is_progressive ? "progressive" : "voice"));
                result.Set("start", Napi::Number::New(env, start_s));
                result.Set("end", Napi::Number::New(env, end_s));
                
                Napi::Float32Array audio = Napi::Float32Array::New(env, audio_copy.size());
                for (size_t j = 0; j < audio_copy.size(); j++) {
                    audio[j] = audio_copy[j];
                }
                result.Set("audio", audio);
                jsCallback.Call({env.Null(), result});
            };
            m_tsfn_callback.NonBlockingCall(callback);
        }
        if (!is_progressive) {
            m_in_speech = false;
            m_speech_audio.clear();
            m_silence_frames = 0;
        }
    };
    
    while (true) {
        std::vector<float> new_audio;
        bool is_finishing = false;

        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait(lock, [this, process_buffer_size] {
                VADStreamState s = m_state.load();
                return s == VADStreamState::STOPPING ||
                       s == VADStreamState::FINISHING ||
                       (s == VADStreamState::RUNNING && m_audio_buffer.size() >= process_buffer_size);
            });
            
            VADStreamState current_state = m_state.load();
            if (current_state == VADStreamState::STOPPING) {
                // Emit any remaining speech segment before stopping
                if (m_in_speech) {
                    emit_segment(false);
                }
                break;
            }
            if (current_state == VADStreamState::PAUSED) {
                continue;
            }

            is_finishing = (current_state == VADStreamState::FINISHING);

            // If finishing, ignore process_buffer_size and consume all remaining audio
            size_t to_copy = is_finishing ? m_audio_buffer.size() : std::min(m_audio_buffer.size(), process_buffer_size);
            if (to_copy > 0) {
                new_audio.assign(m_audio_buffer.begin(), m_audio_buffer.begin() + to_copy);
                m_audio_buffer.erase(m_audio_buffer.begin(), m_audio_buffer.begin() + to_copy);
            }
        }

        if (new_audio.empty() && !is_finishing) {
            continue;
        }
        
        // Add to process buffer
        process_buffer.insert(process_buffer.end(), new_audio.begin(), new_audio.end());

        // [Key 1]: Handle remaining audio < 512 samples in finishing state
        if (is_finishing && !process_buffer.empty() && process_buffer.size() % chunk_samples != 0) {
            // Zero-pad to multiple of chunk_samples for VAD processing
            size_t padding_needed = chunk_samples - (process_buffer.size() % chunk_samples);
            process_buffer.insert(process_buffer.end(), padding_needed, 0.0f);
        }

        // If empty after padding and finishing, finalize immediately
        if (process_buffer.empty()) {
            if (is_finishing) {
                if (m_in_speech) {
                    emit_segment(false);
                }
                break; // Break loop to send 'end' event
            }
            continue;
        }

        bool success = whisper_vad_detect_speech(m_vad_ctx, process_buffer.data(), process_buffer.size());
        if (!success) {
            fprintf(stderr, "VAD detection failed\n");
            m_n_samples_total += process_buffer.size();
            process_buffer.clear();
            if (is_finishing) {
                if (m_in_speech) emit_segment(false);
                break;
            }
            continue;
        }
        
        // Get probabilities
        int n_probs = whisper_vad_n_probs(m_vad_ctx);
        float* probs = whisper_vad_probs(m_vad_ctx);

        if (n_probs > 0 && probs != nullptr) {
            for (int i = 0; i < n_probs; i++) {
                float prob = probs[i];
                bool is_speech = prob >= m_vad_params.threshold;

                size_t frame_start = i * chunk_samples;
                size_t frame_end = std::min(frame_start + chunk_samples, process_buffer.size());

                if (is_speech) {
                    m_silence_frames = 0;
                    speech_frames++;

                    if (!m_in_speech) {
                        if (speech_frames >= min_speech_frames) {
                            m_in_speech = true;
                            int speech_confirm_samples = (speech_frames - 1) * chunk_samples;
                            m_speech_start_sample = m_n_samples_total + frame_start - speech_confirm_samples;

                            int pad_samples = (m_vad_params.speech_pad_ms * WHISPER_SAMPLE_RATE) / 1000;
                            int total_lookback = pad_samples + speech_confirm_samples;

                            m_speech_audio.clear();

                            if ((int)frame_start < total_lookback && !m_prev_buffer.empty()) {
                                int needed_from_prev = total_lookback - (int)frame_start;
                                if (needed_from_prev > (int)m_prev_buffer.size()) {
                                    needed_from_prev = (int)m_prev_buffer.size();
                                }
                                m_speech_audio.insert(m_speech_audio.end(), m_prev_buffer.end() - needed_from_prev, m_prev_buffer.end());
                                m_speech_audio.insert(m_speech_audio.end(), process_buffer.begin(), process_buffer.begin() + frame_end);
                                m_speech_start_sample -= needed_from_prev;
                                if (m_speech_start_sample < 0) m_speech_start_sample = 0;
                            } else {
                                size_t start_idx = 0;
                                if (frame_start > (size_t)total_lookback) {
                                    start_idx = frame_start - total_lookback;
                                }
                                m_speech_audio.assign(process_buffer.begin() + start_idx, process_buffer.begin() + frame_end);
                            }
                            last_progressive_sample = m_speech_start_sample;
                        }
                    } else {
                        if (frame_start < process_buffer.size()) {
                            m_speech_audio.insert(m_speech_audio.end(), process_buffer.begin() + frame_start, process_buffer.begin() + frame_end);
                        }

                        if (m_progressive_update) {
                            int64_t current_sample = m_n_samples_total + frame_end;
                            int64_t elapsed_since_start_ms = ((current_sample - m_speech_start_sample) * 1000) / WHISPER_SAMPLE_RATE;

                            if (elapsed_since_start_ms >= m_progressive_initial_ms) {
                                int64_t elapsed_since_last_prog_ms = ((current_sample - last_progressive_sample) * 1000) / WHISPER_SAMPLE_RATE;
                                if (elapsed_since_last_prog_ms >= m_progressive_interval_ms) {
                                    emit_segment(true);
                                    last_progressive_sample = current_sample;
                                }
                            }
                        }

                        if (m_speech_audio.size() >= max_speech_samples) {
                            emit_segment(false);
                            speech_frames = 0;
                        }
                    }
                } else {
                    if (m_in_speech) {
                        if (frame_start < process_buffer.size()) {
                            m_speech_audio.insert(m_speech_audio.end(), process_buffer.begin() + frame_start, process_buffer.begin() + frame_end);
                        }
                        m_silence_frames++;
                        if (m_silence_frames >= min_silence_frames) {
                            emit_segment(false);
                        }
                    }
                    speech_frames = 0;
                }
            }
        }
        
        m_n_samples_total += process_buffer.size();
        m_prev_buffer = std::move(process_buffer);
        process_buffer.clear();

        // [Key 2]: Force end segment after processing final buffer in finishing state
        if (is_finishing) {
            if (m_in_speech) {
                // Force emit segment regardless of silence duration
                emit_segment(false);
            }
            break; // Exit worker loop
        }
    }
    
    // Send end event
    if (m_tsfn_callback) {
        m_tsfn_callback.BlockingCall([](Napi::Env env, Napi::Function jsCallback) {
            Napi::Object result = Napi::Object::New(env);
            result.Set("type", Napi::String::New(env, "end"));
            jsCallback.Call({env.Null(), result});
        });
    }
}
Napi::Value VADStream::Finish(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    VADStreamState current_state = m_state.load();
    
    if (current_state == VADStreamState::RUNNING || current_state == VADStreamState::PAUSED) {
        m_state = VADStreamState::FINISHING;
        m_cv.notify_one(); 
    }
    return env.Undefined();
}
// ============================================================================
// Batch VAD Detection Function
// ============================================================================

/**
 * vadDetect - Batch VAD detection
 * 
 * Usage:
 *   const segments = vadDetect({
 *       model: './ggml-silero.bin',
 *       pcmf32: float32Array,
 *       threshold: 0.5,
 *       min_speech_duration_ms: 250,
 *       min_silence_duration_ms: 100,
 *       speech_pad_ms: 30,
 *       mode: 'speech'  // 'speech' or 'silence'
 *   });
 *   // Returns: [{ start: 0.5, end: 2.3 }, { start: 5.1, end: 8.2 }]
 */
Napi::Value vadDetect(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (info.Length() < 1 || !info[0].IsObject()) {
        Napi::TypeError::New(env, "vadDetect requires an options object").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    
    Napi::Object options = info[0].As<Napi::Object>();
    
    // Required: model path
    std::string model_path;
    if (options.Has("model") && options.Get("model").IsString()) {
        model_path = options.Get("model").As<Napi::String>();
    } else {
        Napi::TypeError::New(env, "vadDetect options must include a 'model' path").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    
    // Required: pcmf32 audio data
    std::vector<float> pcmf32;
    if (options.Has("pcmf32") && options.Get("pcmf32").IsTypedArray()) {
        Napi::Float32Array arr = options.Get("pcmf32").As<Napi::Float32Array>();
        float* data = arr.Data();
        size_t length = arr.ElementLength();
        pcmf32.assign(data, data + length);
    } else {
        Napi::TypeError::New(env, "vadDetect options must include 'pcmf32' Float32Array").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    
    if (pcmf32.empty()) {
        return Napi::Array::New(env, 0);
    }
    
    // Optional parameters
    int n_threads = 4;
    bool use_gpu = false;
    std::string mode = "speech";  // 'speech' or 'silence'
    
    if (options.Has("n_threads") && options.Get("n_threads").IsNumber()) {
        n_threads = options.Get("n_threads").As<Napi::Number>();
    }
    if (options.Has("use_gpu") && options.Get("use_gpu").IsBoolean()) {
        use_gpu = options.Get("use_gpu").As<Napi::Boolean>();
    }
    if (options.Has("mode") && options.Get("mode").IsString()) {
        mode = options.Get("mode").As<Napi::String>();
    }
    
    // VAD params
    whisper_vad_params vad_params = whisper_vad_default_params();
    
    if (options.Has("threshold") && options.Get("threshold").IsNumber()) {
        vad_params.threshold = options.Get("threshold").As<Napi::Number>().FloatValue();
    }
    if (options.Has("min_speech_duration_ms") && options.Get("min_speech_duration_ms").IsNumber()) {
        vad_params.min_speech_duration_ms = options.Get("min_speech_duration_ms").As<Napi::Number>();
    }
    if (options.Has("min_silence_duration_ms") && options.Get("min_silence_duration_ms").IsNumber()) {
        vad_params.min_silence_duration_ms = options.Get("min_silence_duration_ms").As<Napi::Number>();
    }
    if (options.Has("speech_pad_ms") && options.Get("speech_pad_ms").IsNumber()) {
        vad_params.speech_pad_ms = options.Get("speech_pad_ms").As<Napi::Number>();
    }
    
    // Initialize VAD context
    struct whisper_vad_context_params ctx_params = whisper_vad_default_context_params();
    ctx_params.n_threads = n_threads;
    ctx_params.use_gpu = use_gpu;
    
    whisper_vad_context* vad_ctx = whisper_vad_init_from_file_with_params(model_path.c_str(), ctx_params);
    if (vad_ctx == nullptr) {
        Napi::Error::New(env, "Failed to initialize VAD context from model").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    
    // Run VAD detection
    struct whisper_vad_segments* segments = whisper_vad_segments_from_samples(
        vad_ctx, vad_params, pcmf32.data(), pcmf32.size());
    
    if (segments == nullptr) {
        whisper_vad_free(vad_ctx);
        Napi::Error::New(env, "VAD detection failed").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    
    int n_segments = whisper_vad_segments_n_segments(segments);
    float total_duration = (float)pcmf32.size() / WHISPER_SAMPLE_RATE;
    
    // Build result array based on mode
    if (mode == "speech") {
        // Return speech segments
        Napi::Array result = Napi::Array::New(env, n_segments);
        
        for (int i = 0; i < n_segments; i++) {
            // Note: whisper_vad_segments_get_segment_t0/t1 returns centiseconds
            float t0 = whisper_vad_segments_get_segment_t0(segments, i) / 100.0f;
            float t1 = whisper_vad_segments_get_segment_t1(segments, i) / 100.0f;
            
            Napi::Object seg = Napi::Object::New(env);
            seg.Set("start", Napi::Number::New(env, t0));
            seg.Set("end", Napi::Number::New(env, t1));
            result[i] = seg;
        }
        
        whisper_vad_free_segments(segments);
        whisper_vad_free(vad_ctx);
        return result;
        
    } else {
        // Return silence segments (inverse of speech)
        std::vector<std::pair<float, float>> silence_segments;
        
        float last_end = 0.0f;
        for (int i = 0; i < n_segments; i++) {
            // Note: whisper_vad_segments_get_segment_t0/t1 returns centiseconds
            float t0 = whisper_vad_segments_get_segment_t0(segments, i) / 100.0f;
            float t1 = whisper_vad_segments_get_segment_t1(segments, i) / 100.0f;
            
            // Add silence before this speech segment
            if (t0 > last_end) {
                silence_segments.push_back({last_end, t0});
            }
            last_end = t1;
        }
        
        // Add silence after last speech segment
        if (last_end < total_duration) {
            silence_segments.push_back({last_end, total_duration});
        }
        
        Napi::Array result = Napi::Array::New(env, silence_segments.size());
        for (size_t i = 0; i < silence_segments.size(); i++) {
            Napi::Object seg = Napi::Object::New(env);
            seg.Set("start", Napi::Number::New(env, silence_segments[i].first));
            seg.Set("end", Napi::Number::New(env, silence_segments[i].second));
            result[i] = seg;
        }
        
        whisper_vad_free_segments(segments);
        whisper_vad_free(vad_ctx);
        return result;
    }
}

