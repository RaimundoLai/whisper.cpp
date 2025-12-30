// sense-voice-addon.cpp - SenseVoice N-API bindings
// This file is included by addon.cpp

#include "sense-voice.h"
#include "sense-voice-frontend.h"
#include "whisper.h"  // For whisper VAD API

// ============================================================================
// SenseVoice ASR Implementation - Regular Mode with Segment Support
// ============================================================================

// Common Cantonese-specific characters (Cantonese characters)
// These are used almost exclusively in written Cantonese
static bool is_cantonese_char(uint32_t cp) {
    // 嘅(0x5605) 咁(0x5481) 佢(0x4F62) 冇(0x5187) 嚟(0x569F) 啲(0x5572) 
    // 咩(0x54A9) 喺(0x55BA) 噉(0x5649) 嗰(0x55F0) 係(0x4FC2) 唔(0x5514) 
    // 嘢(0x5622) 咗(0x5497) 畀(0x754C) 喐(0x5590) 嗮(0x55EE) 嚇(0x5687)
    switch(cp) {
        case 0x5605: case 0x5481: case 0x4F62: case 0x5187: case 0x569F: case 0x5572:
        case 0x54A9: case 0x55BA: case 0x5649: case 0x55F0: case 0x4FC2: case 0x5514:
        case 0x5622: case 0x5497: case 0x754C: case 0x5590: case 0x55EE: case 0x5687:
            return true;
        default:
            return false;
    }
}

// Check if character is sentence-ending punctuation (for subtitle splitting)
static bool is_sentence_ending_punctuation(const std::string& s, size_t pos) {
    if (pos >= s.length()) return false;
    unsigned char c = s[pos];
    
    // ASCII punctuation
    if (c == '.' || c == '!' || c == '?' || c == ';' || c == ',') return true;
    
    // Chinese/Japanese punctuation (UTF-8 encoded)
    if (pos + 2 < s.length()) {
        // Common CJK punctuation marks (3-byte UTF-8)
        unsigned char c1 = s[pos + 1];
        unsigned char c2 = s[pos + 2];
        
        // 。(U+3002) = E3 80 82
        if (c == 0xE3 && c1 == 0x80 && c2 == 0x82) return true;
        // ！(U+FF01) = EF BC 81
        if (c == 0xEF && c1 == 0xBC && c2 == 0x81) return true;
        // ？(U+FF1F) = EF BC 9F
        if (c == 0xEF && c1 == 0xBC && c2 == 0x9F) return true;
        // ；(U+FF1B) = EF BC 9B
        if (c == 0xEF && c1 == 0xBC && c2 == 0x9B) return true;
        // ，(U+FF0C) = EF BC 8C
        if (c == 0xEF && c1 == 0xBC && c2 == 0x8C) return true;
        // 、(U+3001) = E3 80 81
        if (c == 0xE3 && c1 == 0x80 && c2 == 0x81) return true;
    }
    
    return false;
}

// Helper to detect language from text (supports zh, en, ja, ko, yue)
static std::string detect_language(const std::string& text) {
    bool has_kana = false;
    bool has_hangul = false;
    int score_han = 0;
    int score_latin = 0;
    int score_cantonese = 0;

    for (size_t i = 0; i < text.length(); ) {
        unsigned char c = text[i];
        
        if (c < 0x80) {
            // ASCII
            if (isalpha(c)) score_latin++;
            i++;
        } else {
            // UTF-8
            uint32_t codepoint = 0;
            int bytes = 0;
            
            if ((c & 0xE0) == 0xC0) {
                codepoint = c & 0x1F;
                bytes = 2;
            } else if ((c & 0xF0) == 0xE0) {
                codepoint = c & 0x0F;
                bytes = 3;
            } else if ((c & 0xF8) == 0xF0) {
                codepoint = c & 0x07;
                bytes = 4;
            } else {
                i++; // Invalid, skip
                continue;
            }
            
            if (i + bytes > text.length()) break;
            
            for (int k = 1; k < bytes; k++) {
                codepoint = (codepoint << 6) | (text[i + k] & 0x3F);
            }
            
            // Check Cantonese-specific characters first
            if (is_cantonese_char(codepoint)) {
                score_cantonese++;
            }
            
            // Check ranges
            if (codepoint >= 0x4E00 && codepoint <= 0x9FFF) {
                score_han++;
            } else if ((codepoint >= 0x3040 && codepoint <= 0x309F) || // Hiragana
                       (codepoint >= 0x30A0 && codepoint <= 0x30FF)) { // Katakana
                has_kana = true;
            } else if ((codepoint >= 0xAC00 && codepoint <= 0xD7A3) || // Hangul Syllables
                       (codepoint >= 0x1100 && codepoint <= 0x11FF)) { // Hangul Jamo
                has_hangul = true;
            }
            
            i += bytes;
        }
    }
    
    if (has_kana) return "ja";
    if (has_hangul) return "ko";
    
    // If multiple Cantonese-specific characters found, likely Cantonese
    if (score_cantonese >= 2 && score_han > score_latin) return "yue";
    
    if (score_han > score_latin) return "zh";
    if (score_latin > 0) return "en";
    
    return "auto";
}

// Helper to check if a codepoint is sentence-ending/splitting punctuation
// Supports: Chinese/Cantonese (，。), Japanese (、。), English (,.), Korean (.)
static bool is_sentence_end(uint32_t cp) {
    // Full-width: 。(0x3002) ！(0xFF01) ？(0xFF1F) ，(0xFF0C) 、(0x3001)
    // Half-width: . (0x002E) ! (0x0021) ? (0x003F) , (0x002C)
    return cp == 0x3002 || cp == 0xFF01 || cp == 0xFF1F || cp == 0xFF0C || cp == 0x3001 ||
           cp == 0x002E || cp == 0x0021 || cp == 0x003F || cp == 0x002C;
}

// Split a segment by sentence-ending punctuation for better subtitle display
// Returns a vector of (relative_offset_ratio, text) pairs
static std::vector<std::pair<float, std::string>> split_by_punctuation(const std::string& text) {
    std::vector<std::pair<float, std::string>> result;
    std::string current;
    int char_count = 0;
    int total_chars = 0;
    
    // First pass: count total characters (for ratio calculation)
    for (size_t i = 0; i < text.length(); ) {
        unsigned char c = text[i];
        if (c < 0x80) { total_chars++; i++; }
        else if ((c & 0xE0) == 0xC0) { total_chars++; i += 2; }
        else if ((c & 0xF0) == 0xE0) { total_chars++; i += 3; }
        else if ((c & 0xF8) == 0xF0) { total_chars++; i += 4; }
        else { i++; }
    }
    
    if (total_chars == 0) return result;
    
    int processed_chars = 0;
    
    // Second pass: split by punctuation
    for (size_t i = 0; i < text.length(); ) {
        unsigned char c = text[i];
        uint32_t codepoint = 0;
        int bytes = 1;
        
        if (c < 0x80) {
            codepoint = c;
            bytes = 1;
        } else if ((c & 0xE0) == 0xC0) {
            codepoint = c & 0x1F;
            bytes = 2;
            for (int k = 1; k < bytes && i + k < text.length(); k++)
                codepoint = (codepoint << 6) | (text[i + k] & 0x3F);
        } else if ((c & 0xF0) == 0xE0) {
            codepoint = c & 0x0F;
            bytes = 3;
            for (int k = 1; k < bytes && i + k < text.length(); k++)
                codepoint = (codepoint << 6) | (text[i + k] & 0x3F);
        } else if ((c & 0xF8) == 0xF0) {
            codepoint = c & 0x07;
            bytes = 4;
            for (int k = 1; k < bytes && i + k < text.length(); k++)
                codepoint = (codepoint << 6) | (text[i + k] & 0x3F);
        }
        
        current.append(text.substr(i, bytes));
        char_count++;
        processed_chars++;
        i += bytes;
        
        // Check if sentence end
        if (is_sentence_end(codepoint) && !current.empty()) {
            float ratio = (float)(processed_chars - char_count / 2) / total_chars;
            result.push_back({ratio, current});
            current.clear();
            char_count = 0;
        }
    }
    
    // Add remaining text
    if (!current.empty()) {
        float ratio = (float)(processed_chars - char_count / 2) / total_chars;
        result.push_back({ratio, current});
    }
    
    return result;
}

struct sense_voice_addon_result {
    std::string full_text;
    std::string language;
    std::string emotion;
    std::string event;
    struct token_data {
        int id;
        int64_t t0;  // start time (ms)
        int64_t t1;  // end time (ms)
        std::string text;
    };
    struct segment_data {
        int64_t start_ms;
        int64_t end_ms;
        std::string text;
        std::string language;
        std::string emotion;
        std::string event;
        std::vector<token_data> tokens;
    };
    std::vector<segment_data> segments;
};

// Free SenseVoice context (application-level cleanup, same as main.cc)
void sense_voice_free(struct sense_voice_context * ctx) {
    if (ctx) {
        ggml_free(ctx->model.ctx);
        //ggml_free(ctx->vad_model.ctx); // not used.
        ggml_backend_buffer_free(ctx->model.buffer);
        ggml_backend_buffer_free(ctx->vad_model.buffer);

        sense_voice_free_state(ctx->state);

        delete ctx->model.model->encoder;
        delete ctx->model.model;
        delete ctx->vad_model.model;
        delete ctx;
    }
}

// ============================================================================
// SenseVoice ASR Worker - Regular Mode (non-stream)
// ============================================================================
// Non-stream mode VAD parameters (using whisper external VAD model):
//   --vad_model                    External VAD model path (ggml-silero.bin)
//   --min_speech_duration_ms      [250  ] VAD parameter, minimum speech length (ms)
//   --max_speech_duration_ms      [15000] VAD parameter, maximum speech length (ms)
//   --min_silence_duration_ms     [100  ] VAD parameter, minimum silence length
//
// VAD logic uses whisper.cpp's VAD API and external Silero model files.
// ============================================================================

class SenseVoiceWorker : public Napi::AsyncWorker {
public:
    // Abort flags - using shared_ptr to ensure they survive after worker destruction
    std::shared_ptr<std::atomic<bool>> m_should_abort;
    std::shared_ptr<std::atomic<bool>> m_was_aborted;

    SenseVoiceWorker(Napi::Function& callback, 
                     std::string model_path,
                     std::string vad_model_path,
                     std::string language,
                     std::vector<double> pcmf32,
                     bool use_gpu,
                     bool flash_attn,
                     int n_threads,
                     bool use_itn,
                     bool use_prefix,
                     float vad_threshold,
                     int min_speech_duration_ms,
                     int max_speech_duration_ms,
                     int min_silence_duration_ms,
                     int speech_pad_ms,
                     bool use_beam_search,
                     int beam_size,
                     bool debug,
                     Napi::Function progress_callback,
                     Napi::Env env)
        : Napi::AsyncWorker(callback),
          m_model_path(model_path),
          m_vad_model_path(vad_model_path), 
          m_language(language),
          m_pcmf32(std::move(pcmf32)),
          m_use_gpu(use_gpu),
          m_flash_attn(flash_attn),
          m_n_threads(n_threads),
          m_use_itn(use_itn),
          m_use_prefix(use_prefix),
          m_vad_threshold(vad_threshold),
          m_min_speech_duration_ms(min_speech_duration_ms),
          m_max_speech_duration_ms(max_speech_duration_ms),
          m_min_silence_duration_ms(min_silence_duration_ms),
          m_speech_pad_ms(speech_pad_ms),
          m_use_beam_search(use_beam_search),
          m_beam_size(beam_size),
          m_debug(debug),
          m_should_abort(std::make_shared<std::atomic<bool>>(false)),
          m_was_aborted(std::make_shared<std::atomic<bool>>(false)) {
        if (!progress_callback.IsEmpty()) {
            m_tsfn = Napi::ThreadSafeFunction::New(
                env,
                progress_callback,
                "SenseVoice Progress Callback",
                0,
                1
            );
        }
    }

    ~SenseVoiceWorker() {
        if (m_tsfn) {
            m_tsfn.Release();
        }
    }

    void OnProgress(int progress) {
        if (m_tsfn && !m_should_abort->load()) {
            auto abort_flag = m_should_abort;
            auto callback = [abort_flag, progress](Napi::Env env, Napi::Function jsCallback) {
                try {
                    Napi::Value result = jsCallback.Call({Napi::Number::New(env, progress)});
                    // If callback returns false, signal abort
                    if (result.IsBoolean() && !result.As<Napi::Boolean>().Value()) {
                        abort_flag->store(true);
                    }
                } catch (...) {
                    // Continue on error
                }
            };
            m_tsfn.BlockingCall(callback);
        }
    }

    bool ShouldAbort() const {
        return m_should_abort->load();
    }

    void Execute() override {
        // Initialize SenseVoice context
        struct sense_voice_context_params cparams = sense_voice_context_default_params();
        cparams.use_gpu = m_use_gpu;
        cparams.use_itn = m_use_itn;
        cparams.flash_attn = m_flash_attn;

        struct sense_voice_context* ctx = sense_voice_small_init_from_file_with_params(
            m_model_path.c_str(), cparams);

        if (ctx == nullptr) {
            SetError("Failed to initialize SenseVoice context");
            return;
        }

        // Set language
        int lang_id = sense_voice_lang_id(m_language.c_str());
        if (lang_id == -1) {
            lang_id = 0; // auto
        }
        ctx->language_id = lang_id;

        // Set up full params with beam search strategy
        sense_voice_decoding_strategy strategy = m_use_beam_search ? 
            SENSE_VOICE_SAMPLING_BEAM_SEARCH : SENSE_VOICE_SAMPLING_GREEDY;
        sense_voice_full_params wparams = sense_voice_full_default_params(strategy);
        wparams.language = m_language.c_str();
        wparams.n_threads = m_n_threads;
        wparams.debug_mode = false;
        wparams.beam_search.beam_size = m_beam_size;

        const int sample_rate = SENSE_VOICE_SAMPLE_RATE;
        std::string full_text;
        int64_t n_samples = m_pcmf32.size();

        // Convert double audio to float for whisper VAD API
        std::vector<float> pcmf32_float(m_pcmf32.begin(), m_pcmf32.end());

        // ====================================================================
        // Use whisper external VAD model for speech detection
        // ====================================================================
        if (!m_vad_model_path.empty()) {
            // Initialize whisper VAD context with external model
            whisper_vad_context_params vad_ctx_params = whisper_vad_default_context_params();
            vad_ctx_params.n_threads = m_n_threads;
            vad_ctx_params.use_gpu = false;  // Use CPU for VAD to avoid CUDA issues
            
            whisper_vad_context* vctx = whisper_vad_init_from_file_with_params(
                m_vad_model_path.c_str(), vad_ctx_params);
            
            if (vctx == nullptr) {
                SetError("Failed to initialize whisper VAD context");
                sense_voice_free(ctx);
                return;
            }

            // Configure VAD parameters
            whisper_vad_params vad_params = whisper_vad_default_params();
            vad_params.threshold = m_vad_threshold;
            vad_params.min_speech_duration_ms = m_min_speech_duration_ms;
            vad_params.min_silence_duration_ms = m_min_silence_duration_ms;
            vad_params.max_speech_duration_s = m_max_speech_duration_ms / 1000.0f;
            vad_params.speech_pad_ms = m_speech_pad_ms;

            // Detect speech segments using whisper VAD
            whisper_vad_segments* segments = whisper_vad_segments_from_samples(
                vctx, vad_params, pcmf32_float.data(), pcmf32_float.size());

            if (segments != nullptr) {
                int n_segments = whisper_vad_segments_n_segments(segments);
                fprintf(stderr, "[SENSE-VAD] Detected %d segments from %zu samples\n", n_segments, pcmf32_float.size());
                
                // Calculate max samples per chunk based on max_speech_duration_ms
                const int64_t max_samples_per_chunk = (static_cast<int64_t>(m_max_speech_duration_ms) * sample_rate) / 1000;
                
                for (int i = 0; i < n_segments; i++) {
                    // Check for abort before processing segment
                    if (ShouldAbort()) {
                        m_was_aborted->store(true);
                        whisper_vad_free_segments(segments);
                        whisper_vad_free(vctx);
                        sense_voice_free(ctx);
                        return;
                    }
                    // Note: whisper_vad_segments_get_segment_t0/t1 returns centiseconds (values * 100)
                    // We need to divide by 100 to get actual seconds
                    float t0 = whisper_vad_segments_get_segment_t0(segments, i) / 100.0f;
                    float t1 = whisper_vad_segments_get_segment_t1(segments, i) / 100.0f;
                    fprintf(stderr, "[SENSE-VAD] Segment %d: %.2fs - %.2fs (duration: %.2fs)\n", i, t0, t1, t1 - t0);
                    
                    // Convert time to sample indices
                    int64_t seg_start_sample = static_cast<int64_t>(t0 * sample_rate);
                    int64_t seg_end_sample = static_cast<int64_t>(t1 * sample_rate);
                    
                    // Clamp to valid range
                    if (seg_start_sample < 0) seg_start_sample = 0;
                    if (seg_end_sample > n_samples) seg_end_sample = n_samples;
                    if (seg_start_sample >= seg_end_sample) continue;
                    
                    // Split segment into smaller chunks if too long
                    int64_t chunk_start = seg_start_sample;
                    while (chunk_start < seg_end_sample) {
                        int64_t chunk_end = std::min(chunk_start + max_samples_per_chunk, seg_end_sample);
                        
                        float chunk_t0 = static_cast<float>(chunk_start) / sample_rate;
                        float chunk_t1 = static_cast<float>(chunk_end) / sample_rate;
                        
                        fprintf(stderr, "[SENSE-VAD] Processing chunk: %.2fs - %.2fs (%lld samples)\n", 
                                chunk_t0, chunk_t1, (long long)(chunk_end - chunk_start));
                        
                        // Extract speech segment and scale by 32768 (same as streaming mode)
                        std::vector<double> speech_segment;
                        speech_segment.reserve(chunk_end - chunk_start);
                        for (int64_t j = chunk_start; j < chunk_end; j++) {
                            speech_segment.push_back(m_pcmf32[j] * 32768.0);
                        }
                        
                        // Process segment with SenseVoice
                        if (sense_voice_full_parallel(ctx, wparams, speech_segment, speech_segment.size(), 1) == 0) {
                            // Extract result with token timestamps
                            std::string segment_text;
                            std::string seg_language, seg_emotion, seg_event;
                            std::vector<sense_voice_addon_result::token_data> token_list;
                            
                            int64_t seg_start_ms = static_cast<int64_t>(chunk_t0 * 1000);
                            int64_t seg_end_ms = static_cast<int64_t>(chunk_t1 * 1000);
                            int64_t seg_duration_ms = seg_end_ms - seg_start_ms;

                            if (ctx->state->ids.size() > 4) {
                                auto it_lang = ctx->vocab.id_to_token.find(ctx->state->ids[0]);
                                if (it_lang != ctx->vocab.id_to_token.end()) seg_language = it_lang->second;
                                auto it_emo = ctx->vocab.id_to_token.find(ctx->state->ids[1]);
                                if (it_emo != ctx->vocab.id_to_token.end()) seg_emotion = it_emo->second;
                                auto it_evt = ctx->vocab.id_to_token.find(ctx->state->ids[2]);
                                if (it_evt != ctx->vocab.id_to_token.end()) seg_event = it_evt->second;

                                // Count valid tokens for timestamp estimation
                                size_t n_valid_tokens = 0;
                                for (size_t k = 4; k < ctx->state->ids.size(); k++) {
                                    int id = ctx->state->ids[k];
                                    if (id != 0 && id != 1 && id != 2) n_valid_tokens++;
                                }

                                // Extract tokens with estimated timestamps
                                size_t token_idx = 0;
                                for (size_t k = 4; k < ctx->state->ids.size(); k++) {
                                    int id = ctx->state->ids[k];
                                    if (id != 0 && id != 1 && id != 2) {
                                        auto it = ctx->vocab.id_to_token.find(id);
                                        if (it != ctx->vocab.id_to_token.end()) {
                                            // Estimate token timestamps (uniform distribution)
                                            int64_t token_t0 = seg_start_ms + (seg_duration_ms * token_idx) / n_valid_tokens;
                                            int64_t token_t1 = seg_start_ms + (seg_duration_ms * (token_idx + 1)) / n_valid_tokens;
                                            
                                            sense_voice_addon_result::token_data tok;
                                            tok.id = id;
                                            tok.t0 = token_t0;
                                            tok.t1 = token_t1;
                                            tok.text = it->second;
                                            token_list.push_back(tok);
                                            
                                            segment_text += it->second;
                                            token_idx++;
                                        }
                                    }
                                }
                            }

                            // If use_itn is enabled, split by punctuation into sub-segments
                            if (m_use_itn && !token_list.empty()) {
                                std::vector<sense_voice_addon_result::token_data> current_tokens;
                                std::string current_text;
                                int64_t current_start = token_list[0].t0;
                                
                                for (size_t tidx = 0; tidx < token_list.size(); tidx++) {
                                    const auto& tok = token_list[tidx];
                                    current_tokens.push_back(tok);
                                    current_text += tok.text;
                                    
                                    // Check if this token ends with punctuation
                                    bool is_end = (tidx == token_list.size() - 1);
                                    bool has_punct = false;
                                    if (!tok.text.empty()) {
                                        for (size_t p = 0; p < tok.text.length(); p++) {
                                            if (is_sentence_ending_punctuation(tok.text, p)) {
                                                has_punct = true;
                                                break;
                                            }
                                        }
                                    }
                                    
                                    if (has_punct || is_end) {
                                        // Create sub-segment
                                        sense_voice_addon_result::segment_data seg;
                                        seg.start_ms = current_start;
                                        seg.end_ms = tok.t1;
                                        seg.text = current_text;
                                        seg.language = seg_language;
                                        seg.emotion = seg_emotion;
                                        seg.event = seg_event;
                                        seg.tokens = current_tokens;
                                        m_result.segments.push_back(seg);
                                        
                                        // Reset for next sub-segment
                                        current_tokens.clear();
                                        current_text.clear();
                                        if (tidx + 1 < token_list.size()) {
                                            current_start = token_list[tidx + 1].t0;
                                        }
                                    }
                                }
                            } else {
                                // No punctuation splitting - create single segment
                                sense_voice_addon_result::segment_data seg;
                                seg.start_ms = seg_start_ms;
                                seg.end_ms = seg_end_ms;
                                seg.text = segment_text;
                                seg.language = seg_language;
                                seg.emotion = seg_emotion;
                                seg.event = seg_event;
                                seg.tokens = token_list;
                                m_result.segments.push_back(seg);
                            }

                            m_result.full_text += segment_text;
                        }
                        
                        chunk_start = chunk_end;
                        
                        // Report progress and check for abort
                        int progress = static_cast<int>((chunk_end * 100) / n_samples);
                        OnProgress(progress);
                        if (ShouldAbort()) {
                            m_was_aborted->store(true);
                            whisper_vad_free_segments(segments);
                            whisper_vad_free(vctx);
                            sense_voice_free(ctx);
                            return;
                        }
                    }
                }

                whisper_vad_free_segments(segments);
            }

            whisper_vad_free(vctx);
        } else {
            // No VAD model - process entire audio as one segment
            // Scale audio by 32768 (same as streaming mode)
            std::vector<double> scaled_audio;
            scaled_audio.reserve(m_pcmf32.size());
            for (size_t i = 0; i < m_pcmf32.size(); i++) {
                scaled_audio.push_back(m_pcmf32[i] * 32768.0);
            }
            
            if (sense_voice_full_parallel(ctx, wparams, scaled_audio, scaled_audio.size(), 1) == 0) {
                std::string segment_text;
                std::string seg_language, seg_emotion, seg_event;
                std::vector<sense_voice_addon_result::token_data> token_list;
                
                int64_t seg_start_ms = 0;
                int64_t seg_end_ms = (n_samples * 1000) / sample_rate;
                int64_t seg_duration_ms = seg_end_ms - seg_start_ms;

                if (ctx->state->ids.size() > 4) {
                    auto it_lang = ctx->vocab.id_to_token.find(ctx->state->ids[0]);
                    if (it_lang != ctx->vocab.id_to_token.end()) seg_language = it_lang->second;
                    auto it_emo = ctx->vocab.id_to_token.find(ctx->state->ids[1]);
                    if (it_emo != ctx->vocab.id_to_token.end()) seg_emotion = it_emo->second;
                    auto it_evt = ctx->vocab.id_to_token.find(ctx->state->ids[2]);
                    if (it_evt != ctx->vocab.id_to_token.end()) seg_event = it_evt->second;

                    // Count valid tokens for timestamp estimation
                    size_t n_valid_tokens = 0;
                    for (size_t k = 4; k < ctx->state->ids.size(); k++) {
                        int id = ctx->state->ids[k];
                        if (id != 0 && id != 1 && id != 2) n_valid_tokens++;
                    }

                    // Extract tokens with estimated timestamps
                    size_t token_idx = 0;
                    for (size_t k = 4; k < ctx->state->ids.size(); k++) {
                        int id = ctx->state->ids[k];
                        if (id != 0 && id != 1 && id != 2) {
                            auto it = ctx->vocab.id_to_token.find(id);
                            if (it != ctx->vocab.id_to_token.end()) {
                                int64_t token_t0 = seg_start_ms + (seg_duration_ms * token_idx) / n_valid_tokens;
                                int64_t token_t1 = seg_start_ms + (seg_duration_ms * (token_idx + 1)) / n_valid_tokens;
                                
                                sense_voice_addon_result::token_data tok;
                                tok.id = id;
                                tok.t0 = token_t0;
                                tok.t1 = token_t1;
                                tok.text = it->second;
                                token_list.push_back(tok);
                                
                                segment_text += it->second;
                                token_idx++;
                            }
                        }
                    }
                }

                // If use_itn is enabled, split by punctuation into sub-segments
                if (m_use_itn && !token_list.empty()) {
                    std::vector<sense_voice_addon_result::token_data> current_tokens;
                    std::string current_text;
                    int64_t current_start = token_list[0].t0;
                    
                    for (size_t tidx = 0; tidx < token_list.size(); tidx++) {
                        const auto& tok = token_list[tidx];
                        current_tokens.push_back(tok);
                        current_text += tok.text;
                        
                        bool is_end = (tidx == token_list.size() - 1);
                        bool has_punct = false;
                        if (!tok.text.empty()) {
                            for (size_t p = 0; p < tok.text.length(); p++) {
                                if (is_sentence_ending_punctuation(tok.text, p)) {
                                    has_punct = true;
                                    break;
                                }
                            }
                        }
                        
                        if (has_punct || is_end) {
                            sense_voice_addon_result::segment_data seg;
                            seg.start_ms = current_start;
                            seg.end_ms = tok.t1;
                            seg.text = current_text;
                            seg.language = seg_language;
                            seg.emotion = seg_emotion;
                            seg.event = seg_event;
                            seg.tokens = current_tokens;
                            m_result.segments.push_back(seg);
                            
                            current_tokens.clear();
                            current_text.clear();
                            if (tidx + 1 < token_list.size()) {
                                current_start = token_list[tidx + 1].t0;
                            }
                        }
                    }
                } else {
                    sense_voice_addon_result::segment_data seg;
                    seg.start_ms = seg_start_ms;
                    seg.end_ms = seg_end_ms;
                    seg.text = segment_text;
                    seg.language = seg_language;
                    seg.emotion = seg_emotion;
                    seg.event = seg_event;
                    seg.tokens = token_list;
                    m_result.segments.push_back(seg);
                }

                m_result.full_text = segment_text;
                
                // Report 100% progress for non-VAD path
                OnProgress(100);
            }
        }

        // Set result metadata from first segment if available
        if (!m_result.segments.empty()) {
            m_result.language = m_result.segments[0].language;
            m_result.emotion = m_result.segments[0].emotion;
            m_result.event = m_result.segments[0].event;
        }

        sense_voice_free(ctx);
    }



    void OnOK() override {
        Napi::HandleScope scope(Env());
        Napi::Object result = Napi::Object::New(Env());
        result.Set("text", Napi::String::New(Env(), m_result.full_text));
        result.Set("language", Napi::String::New(Env(), m_result.language));
        result.Set("emotion", Napi::String::New(Env(), m_result.emotion));
        result.Set("event", Napi::String::New(Env(), m_result.event));

        // Create segments array
        Napi::Array segments = Napi::Array::New(Env(), m_result.segments.size());
        for (size_t i = 0; i < m_result.segments.size(); i++) {
            Napi::Object seg = Napi::Object::New(Env());
            seg.Set("start", Napi::Number::New(Env(), m_result.segments[i].start_ms));
            seg.Set("end", Napi::Number::New(Env(), m_result.segments[i].end_ms));
            seg.Set("text", Napi::String::New(Env(), m_result.segments[i].text));
            seg.Set("language", Napi::String::New(Env(), m_result.segments[i].language));
            
            // Only output emotion and event if m_use_prefix is true
            if (m_use_prefix) {
                seg.Set("emotion", Napi::String::New(Env(), m_result.segments[i].emotion));
                seg.Set("event", Napi::String::New(Env(), m_result.segments[i].event));
            }
            
            // Add token timestamps if available
            const auto& tokenList = m_result.segments[i].tokens;
            Napi::Array tokens = Napi::Array::New(Env(), tokenList.size());
            for (size_t j = 0; j < tokenList.size(); j++) {
                Napi::Object token = Napi::Object::New(Env());
                token.Set("id", Napi::Number::New(Env(), tokenList[j].id));
                token.Set("t0", Napi::Number::New(Env(), tokenList[j].t0));
                token.Set("t1", Napi::Number::New(Env(), tokenList[j].t1));
                token.Set("text", Napi::String::New(Env(), tokenList[j].text));
                tokens[j] = token;
            }
            seg.Set("tokens", tokens);
            
            segments[i] = seg;
        }
        result.Set("segments", segments);
        result.Set("aborted", Napi::Boolean::New(Env(), m_was_aborted->load()));

        Callback().Call({Env().Null(), result});
    }

private:
    std::string m_model_path;
    std::string m_vad_model_path;
    std::string m_language;
    std::vector<double> m_pcmf32;
    bool m_use_gpu;
    bool m_flash_attn;
    int m_n_threads;
    bool m_use_itn;
    bool m_use_prefix;
    float m_vad_threshold;
    int m_min_speech_duration_ms;
    int m_max_speech_duration_ms;
    int m_min_silence_duration_ms;
    int m_speech_pad_ms;
    bool m_use_beam_search;
    int m_beam_size;
    bool m_debug = false;  // Debug logging flag
    sense_voice_addon_result m_result;
    Napi::ThreadSafeFunction m_tsfn;  // Thread-safe function for progress callback
};

Napi::Value senseVoice(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2 || !info[0].IsObject() || !info[1].IsFunction()) {
        Napi::TypeError::New(env, "Usage: senseVoice(options, callback)").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Object options = info[0].As<Napi::Object>();

    // Required: model path
    if (!options.Has("model") || !options.Get("model").IsString()) {
        Napi::TypeError::New(env, "options.model (string) is required").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    std::string model_path = options.Get("model").As<Napi::String>();

    // Audio input: either pcmf32 (Float32Array) or file (string path to WAV file)
    std::vector<double> pcmf32;
    
    if (options.Has("file") && options.Get("file").IsString()) {
        // Load audio from WAV file
        std::string file_path = options.Get("file").As<Napi::String>();
        int32_t sample_rate = 0;
        if (!load_wav_file(file_path.c_str(), &sample_rate, pcmf32)) {
            Napi::TypeError::New(env, "Failed to load WAV file: " + file_path).ThrowAsJavaScriptException();
            return env.Undefined();
        }
        if (sample_rate != 16000) {
            Napi::TypeError::New(env, "WAV file must be 16kHz sample rate, got: " + std::to_string(sample_rate)).ThrowAsJavaScriptException();
            return env.Undefined();
        }
    } else if (options.Has("pcmf32") && options.Get("pcmf32").IsTypedArray()) {
        // Convert Float32Array to std::vector<double>
        // Note: Input is expected to be normalized [-1, 1], scaling is done in FBANK
        Napi::Float32Array pcmf32_arr = options.Get("pcmf32").As<Napi::Float32Array>();
        pcmf32.reserve(pcmf32_arr.ElementLength());
        for (size_t i = 0; i < pcmf32_arr.ElementLength(); i++) {
            pcmf32.push_back(static_cast<double>(pcmf32_arr[i]));
        }
    } else {
        Napi::TypeError::New(env, "Either options.file (WAV path) or options.pcmf32 (Float32Array) is required").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Optional parameters with defaults
    std::string language = "auto";
    if (options.Has("language") && options.Get("language").IsString()) {
        language = options.Get("language").As<Napi::String>();
    }

    bool use_gpu = true;
    if (options.Has("use_gpu") && options.Get("use_gpu").IsBoolean()) {
        use_gpu = options.Get("use_gpu").As<Napi::Boolean>();
    }

    bool flash_attn = false;
    if (options.Has("flash_attn") && options.Get("flash_attn").IsBoolean()) {
        flash_attn = options.Get("flash_attn").As<Napi::Boolean>();
    }

    int n_threads = std::min(4, static_cast<int>(std::thread::hardware_concurrency()));
    if (options.Has("n_threads") && options.Get("n_threads").IsNumber()) {
        n_threads = options.Get("n_threads").As<Napi::Number>().Int32Value();
    }

    bool use_itn = false;
    if (options.Has("use_itn") && options.Get("use_itn").IsBoolean()) {
        use_itn = options.Get("use_itn").As<Napi::Boolean>();
    }

    bool use_prefix = false;
    if (options.Has("use_prefix") && options.Get("use_prefix").IsBoolean()) {
        use_prefix = options.Get("use_prefix").As<Napi::Boolean>();
    }

    // VAD parameters
    float vad_threshold = 0.5f;
    if (options.Has("vad_threshold") && options.Get("vad_threshold").IsNumber()) {
        vad_threshold = options.Get("vad_threshold").As<Napi::Number>().FloatValue();
    }

    // VAD model path (external Silero VAD model)
    std::string vad_model_path = "";
    if (options.Has("vad_model") && options.Get("vad_model").IsString()) {
        vad_model_path = options.Get("vad_model").As<Napi::String>();
    }

    int min_speech_duration_ms = 250;
    if (options.Has("min_speech_duration_ms") && options.Get("min_speech_duration_ms").IsNumber()) {
        min_speech_duration_ms = options.Get("min_speech_duration_ms").As<Napi::Number>().Int32Value();
    }

    int max_speech_duration_ms = 15000;
    if (options.Has("max_speech_duration_ms") && options.Get("max_speech_duration_ms").IsNumber()) {
        max_speech_duration_ms = options.Get("max_speech_duration_ms").As<Napi::Number>().Int32Value();
    }

    int min_silence_duration_ms = 100;
    if (options.Has("min_silence_duration_ms") && options.Get("min_silence_duration_ms").IsNumber()) {
        min_silence_duration_ms = options.Get("min_silence_duration_ms").As<Napi::Number>().Int32Value();
    }

    int speech_pad_ms = 30;
    if (options.Has("speech_pad_ms") && options.Get("speech_pad_ms").IsNumber()) {
        speech_pad_ms = options.Get("speech_pad_ms").As<Napi::Number>().Int32Value();
    }

    // CTC Beam Search parameters
    bool use_beam_search = false;
    if (options.Has("use_beam_search") && options.Get("use_beam_search").IsBoolean()) {
        use_beam_search = options.Get("use_beam_search").As<Napi::Boolean>();
    }

    int beam_size = 3;  // Default beam size
    if (options.Has("beam_size") && options.Get("beam_size").IsNumber()) {
        beam_size = options.Get("beam_size").As<Napi::Number>().Int32Value();
    }
    // Ensure greedy decoding when use_beam_search is false
    if (!use_beam_search) {
        beam_size = 1;
    }

    // Debug logging flag
    bool debug = false;
    if (options.Has("debug") && options.Get("debug").IsBoolean()) {
        debug = options.Get("debug").As<Napi::Boolean>();
    }

    // Progress callback (optional)
    Napi::Function progress_callback;
    if (options.Has("progress_callback") && options.Get("progress_callback").IsFunction()) {
        progress_callback = options.Get("progress_callback").As<Napi::Function>();
    }

    Napi::Function callback = info[1].As<Napi::Function>();
    SenseVoiceWorker* worker = new SenseVoiceWorker(
        callback, model_path, vad_model_path, language, std::move(pcmf32),
        use_gpu, flash_attn, n_threads, use_itn, use_prefix,
        vad_threshold,
        min_speech_duration_ms, max_speech_duration_ms, min_silence_duration_ms, speech_pad_ms,
        use_beam_search, beam_size, debug, progress_callback, env);
    worker->Queue();

    return env.Undefined();
}

// ============================================================================
// SenseVoiceStream Class - Streaming ASR with Segment Timestamps
// ============================================================================

class SenseVoiceStream : public Napi::ObjectWrap<SenseVoiceStream> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    SenseVoiceStream(const Napi::CallbackInfo& info);
    ~SenseVoiceStream();

private:
    // N-API Methods
    Napi::Value Start(const Napi::CallbackInfo& info);
    Napi::Value AddAudio(const Napi::CallbackInfo& info);
    Napi::Value Stop(const Napi::CallbackInfo& info);
    Napi::Value Finish(const Napi::CallbackInfo& info);
    Napi::Value Pause(const Napi::CallbackInfo& info);
    Napi::Value Resume(const Napi::CallbackInfo& info);

    // Worker thread
    void StreamWorker();
    void processAndOutput(sense_voice_full_params& wparams, std::vector<double>& audio, int64_t start_time_ms, int64_t end_time_ms);
    std::string ExtractText(sense_voice_context* ctx, bool use_prefix);

    // SenseVoice context
    sense_voice_context* m_ctx = nullptr;

    // Configuration
    std::string m_model_path;
    std::string m_language;
    int m_n_threads = 4;
    bool m_use_gpu = true;
    bool m_flash_attn = false;
    bool m_use_itn = false;
    bool m_use_prefix = false;

    // ============================================================================
    // Streaming VAD parameters (chunk-based)
    // ============================================================================
    // Stream mode specific parameters:
    //   -mmc  / --min-mute-chunks    [10   ] Minimum number of mute chunks
    //   -mnc  / --max-nomute-chunks  [80   ] Maximum number of non-mute chunks
    //         --use-vad              [true ] Whether to use VAD
    // ============================================================================
    bool m_use_vad = true;          // Whether to enable VAD
    int m_chunk_size_ms = 100;      // VAD chunk size (ms)
    int m_min_mute_chunks = 10;     // Minimum number of mute chunks
    int m_max_nomute_chunks = 80;   // Maximum number of non-mute chunks
    float m_vad_threshold = 0.5f;   // Energy threshold

    // Audio buffer and state
    std::deque<float> m_audio_buffer;
    std::vector<double> m_pcmf32_local;
    int64_t m_n_samples_processed = 0;      // Samples sent to model (for internal use)
    int64_t m_n_total_samples_received = 0; // Total samples received (for timeline)

    std::thread m_worker_thread;
    std::atomic<StreamState> m_state;
    std::mutex m_mutex;
    std::condition_variable m_cv;

    Napi::ThreadSafeFunction m_tsfn_callback;
    
    // External Silero VAD (optional, for streaming)
    std::string m_vad_model_path;           // External VAD model path (empty for energy VAD)
    whisper_vad_context* m_vctx = nullptr;  // whisper VAD context
    
    // Debug logging flag
    bool m_debug = false;
};

Napi::Object SenseVoiceStream::Init(Napi::Env env, Napi::Object exports) {
    Napi::HandleScope scope(env);
    Napi::Function func = DefineClass(env, "SenseVoiceStream", {
        InstanceMethod("start", &SenseVoiceStream::Start),
        InstanceMethod("addAudio", &SenseVoiceStream::AddAudio),
        InstanceMethod("stop", &SenseVoiceStream::Stop),
        InstanceMethod("finish", &SenseVoiceStream::Finish),
        InstanceMethod("pause", &SenseVoiceStream::Pause),
        InstanceMethod("resume", &SenseVoiceStream::Resume),
    });
    exports.Set("SenseVoiceStream", func);
    return exports;
}

SenseVoiceStream::SenseVoiceStream(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<SenseVoiceStream>(info), m_state(StreamState::IDLE) {
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
    
    // Language
    if (options.Has("language") && options.Get("language").IsString()) 
        m_language = options.Get("language").As<Napi::String>();
    else 
        m_language = "auto";
    
    // Threads
    if (options.Has("n_threads") && options.Get("n_threads").IsNumber()) 
        m_n_threads = options.Get("n_threads").As<Napi::Number>();
    
    // GPU options
    if (options.Has("use_gpu") && options.Get("use_gpu").IsBoolean()) 
        m_use_gpu = options.Get("use_gpu").As<Napi::Boolean>();
    if (options.Has("flash_attn") && options.Get("flash_attn").IsBoolean()) 
        m_flash_attn = options.Get("flash_attn").As<Napi::Boolean>();
    
    // ITN and prefix
    if (options.Has("use_itn") && options.Get("use_itn").IsBoolean()) 
        m_use_itn = options.Get("use_itn").As<Napi::Boolean>();
    if (options.Has("use_prefix") && options.Get("use_prefix").IsBoolean()) 
        m_use_prefix = options.Get("use_prefix").As<Napi::Boolean>();
    
    // Streaming VAD parameters (chunk-based)
    if (options.Has("use_vad") && options.Get("use_vad").IsBoolean()) 
        m_use_vad = options.Get("use_vad").As<Napi::Boolean>();
    if (options.Has("chunk_size") && options.Get("chunk_size").IsNumber()) 
        m_chunk_size_ms = options.Get("chunk_size").As<Napi::Number>().Int32Value();
    if (options.Has("min_mute_chunks") && options.Get("min_mute_chunks").IsNumber()) 
        m_min_mute_chunks = options.Get("min_mute_chunks").As<Napi::Number>().Int32Value();
    if (options.Has("max_nomute_chunks") && options.Get("max_nomute_chunks").IsNumber()) 
        m_max_nomute_chunks = options.Get("max_nomute_chunks").As<Napi::Number>().Int32Value();
    if (options.Has("vad_threshold") && options.Get("vad_threshold").IsNumber()) 
        m_vad_threshold = options.Get("vad_threshold").As<Napi::Number>().FloatValue();
    
    // External Silero VAD model path (optional)
    if (options.Has("vad_model") && options.Get("vad_model").IsString()) 
        m_vad_model_path = options.Get("vad_model").As<Napi::String>();
    
    // Debug logging flag
    if (options.Has("debug") && options.Get("debug").IsBoolean()) 
        m_debug = options.Get("debug").As<Napi::Boolean>();
}

SenseVoiceStream::~SenseVoiceStream() {
    m_state = StreamState::STOPPING;
    m_cv.notify_one();
    if (m_worker_thread.joinable()) {
        m_worker_thread.join();
    }
    if (m_tsfn_callback) {
        m_tsfn_callback.Release();
    }
    if (m_ctx) {
        sense_voice_free(m_ctx);
        m_ctx = nullptr;
    }
    if (m_vctx) {
        whisper_vad_free(m_vctx);
        m_vctx = nullptr;
    }
}

Napi::Value SenseVoiceStream::Start(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (m_state.load() != StreamState::IDLE) {
        Napi::Error::New(env, "Stream is already running or stopping").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    
    if (info.Length() < 1 || !info[0].IsFunction()) {
        Napi::TypeError::New(env, "start() requires a callback function").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    
    // Initialize context
    struct sense_voice_context_params cparams = sense_voice_context_default_params();
    cparams.use_gpu = m_use_gpu;
    cparams.use_itn = m_use_itn;
    cparams.flash_attn = m_flash_attn;
    
    m_ctx = sense_voice_small_init_from_file_with_params(m_model_path.c_str(), cparams);
    if (!m_ctx) {
        Napi::Error::New(env, "Failed to initialize SenseVoice context").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    
    // Set language
    int lang_id = sense_voice_lang_id(m_language.c_str());
    if (lang_id == -1) lang_id = 0;
    m_ctx->language_id = lang_id;
    
    // Initialize external Silero VAD if model path provided
    if (!m_vad_model_path.empty()) {
        whisper_vad_context_params vad_ctx_params = whisper_vad_default_context_params();
        vad_ctx_params.n_threads = m_n_threads;
        vad_ctx_params.use_gpu = false;  // Use CPU for VAD to avoid CUDA issues
        
        m_vctx = whisper_vad_init_from_file_with_params(m_vad_model_path.c_str(), vad_ctx_params);
        if (!m_vctx) {
            fprintf(stderr, "[STREAM] Warning: Failed to load VAD model, falling back to energy VAD\n");
        } else {
            fprintf(stderr, "[STREAM] Using external Silero VAD model\n");
        }
    }
    
    // Create thread-safe function for callbacks
    Napi::Function callback = info[0].As<Napi::Function>();
    m_tsfn_callback = Napi::ThreadSafeFunction::New(env, callback, "SenseVoiceStreamCallback", 0, 1);
    
    m_state = StreamState::RUNNING;
    m_worker_thread = std::thread(&SenseVoiceStream::StreamWorker, this);
    
    return env.Undefined();
}

Napi::Value SenseVoiceStream::AddAudio(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    StreamState current = m_state.load();
    if (current != StreamState::RUNNING && current != StreamState::PAUSED) {
        return env.Undefined();
    }
    
    if (info.Length() < 1 || !info[0].IsTypedArray()) {
        Napi::TypeError::New(env, "addAudio() requires a Float32Array").ThrowAsJavaScriptException();
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

Napi::Value SenseVoiceStream::Stop(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    StreamState current = m_state.load();
    if (current == StreamState::RUNNING || current == StreamState::PAUSED || current == StreamState::FINISHING) {
        m_state = StreamState::STOPPING;
        m_cv.notify_one();
    }
    
    if (m_worker_thread.joinable()) {
        m_worker_thread.join();
    }
    
    if (m_ctx) {
        sense_voice_free(m_ctx);
        m_ctx = nullptr;
    }
    
    m_state = StreamState::IDLE;
    
    return env.Undefined();
}

Napi::Value SenseVoiceStream::Finish(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    StreamState current = m_state.load();
    if (current == StreamState::RUNNING || current == StreamState::PAUSED) {
        m_state = StreamState::FINISHING;
        m_cv.notify_one();
    }
    
    return env.Undefined();
}

Napi::Value SenseVoiceStream::Pause(const Napi::CallbackInfo& info) {
    if (m_state.load() == StreamState::RUNNING) {
        m_state = StreamState::PAUSED;
    }
    return info.Env().Undefined();
}

Napi::Value SenseVoiceStream::Resume(const Napi::CallbackInfo& info) {
    if (m_state.load() == StreamState::PAUSED) {
        m_state = StreamState::RUNNING;
        m_cv.notify_one();
    }
    return info.Env().Undefined();
}

std::string SenseVoiceStream::ExtractText(sense_voice_context* ctx, bool use_prefix) {
    std::string result_text;
    size_t start_idx = use_prefix ? 0 : 4;
    for (size_t i = start_idx; i < ctx->state->ids.size(); i++) {
        int id = ctx->state->ids[i];
        if (i > 0 && ctx->state->ids[i - 1] == ctx->state->ids[i]) {
            continue;
        }
        if (id != 0) {
            auto it = ctx->vocab.id_to_token.find(id);
            if (it != ctx->vocab.id_to_token.end()) {
                result_text += it->second;
            }
        }
    }
    return result_text;
}

void SenseVoiceStream::StreamWorker() {
    const int sample_rate = SENSE_VOICE_SAMPLE_RATE;
    
    // Chunk-based VAD parameters
    const int chunk_samples = (m_chunk_size_ms * sample_rate) / 1000;  // samples per chunk (512 for 32ms)
    const int max_speech_chunks = m_max_nomute_chunks;  // max chunks before processing
    const int min_silence_chunks = m_min_mute_chunks;   // min silent chunks to end speech
    // Note: m_vad_threshold is used with Silero VAD (speech probability comparison)
    
    sense_voice_full_params wparams = sense_voice_full_default_params(SENSE_VOICE_SAMPLING_GREEDY);
    wparams.language = m_language.c_str();
    wparams.n_threads = m_n_threads;
    wparams.debug_mode = false;
    
    m_pcmf32_local.clear();
    std::vector<double> speech_buffer;  // Only accumulate speech chunks
    bool in_speech = false;
    int silence_chunk_count = 0;
    int speech_chunk_count = 0;
    int64_t speech_start_sample = 0;  // Track when speech started (in total samples received)
    
    // Reset total samples counter for timeline
    m_n_total_samples_received = 0;
    
    // Accumulated samples buffer (persists across loop iterations)
    std::vector<double> accumulated_samples;
    
    while (true) {
        bool is_finishing = false;
        
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait(lock, [this] {
                StreamState s = m_state.load();
                // When PAUSED, wait until RUNNING/STOPPING/FINISHING
                return s == StreamState::STOPPING || 
                       s == StreamState::FINISHING || 
                       (s == StreamState::RUNNING && !m_audio_buffer.empty());
            });

            StreamState current_state = m_state.load();
            if (current_state == StreamState::STOPPING) {
                break;
            }
            if (current_state == StreamState::PAUSED) {
                continue; // Skip this iteration, wait again
            }
            is_finishing = (current_state == StreamState::FINISHING);
            
            // Append new samples to accumulated buffer (convert float to double)
            for (float sample : m_audio_buffer) {
                accumulated_samples.push_back(static_cast<double>(sample));
            }
            m_audio_buffer.clear();
        }

        // Process accumulated samples in chunks using Silero VAD
        size_t processed_samples = 0;
        for (size_t i = 0; i + chunk_samples <= accumulated_samples.size(); i += chunk_samples) {
            // Prepare float chunk for VAD
            std::vector<float> vad_chunk(chunk_samples);
            for (int j = 0; j < chunk_samples; j++) {
                vad_chunk[j] = static_cast<float>(accumulated_samples[i + j]);
            }
            
            // VAD detection (if enabled)
            bool is_speech;
            if (m_use_vad) {
                float speech_prob = 0.0f;
                
                // Check if external Silero VAD is available
                if (m_vctx) {
                    // Use Silero VAD - detect speech and get probability
                    // Note: we need to call whisper_vad_detect_speech which handles internal state
                    whisper_vad_detect_speech(m_vctx, vad_chunk.data(), vad_chunk.size());
                    // Get the last probability value
                    int n_probs = whisper_vad_n_probs(m_vctx);
                    if (n_probs > 0) {
                        speech_prob = whisper_vad_probs(m_vctx)[n_probs - 1];
                    }
                } else {
                    // Fallback: Simple energy-based VAD
                    float energy = 0.0f;
                    for (int j = 0; j < chunk_samples; j++) {
                        energy += vad_chunk[j] * vad_chunk[j];
                    }
                    energy = std::sqrt(energy / chunk_samples);
                    speech_prob = energy > 0.01f ? 1.0f : 0.0f;  // Simple threshold
                }
                is_speech = (speech_prob >= m_vad_threshold);
            } else {
                // VAD disabled - treat all audio as speech
                is_speech = true;
            }
            
            if (is_speech) {
                // Track when speech starts
                if (!in_speech) {
                    speech_start_sample = m_n_total_samples_received + static_cast<int64_t>(i);
                }
                // Append speech chunk
                for (int j = 0; j < chunk_samples; j++) {
                    speech_buffer.push_back(accumulated_samples[i + j] * 32768.0);  // Scale for model
                }
                in_speech = true;
                silence_chunk_count = 0;
                speech_chunk_count++;
            } else if (in_speech) {
                // We were in speech, count silence chunks
                silence_chunk_count++;
                
                // Include some silence for natural boundaries
                if (silence_chunk_count <= min_silence_chunks) {
                    for (int j = 0; j < chunk_samples; j++) {
                        speech_buffer.push_back(accumulated_samples[i + j] * 32768.0);
                    }
                }
                
                // End of speech detected
                if (silence_chunk_count > min_silence_chunks) {
                    in_speech = false;
                    
                    // Calculate timestamps
                    int64_t speech_end_sample = m_n_total_samples_received + static_cast<int64_t>(i) + chunk_samples;
                    int64_t start_time_ms = (speech_start_sample * 1000) / sample_rate;
                    int64_t end_time_ms = (speech_end_sample * 1000) / sample_rate;
                    
                    // Process accumulated speech
                    if (!speech_buffer.empty()) {
                        processAndOutput(wparams, speech_buffer, start_time_ms, end_time_ms);
                    }
                    speech_buffer.clear();
                    speech_chunk_count = 0;
                }
            }
            // Skip silent chunks when not in speech (don't accumulate)
            processed_samples = i + chunk_samples;
        }
        
        // Remove processed samples from accumulator, keep remaining
        if (processed_samples > 0) {
            // Update total samples received counter for timeline
            m_n_total_samples_received += processed_samples;
            accumulated_samples.erase(accumulated_samples.begin(), accumulated_samples.begin() + processed_samples);
        }
        
        // Force process if max chunks exceeded
        if (speech_chunk_count >= max_speech_chunks) {
            // Calculate timestamps for forced processing
            int64_t current_sample = m_n_total_samples_received;
            int64_t start_time_ms = (speech_start_sample * 1000) / sample_rate;
            int64_t end_time_ms = (current_sample * 1000) / sample_rate;
            
            processAndOutput(wparams, speech_buffer, start_time_ms, end_time_ms);
            speech_buffer.clear();
            speech_chunk_count = 0;
            speech_start_sample = current_sample;  // Reset for next speech segment
            in_speech = false;
        }
        
        if (is_finishing) {
            // Add any remaining accumulated samples to speech buffer
            for (size_t i = 0; i < accumulated_samples.size(); i++) {
                speech_buffer.push_back(accumulated_samples[i] * 32768.0);
            }
            // Calculate final timestamps
            int64_t final_sample = m_n_total_samples_received + static_cast<int64_t>(accumulated_samples.size());
            int64_t start_time_ms = in_speech ? (speech_start_sample * 1000) / sample_rate : (m_n_total_samples_received * 1000) / sample_rate;
            int64_t end_time_ms = (final_sample * 1000) / sample_rate;
            
            // Process any remaining speech
            if (!speech_buffer.empty()) {
                processAndOutput(wparams, speech_buffer, start_time_ms, end_time_ms);
            }
            break;
        }
    }

    // Send 'end' signal
    if (m_tsfn_callback) {
        m_tsfn_callback.BlockingCall([](Napi::Env env, Napi::Function jsCallback) {
            Napi::Object result = Napi::Object::New(env);
            result.Set("type", "end");
            jsCallback.Call({env.Null(), result});
        });
    }
}

void SenseVoiceStream::processAndOutput(sense_voice_full_params& wparams, std::vector<double>& audio, int64_t start_time_ms, int64_t end_time_ms) {
    const int sample_rate = SENSE_VOICE_SAMPLE_RATE;
    
    // Safety check
    if (!m_ctx || audio.empty()) {
        return;
    }
    
    try {
        // Use sense_voice_full_parallel for streaming
        if (sense_voice_full_parallel(m_ctx, wparams, audio, audio.size(), m_n_threads) == 0) {
            // Extract text and prefix info directly from state (API functions not available in this version)
            std::string text_str;
            std::string lang_str, emo_str, evt_str;
            
            if (m_ctx->state->ids.size() > 4) {
                // Extract prefix tokens
                auto it_lang = m_ctx->vocab.id_to_token.find(m_ctx->state->ids[0]);
                if (it_lang != m_ctx->vocab.id_to_token.end()) lang_str = it_lang->second;
                auto it_emo = m_ctx->vocab.id_to_token.find(m_ctx->state->ids[1]);
                if (it_emo != m_ctx->vocab.id_to_token.end()) emo_str = it_emo->second;
                auto it_evt = m_ctx->vocab.id_to_token.find(m_ctx->state->ids[2]);
                if (it_evt != m_ctx->vocab.id_to_token.end()) evt_str = it_evt->second;
                
                // Extract text tokens (skip first 4 prefix tokens, deduplicate)
                int prev_id = -1;
                for (size_t i = 4; i < m_ctx->state->ids.size(); i++) {
                    int id = m_ctx->state->ids[i];
                    if (id != 0 && id != prev_id) {
                        auto it = m_ctx->vocab.id_to_token.find(id);
                        if (it != m_ctx->vocab.id_to_token.end()) {
                            text_str += it->second;
                        }
                    }
                    prev_id = id;
                }
            }
            
            // Compute token timestamps using passed time range
            std::vector<std::tuple<int64_t, int64_t, std::string>> token_data;
            int64_t segment_duration_ms = end_time_ms - start_time_ms;
            
            if (m_ctx->state->ids.size() > 4) {
                // Count valid tokens for uniform timestamp distribution
                size_t n_valid_tokens = 0;
                int prev_id = -1;
                for (size_t i = 4; i < m_ctx->state->ids.size(); i++) {
                    int id = m_ctx->state->ids[i];
                    if (id != 0 && id != prev_id) {
                        n_valid_tokens++;
                    }
                    prev_id = id;
                }
                
                // Extract tokens with estimated timestamps (uniform distribution)
                size_t token_idx = 0;
                prev_id = -1;
                for (size_t i = 4; i < m_ctx->state->ids.size(); i++) {
                    int id = m_ctx->state->ids[i];
                    if (id != 0 && id != prev_id) {
                        int64_t t0 = start_time_ms + (segment_duration_ms * token_idx) / (n_valid_tokens > 0 ? n_valid_tokens : 1);
                        int64_t t1 = start_time_ms + (segment_duration_ms * (token_idx + 1)) / (n_valid_tokens > 0 ? n_valid_tokens : 1);
                        auto it = m_ctx->vocab.id_to_token.find(id);
                        std::string tok_text = (it != m_ctx->vocab.id_to_token.end()) ? it->second : "";
                        token_data.push_back(std::make_tuple(t0, t1, tok_text));
                        token_idx++;
                    }
                    prev_id = id;
                }
            }
            
            if (!text_str.empty() && m_tsfn_callback) {
                auto callback_data = std::make_tuple(start_time_ms, end_time_ms, text_str, lang_str, emo_str, evt_str, token_data);
                m_tsfn_callback.BlockingCall([callback_data](Napi::Env env, Napi::Function jsCallback) {
                    Napi::Object result = Napi::Object::New(env);
                    result.Set("type", "segment");
                    result.Set("start", Napi::Number::New(env, std::get<0>(callback_data)));
                    result.Set("end", Napi::Number::New(env, std::get<1>(callback_data)));
                    result.Set("text", Napi::String::New(env, std::get<2>(callback_data)));
                    result.Set("language", Napi::String::New(env, std::get<3>(callback_data)));
                    result.Set("emotion", Napi::String::New(env, std::get<4>(callback_data)));
                    result.Set("event", Napi::String::New(env, std::get<5>(callback_data)));
                    
                    // Add token timestamps
                    const auto& tokens = std::get<6>(callback_data);
                    Napi::Array tokensArr = Napi::Array::New(env, tokens.size());
                    for (size_t i = 0; i < tokens.size(); i++) {
                        Napi::Object tok = Napi::Object::New(env);
                        tok.Set("t0", Napi::Number::New(env, std::get<0>(tokens[i])));
                        tok.Set("t1", Napi::Number::New(env, std::get<1>(tokens[i])));
                        tok.Set("text", Napi::String::New(env, std::get<2>(tokens[i])));
                        tokensArr[i] = tok;
                    }
                    result.Set("tokens", tokensArr);
                    
                    jsCallback.Call({env.Null(), result});
                });
            }
        }
        
        m_n_samples_processed += audio.size();
    } catch (...) {
        // Catch any exception to prevent crash
    }
}



