#include "napi.h"
#include "common.h"
#include "common-whisper.h"
#include "whisper.h"
#include "sense-voice.h"

#include <string>
#include <thread>
#include <vector>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cfloat>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <algorithm>
#include <memory> 

// Helper function for UTF-8 fix
std::string string_to_hex(const char* text) {
    if (text == nullptr) {
        return "";
    }
    std::stringstream hex_stream;
    hex_stream << std::hex << std::setfill('0');

    for (size_t i = 0; text[i] != '\0'; ++i) {
        hex_stream << std::setw(2) << static_cast<int>(static_cast<unsigned char>(text[i]));
    }

    return hex_stream.str();
}

// Checks if a string is a sequence of one or more valid UTF-8 characters
bool is_valid_utf8(const std::string &s) {
    int i = 0;
    while (i < s.length()) {
        unsigned char c = s[i];
        int len = 0;
        if ((c & 0x80) == 0x00) { // 1-byte
            len = 1;
        } else if ((c & 0xE0) == 0xC0) { // 2-byte
            len = 2;
        } else if ((c & 0xF0) == 0xE0) { // 3-byte
            len = 3;
        } else if ((c & 0xF8) == 0xF0) { // 4-byte
            len = 4;
        } else {
            return false; // Invalid start byte
        }

        if (i + len > s.length()) {
            return false; // Incomplete string
        }

        for (int j = 1; j < len; ++j) {
            if ((s[i + j] & 0xC0) != 0x80) {
                return false; // Invalid subsequent byte
            }
        }
        i += len;
    }
    return true;
}


struct token_data {
    std::string text;
    std::string start_timestamp;
    std::string end_timestamp;
    int64_t id;
    float p;
};

struct segment_data {
    std::string start_timestamp;
    std::string end_timestamp;
    std::string text;
    std::vector<token_data> tokens;
};

// Combined parameters from both versions, with a new option for tokens
struct whisper_params {
    int32_t n_threads      = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_processors   = 1;
    int32_t offset_t_ms    = 0;
    int32_t offset_n       = 0;
    int32_t duration_ms    = 0;
    int32_t max_context    = -1;
    int32_t max_len        = 0;
    int32_t best_of        = 5;
    int32_t beam_size      = -1;
    int32_t audio_ctx      = 0;

    float word_thold    = 0.01f;
    float entropy_thold = 2.4f;
    float logprob_thold = -1.0f;

    bool translate       = false;
    bool diarize         = false;
    bool output_txt      = false;
    bool output_vtt      = false;
    bool output_srt      = false;
    bool output_wts      = false;
    bool output_csv      = false;
    bool print_special   = false;
    bool print_colors    = false;
    bool print_progress  = false;
    bool no_timestamps   = false;
    bool no_prints       = false;
    bool detect_language = false;
    bool use_gpu         = true;
    bool flash_attn      = false;
    bool comma_in_time   = true;
    bool output_tokens   = false; // [New] Option to control whether to output token data
    bool debug_mode      = false;

    std::string language = "en";
    std::string prompt;
    std::string model    = "../../ggml-large.bin";

    std::vector<std::string> fname_inp = {};
    std::vector<std::string> fname_out = {};

    std::vector<float> pcmf32 = {};

    // Voice Activity Detection (VAD) parameters
    bool        vad                       = false;
    std::string vad_model                 = "";
    float       vad_threshold             = 0.5f;
    int         vad_min_speech_duration_ms = 250;
    int         vad_min_silence_duration_ms = 100;
    float       vad_max_speech_duration_s = FLT_MAX;
    int         vad_speech_pad_ms         = 30;
    float       vad_samples_overlap       = 0.1f;
	
    // Streaming parameters
    int32_t step_ms      = 3000;
    int32_t length_ms    = 10000;
    int32_t keep_ms      = 200;
};

struct whisper_print_user_data {
    const whisper_params * params;
    const std::vector<std::vector<float>> * pcmf32s;
};

void whisper_print_segment_callback(struct whisper_context * ctx, struct whisper_state * state, int n_new, void * user_data) {
    const auto & params  = *((whisper_print_user_data *) user_data)->params;
    const auto & pcmf32s = *((whisper_print_user_data *) user_data)->pcmf32s;
    const int n_segments = whisper_full_n_segments(ctx);
    std::string speaker = "";
    int64_t t0;
    int64_t t1;

    const int s0 = n_segments - n_new;
    if (s0 == 0 && !params.no_prints) {
        printf("\n");
    }

    for (int i = s0; i < n_segments; i++) {
        if (!params.no_timestamps || params.diarize) {
            t0 = whisper_full_get_segment_t0(ctx, i);
            t1 = whisper_full_get_segment_t1(ctx, i);
        }

        if (!params.no_timestamps && !params.no_prints) {
            printf("[%s --> %s]  ", to_timestamp(t0).c_str(), to_timestamp(t1).c_str());
        }

        if (params.diarize && pcmf32s.size() == 2) {
            const int64_t n_samples = pcmf32s[0].size();
            const int64_t is0 = timestamp_to_sample(t0, n_samples, WHISPER_SAMPLE_RATE);
            const int64_t is1 = timestamp_to_sample(t1, n_samples, WHISPER_SAMPLE_RATE);
            double energy0 = 0.0f;
            double energy1 = 0.0f;
            for (int64_t j = is0; j < is1; j++) {
                energy0 += fabs(pcmf32s[0][j]);
                energy1 += fabs(pcmf32s[1][j]);
            }
            if (energy0 > 1.1*energy1) {
                speaker = "(speaker 0)";
            } else if (energy1 > 1.1*energy0) {
                speaker = "(speaker 1)";
            } else {
                speaker = "(speaker ?)";
            }
        }

        if (!params.no_prints) {
            const char * text = whisper_full_get_segment_text(ctx, i);
            printf("%s%s", speaker.c_str(), text);
        }

        if ((!params.no_timestamps || params.diarize) && !params.no_prints) {
            printf("\n");
        }
        fflush(stdout);
    }
}

std::atomic<bool> g_addon_debug_mode{false};

// Global log callback to suppress INFO logs unless requested
void addon_whisper_log_callback(ggml_log_level level, const char * text, void * user_data) {
    // Check environment variable once
    static bool debug_env_checked = false;
    static bool debug_enabled = false;
    if (!debug_env_checked) {
        const char* env = std::getenv("WHISPER_CPP_DEBUG");
        debug_enabled = (env != nullptr && std::string(env) == "1");
        debug_env_checked = true;
    }

    if (!debug_enabled && !g_addon_debug_mode.load() && level < GGML_LOG_LEVEL_WARN) {
        return;
    }
    fprintf(stderr, "%s", text);
}

void cb_log_disable(enum ggml_log_level, const char *, void *) {}

struct whisper_result {
    std::vector<segment_data> segments;
    std::string language;
};

class ProgressWorker : public Napi::AsyncWorker {
public:
    // Abort flags - using shared_ptr to ensure they survive after worker destruction
    // This is important for NonBlockingCall callbacks that may execute after worker is destroyed
    std::shared_ptr<std::atomic<bool>> m_should_abort;
    std::shared_ptr<std::atomic<bool>> m_was_aborted;

    ProgressWorker(Napi::Function& callback, whisper_params params, Napi::Function progress_callback, Napi::Env env)
        : Napi::AsyncWorker(callback), params(params), env(env),
          m_should_abort(std::make_shared<std::atomic<bool>>(false)),
          m_was_aborted(std::make_shared<std::atomic<bool>>(false)) {
        if (!progress_callback.IsEmpty()) {
            tsfn = Napi::ThreadSafeFunction::New(
                env,
                progress_callback,
                "Progress Callback",
                0,
                1
            );
        }
    }

    ~ProgressWorker() {
        if (tsfn) {
            tsfn.Release();
        }
    }

    void Execute() override {
        run_with_progress(params, result);
    }

    void OnOK() override {
        Napi::HandleScope scope(Env());

        Napi::Array segments_array = Napi::Array::New(Env(), result.segments.size());
        for (uint64_t i = 0; i < result.segments.size(); ++i) {
            const auto& seg = result.segments[i];
            Napi::Object segment_obj = Napi::Object::New(Env());
            segment_obj.Set("start", Napi::String::New(Env(), seg.start_timestamp));
            segment_obj.Set("end", Napi::String::New(Env(), seg.end_timestamp));
            segment_obj.Set("text", Napi::String::New(Env(), seg.text));

            // If there is token data, create a tokens array
            if (!seg.tokens.empty()) {
                Napi::Array tokens_array = Napi::Array::New(Env(), seg.tokens.size());
                for (uint64_t j = 0; j < seg.tokens.size(); ++j) {
                    const auto& tok = seg.tokens[j];
                    Napi::Object token_obj = Napi::Object::New(Env());
                    token_obj.Set("text", Napi::String::New(Env(), tok.text));
                    token_obj.Set("id", Napi::Number::New(Env(), tok.id));
                    token_obj.Set("p", Napi::Number::New(Env(), tok.p));
                    token_obj.Set("start", Napi::String::New(Env(), tok.start_timestamp));
                    token_obj.Set("end", Napi::String::New(Env(), tok.end_timestamp));
                    tokens_array[j] = token_obj;
                }
                segment_obj.Set("tokens", tokens_array);
            }

            segments_array[i] = segment_obj;
        }

        Napi::Object final_res = Napi::Object::New(Env());
        final_res.Set("segments", segments_array);
        final_res.Set("language", Napi::String::New(Env(), result.language));
        final_res.Set("aborted", Napi::Boolean::New(Env(), m_was_aborted->load()));

        Callback().Call({Env().Null(), final_res});
    }

    void OnProgress(int progress) {
        if (tsfn && !m_should_abort->load()) {
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
            tsfn.BlockingCall(callback);
        }
    }

    // Method to request abort from outside
    void RequestAbort() {
        m_should_abort->store(true);
    }

    bool ShouldAbort() const {
        return m_should_abort->load();
    }

private:
    whisper_params params;
    whisper_result result;
    Napi::Env env;
    Napi::ThreadSafeFunction tsfn;

    int run_with_progress(whisper_params &params, whisper_result &result) {
        if (params.no_prints) {
            whisper_log_set(cb_log_disable, NULL);
        }

        if (params.fname_inp.empty() && params.pcmf32.empty()) {
            fprintf(stderr, "error: no input files or audio buffer specified\n");
            return 2;
        }

        if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1) {
            fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
            exit(0);
        }

        struct whisper_context_params cparams = whisper_context_default_params();
        cparams.use_gpu = params.use_gpu;
        cparams.flash_attn = params.flash_attn;
        struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);

        if (ctx == nullptr) {
            fprintf(stderr, "error: failed to initialize whisper context\n");
            return 3;
        }

        if (!params.pcmf32.empty()) {
            if (!params.no_prints) fprintf(stderr, "info: using audio buffer as input\n");
            params.fname_inp.clear();
            params.fname_inp.emplace_back("buffer");
        }

        for (int f = 0; f < (int) params.fname_inp.size(); ++f) {
            const auto fname_inp = params.fname_inp[f];
            std::vector<float> pcmf32;
            std::vector<std::vector<float>> pcmf32s;

            if (params.pcmf32.empty()) {
                if (!::read_audio_data(fname_inp, pcmf32, pcmf32s, params.diarize)) {
                    fprintf(stderr, "error: failed to read audio file '%s'\n", fname_inp.c_str());
                    continue;
                }
            } else {
                pcmf32 = params.pcmf32;
            }

            if (!params.no_prints) {
                fprintf(stderr, "\n");
                fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                        params.n_threads*params.n_processors, std::thread::hardware_concurrency(), whisper_print_system_info());
                fprintf(stderr, "\n");
                if (!whisper_is_multilingual(ctx)) {
                    if (params.language != "en" || params.translate) {
                        params.language = "en";
                        params.translate = false;
                        fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
                    }
                }
                fprintf(stderr, "%s: processing '%s' (%d samples, %.1f sec), %d threads, %d processors, lang = %s, task = %s, timestamps = %d, audio_ctx = %d ...\n",
                        __func__, fname_inp.c_str(), int(pcmf32.size()), float(pcmf32.size())/WHISPER_SAMPLE_RATE,
                        params.n_threads, params.n_processors,
                        params.language.c_str(),
                        params.translate ? "translate" : "transcribe",
                        params.no_timestamps ? 0 : 1,
                        params.audio_ctx);
                fprintf(stderr, "\n");
            }

            {
                whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
                wparams.strategy = params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;
                wparams.print_realtime   = false;
                wparams.print_progress   = params.print_progress;
                wparams.print_timestamps = !params.no_timestamps;
                wparams.print_special    = params.print_special;
                wparams.translate        = params.translate;
                wparams.language         = params.language.c_str();
                wparams.detect_language  = params.detect_language;
                wparams.n_threads        = params.n_threads;
                wparams.n_max_text_ctx   = params.max_context >= 0 ? params.max_context : wparams.n_max_text_ctx;
                wparams.offset_ms        = params.offset_t_ms;
                wparams.duration_ms      = params.duration_ms;
                wparams.thold_pt         = params.word_thold;
                wparams.entropy_thold    = params.entropy_thold;
                wparams.logprob_thold    = params.logprob_thold;
                wparams.max_len          = params.output_wts && params.max_len == 0 ? 60 : params.max_len;
                wparams.audio_ctx        = params.audio_ctx;
                wparams.greedy.best_of   = params.best_of;
                wparams.beam_search.beam_size = params.beam_size;
                wparams.initial_prompt   = params.prompt.c_str();
                wparams.no_timestamps    = params.no_timestamps;

                // [Important] If the user requests token output, token_timestamps must be enabled
                wparams.token_timestamps = params.output_wts || params.max_len > 0 || params.output_tokens;

                whisper_print_user_data user_data = { &params, &pcmf32s };
                if (!wparams.print_realtime) {
                    wparams.new_segment_callback           = whisper_print_segment_callback;
                    wparams.new_segment_callback_user_data = &user_data;
                }

                wparams.progress_callback = [](struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, int progress, void * user_data) {
                    static_cast<ProgressWorker*>(user_data)->OnProgress(progress);
                };
                wparams.progress_callback_user_data = this;

                // Set up abort callback - returns true to abort when m_should_abort is set
                wparams.abort_callback = [](void * user_data) -> bool {
                    auto* worker = static_cast<ProgressWorker*>(user_data);
                    if (worker->m_should_abort->load()) {
                        worker->m_was_aborted->store(true);
                        return true; // Return true to abort
                    }
                    return false;
                };
                wparams.abort_callback_user_data = this;

                // VAD parameters
                wparams.vad                           = params.vad;
                wparams.vad_model_path                = params.vad_model.c_str();
                wparams.vad_params.threshold          = params.vad_threshold;
                wparams.vad_params.min_speech_duration_ms  = params.vad_min_speech_duration_ms;
                wparams.vad_params.min_silence_duration_ms = params.vad_min_silence_duration_ms;
                wparams.vad_params.max_speech_duration_s   = params.vad_max_speech_duration_s;
                wparams.vad_params.speech_pad_ms           = params.vad_speech_pad_ms;
                wparams.vad_params.samples_overlap         = params.vad_samples_overlap;

                if (whisper_full_parallel(ctx, wparams, pcmf32.data(), pcmf32.size(), params.n_processors) != 0) {
                    fprintf(stderr, "failed to process audio\n");
                    whisper_free(ctx);
                    return 10;
                }
            }
        }

        int lang_id = whisper_full_lang_id(ctx);
        result.language = (lang_id != -1) ? whisper_lang_str(lang_id) : "unknown";
        if (!params.no_prints && (params.detect_language || params.language == "auto")) {
            fprintf(stderr, "%s: detected language: %s\n", __func__, result.language.c_str());
        }

        // Processing loop that integrates UTF-8 fix and token extraction
        const int n_segments = whisper_full_n_segments(ctx);
        result.segments.clear();
        std::string hex_buffer;
        int64_t pending_t0 = -1;

        for (int i = 0; i < n_segments; ++i) {
            const char * text = whisper_full_get_segment_text(ctx, i);
            int64_t current_t0 = whisper_full_get_segment_t0(ctx, i);
            const int64_t current_t1 = whisper_full_get_segment_t1(ctx, i);
            if (current_t0 < 0) current_t0 = 0;

            std::string current_text_hex = string_to_hex(text);
            std::string combined_hex = hex_buffer + current_text_hex;
            hex_buffer.clear();
            std::string complete_text;
            size_t consumed_hex_len = 0;

            while (consumed_hex_len + 2 <= combined_hex.length()) {
                int char_length = 0;
                unsigned int first_byte;
                std::stringstream ss_first;
                ss_first << std::hex << combined_hex.substr(consumed_hex_len, 2);
                ss_first >> first_byte;

                if ((first_byte & 0x80) == 0) { char_length = 2; }
                else if ((first_byte & 0xE0) == 0xC0) { char_length = 4; }
                else if ((first_byte & 0xF0) == 0xE0) { char_length = 6; }
                else if ((first_byte & 0xF8) == 0xF0) { char_length = 8; }
                else {
                    consumed_hex_len += 2;
                    continue;
                }

                if (consumed_hex_len + char_length <= combined_hex.length()) {
                    std::string hex_char = combined_hex.substr(consumed_hex_len, char_length);
                    std::string byte_str;
                    bool conversion_ok = true;
                    for (size_t j = 0; j < hex_char.length(); j += 2) {
                        unsigned int byte;
                        std::stringstream ss_byte;
                        ss_byte << std::hex << hex_char.substr(j, 2);
                        if (!(ss_byte >> byte)) {
                            conversion_ok = false;
                            break;
                        }
                        byte_str += static_cast<char>(byte);
                    }

                    if (conversion_ok) {
                        bool valid_utf8 = true;
                        if (char_length > 2) {
                            for(size_t k = 1; k < byte_str.length(); ++k) {
                                if ((static_cast<unsigned char>(byte_str[k]) & 0xC0) != 0x80) {
                                    valid_utf8 = false;
                                    break;
                                }
                            }
                        }
                        if (valid_utf8) {
                            complete_text += byte_str;
                            consumed_hex_len += char_length;
                        } else {
                            consumed_hex_len += 2;
                        }
                    } else {
                        consumed_hex_len += 2;
                    }
                } else {
                    break;
                }
            }

            hex_buffer = combined_hex.substr(consumed_hex_len);

            auto create_and_add_segment = [&](const std::string& text, int64_t t0, int64_t t1, int segment_index) {
                if (text.empty()) return;
                
                segment_data seg;
                seg.start_timestamp = to_timestamp(t0, params.comma_in_time);
                seg.end_timestamp = to_timestamp(t1, params.comma_in_time);
                seg.text = text;

                if (params.output_tokens) {
                    const int n_tokens = whisper_full_n_tokens(ctx, segment_index);
                    if (n_tokens > 0) {
                        auto token_data_first_overall = whisper_full_get_token_data(ctx, segment_index, 0);
                        int64_t segment_t0 = whisper_full_get_segment_t0(ctx, segment_index);
                        int64_t segment_t1 = whisper_full_get_segment_t1(ctx, segment_index);
                        
                        // Calculate missing VAD/chunk offset
                        int64_t missing_offset_10ms = segment_t0 - token_data_first_overall.t0;

                        int j = 0;
                        while (j < n_tokens) {
                            auto token_data_first = whisper_full_get_token_data(ctx, segment_index, j);
                            std::string current_text = whisper_token_to_str(ctx, token_data_first.id);

                            if (is_valid_utf8(current_text)) {
                                // Token is a valid UTF-8 character, add it directly
                                token_data td;
                                td.text = current_text;
                                td.id = token_data_first.id;
                                td.p = token_data_first.p;

                                int64_t t0 = std::max(segment_t0, std::min(segment_t1, token_data_first.t0 + missing_offset_10ms));
                                int64_t t1 = std::max(segment_t0, std::min(segment_t1, token_data_first.t1 + missing_offset_10ms));

                                td.start_timestamp = to_timestamp(t0, params.comma_in_time);
                                td.end_timestamp = to_timestamp(t1, params.comma_in_time);
                                seg.tokens.push_back(td);
                                j++;
                            } else {
                                // Token is an incomplete UTF-8, start merging
                                std::string merged_text = current_text;
                                int64_t start_time = token_data_first.t0;
                                int64_t end_time = token_data_first.t1;
                                int k = j + 1;
                                
                                while (k < n_tokens) {
                                    auto token_data_next = whisper_full_get_token_data(ctx, segment_index, k);
                                    merged_text += whisper_token_to_str(ctx, token_data_next.id);
                                    
                                    if (is_valid_utf8(merged_text)) {
                                        // Becomes a valid UTF-8 character after merging
                                        end_time = token_data_next.t1;
                                        break;
                                    }
                                    k++;
                                }

                                token_data td;
                                td.text = merged_text;
                                td.id = token_data_first.id; // Use the id and p of the first token
                                td.p = token_data_first.p;

                                int64_t t0 = std::max(segment_t0, std::min(segment_t1, start_time + missing_offset_10ms));
                                int64_t t1 = std::max(segment_t0, std::min(segment_t1, end_time + missing_offset_10ms));

                                td.start_timestamp = to_timestamp(t0, params.comma_in_time);
                                td.end_timestamp = to_timestamp(t1, params.comma_in_time);
                                seg.tokens.push_back(td);

                                j = k + 1; // Update the index of the main loop
                            }
                        }
                    }
                }
                result.segments.push_back(seg);
            };

            if (!hex_buffer.empty()) {
                if (pending_t0 < 0) {
                    pending_t0 = current_t0;
                }
                create_and_add_segment(complete_text, (pending_t0 >= 0 && pending_t0 != current_t0) ? pending_t0 : current_t0, current_t1, i);
            } else {
                create_and_add_segment(complete_text, (pending_t0 >= 0) ? pending_t0 : current_t0, current_t1, i);
                pending_t0 = -1;
            }
        }

        if (!hex_buffer.empty()) {
            fprintf(stderr, "Warning: Transcription ended with incomplete UTF-8 sequence (hex: %s). Discarding.\n", hex_buffer.c_str());
        }

        whisper_print_timings(ctx);
        whisper_free(ctx);
        return 0;
    }
};

Napi::Value whisper(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() <= 1 || !info[0].IsObject() || !info[1].IsFunction()) {
        Napi::TypeError::New(env, "Usage: whisper(options, callback)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    whisper_params params;

    Napi::Object whisper_params = info[0].As<Napi::Object>();
    params.language = whisper_params.Get("language").As<Napi::String>();
    params.model = whisper_params.Get("model").As<Napi::String>();
    params.fname_inp.emplace_back(whisper_params.Get("fname_inp").As<Napi::String>());

    if (whisper_params.Has("use_gpu") && whisper_params.Get("use_gpu").IsBoolean()) {
        params.use_gpu = whisper_params.Get("use_gpu").As<Napi::Boolean>();
    }
    if (whisper_params.Has("flash_attn") && whisper_params.Get("flash_attn").IsBoolean()) {
        params.flash_attn = whisper_params.Get("flash_attn").As<Napi::Boolean>();
    }
    if (whisper_params.Has("no_prints") && whisper_params.Get("no_prints").IsBoolean()) {
        params.no_prints = whisper_params.Get("no_prints").As<Napi::Boolean>();
    }
    if (whisper_params.Has("no_timestamps") && whisper_params.Get("no_timestamps").IsBoolean()) {
        params.no_timestamps = whisper_params.Get("no_timestamps").As<Napi::Boolean>();
    }
    if (whisper_params.Has("detect_language") && whisper_params.Get("detect_language").IsBoolean()) {
        params.detect_language = whisper_params.Get("detect_language").As<Napi::Boolean>();
    }
    if (whisper_params.Has("audio_ctx") && whisper_params.Get("audio_ctx").IsNumber()) {
        params.audio_ctx = whisper_params.Get("audio_ctx").As<Napi::Number>();
    }
    if (whisper_params.Has("comma_in_time") && whisper_params.Get("comma_in_time").IsBoolean()) {
        params.comma_in_time = whisper_params.Get("comma_in_time").As<Napi::Boolean>();
    }
    if (whisper_params.Has("max_len") && whisper_params.Get("max_len").IsNumber()) {
        params.max_len = whisper_params.Get("max_len").As<Napi::Number>();
    }
    if (whisper_params.Has("max_context") && whisper_params.Get("max_context").IsNumber()) {
        params.max_context = whisper_params.Get("max_context").As<Napi::Number>();
    }
    if (whisper_params.Has("prompt") && whisper_params.Get("prompt").IsString()) {
        params.prompt = whisper_params.Get("prompt").As<Napi::String>();
    }
    if (whisper_params.Has("print_progress") && whisper_params.Get("print_progress").IsBoolean()) {
        params.print_progress = whisper_params.Get("print_progress").As<Napi::Boolean>();
    }
    if (whisper_params.Has("output_tokens") && whisper_params.Get("output_tokens").IsBoolean()) {
        params.output_tokens = whisper_params.Get("output_tokens").As<Napi::Boolean>();
    }
    if (whisper_params.Has("debug") && whisper_params.Get("debug").IsBoolean()) {
        params.debug_mode = whisper_params.Get("debug").As<Napi::Boolean>();
        g_addon_debug_mode.store(params.debug_mode);
    }

    Napi::Function progress_callback;
    if (whisper_params.Has("progress_callback") && whisper_params.Get("progress_callback").IsFunction()) {
        progress_callback = whisper_params.Get("progress_callback").As<Napi::Function>();
    }

    // VAD parameters
    if (whisper_params.Has("vad") && whisper_params.Get("vad").IsBoolean()) {
        params.vad = whisper_params.Get("vad").As<Napi::Boolean>();
    }
    if (whisper_params.Has("vad_model") && whisper_params.Get("vad_model").IsString()) {
        params.vad_model = whisper_params.Get("vad_model").As<Napi::String>();
    }
    if (whisper_params.Has("vad_threshold") && whisper_params.Get("vad_threshold").IsNumber()) {
        params.vad_threshold = whisper_params.Get("vad_threshold").As<Napi::Number>();
    }
    if (whisper_params.Has("vad_min_speech_duration_ms") && whisper_params.Get("vad_min_speech_duration_ms").IsNumber()) {
        params.vad_min_speech_duration_ms = whisper_params.Get("vad_min_speech_duration_ms").As<Napi::Number>();
    }
    if (whisper_params.Has("vad_min_silence_duration_ms") && whisper_params.Get("vad_min_silence_duration_ms").IsNumber()) {
        params.vad_min_silence_duration_ms = whisper_params.Get("vad_min_silence_duration_ms").As<Napi::Number>();
    }
    if (whisper_params.Has("vad_max_speech_duration_s") && whisper_params.Get("vad_max_speech_duration_s").IsNumber()) {
        params.vad_max_speech_duration_s = whisper_params.Get("vad_max_speech_duration_s").As<Napi::Number>();
    }
    if (whisper_params.Has("vad_speech_pad_ms") && whisper_params.Get("vad_speech_pad_ms").IsNumber()) {
        params.vad_speech_pad_ms = whisper_params.Get("vad_speech_pad_ms").As<Napi::Number>();
    }
    if (whisper_params.Has("vad_samples_overlap") && whisper_params.Get("vad_samples_overlap").IsNumber()) {
        params.vad_samples_overlap = whisper_params.Get("vad_samples_overlap").As<Napi::Number>();
    }


    Napi::Value pcmf32Value = whisper_params.Get("pcmf32");
    if (pcmf32Value.IsTypedArray()) {
        Napi::Float32Array pcmf32 = pcmf32Value.As<Napi::Float32Array>();
        size_t length = pcmf32.ElementLength();
        params.pcmf32.reserve(length);
        for (size_t i = 0; i < length; i++) {
            params.pcmf32.push_back(pcmf32[i]);
        }
    }

    Napi::Function callback = info[1].As<Napi::Function>();
    ProgressWorker* worker = new ProgressWorker(callback, params, progress_callback, env);
    worker->Queue();
    return env.Undefined();
}
struct StreamTokenData {
    std::string text;
    int64_t id;
    float p;
    int64_t start_ms;
    int64_t end_ms;
};

struct SegmentData {
    int64_t start_ms;
    int64_t end_ms;
    std::string text;
    bool speaker_turn;
    std::vector<StreamTokenData> tokens;
};

enum class StreamState {
    IDLE,
    RUNNING,
    PAUSED,
    STOPPING,  // Hard stop
    FINISHING  // Soft stop (graceful shutdown)
};

// --- WhisperStream Class Definition ---

class WhisperStream : public Napi::ObjectWrap<WhisperStream> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    WhisperStream(const Napi::CallbackInfo& info);
    ~WhisperStream();

private:
    // N-API Methods
    Napi::Value Start(const Napi::CallbackInfo& info);
    Napi::Value AddAudio(const Napi::CallbackInfo& info);
    Napi::Value Stop(const Napi::CallbackInfo& info);
    Napi::Value Pause(const Napi::CallbackInfo& info);
    Napi::Value Resume(const Napi::CallbackInfo& info);
    Napi::Value Finish(const Napi::CallbackInfo& info); // New method

    // Internal worker and callback
    void StreamWorker();
    void StreamWorkerVAD();
    static void OnNewSegmentCallback(struct whisper_context * ctx, struct whisper_state * state, int n_new, void * user_data);

    // Whisper context and parameters
    whisper_context* m_ctx = nullptr;
    whisper_full_params m_wparams;

    // Model and language settings
    std::string m_model_path;
    std::string m_language;
    int m_n_threads = 4;
    bool m_use_gpu = true;

    // Streaming parameters
    int m_step_ms = 3000;
    int m_length_ms = 10000;
    int m_keep_ms = 200;

    // Progressive parameters
    bool m_progressive_update = false;
    int m_progressive_interval_ms = 500;
    int m_progressive_initial_ms = 5000;
    int m_progressive_window_tokens = 3;

    // VAD parameters (for vad_simple - used to decide when to process audio in StreamWorkerVAD)
    bool m_use_vad = false;
    float m_vad_thold = 0.6f;
    float m_freq_thold = 100.0f;

    // VAD parameters (for whisper built-in VAD - used for precise speech segmentation during transcription)
    // Note: These two VAD methods can be used independently or together:
    // - vad_simple (m_use_vad): Fast VAD to decide when to process audio chunks
    // - whisper VAD (m_vad_model): Precise VAD for speech segmentation within chunks
    std::string m_vad_model = "";
    float m_vad_threshold = 0.5f;
    int m_vad_min_speech_duration_ms = 250;
    int m_vad_min_silence_duration_ms = 100;
    float m_vad_max_speech_duration_s = FLT_MAX;
    int m_vad_speech_pad_ms = 30;
    float m_vad_samples_overlap = 0.1f;

    // Audio buffer and state
    std::deque<float> m_audio_buffer;
    std::vector<float> m_pcmf32_local;
    int64_t m_n_samples_processed = 0;
    int64_t m_current_callback_offset_samples = 0;

    std::thread m_worker_thread;
    std::atomic<StreamState> m_state;
    std::mutex m_mutex;
    std::condition_variable m_cv;

    Napi::ThreadSafeFunction m_tsfn_callback;
    
    // Other whisper params
    bool m_tinydiarize = false;
    int m_max_tokens = 0;
    int m_audio_ctx = 0;
    bool m_translate = false;
    bool m_single_segment = false;
    bool m_no_timestamps = false;
    std::string m_prompt;
    bool m_debug_mode = false;
    bool m_output_tokens = false;
};

// --- Implementation of WhisperStream Methods ---

Napi::Object WhisperStream::Init(Napi::Env env, Napi::Object exports) {
    Napi::HandleScope scope(env);
    Napi::Function func = DefineClass(env, "WhisperStream", {
        InstanceMethod("start", &WhisperStream::Start),
        InstanceMethod("addAudio", &WhisperStream::AddAudio),
        InstanceMethod("stop", &WhisperStream::Stop),
        InstanceMethod("pause", &WhisperStream::Pause),
        InstanceMethod("resume", &WhisperStream::Resume),
        InstanceMethod("finish", &WhisperStream::Finish), // Register new method
    });
    exports.Set("WhisperStream", func);
    return exports;
}

WhisperStream::WhisperStream(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<WhisperStream>(info), m_state(StreamState::IDLE) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsObject()) {
        Napi::TypeError::New(env, "Constructor requires an options object").ThrowAsJavaScriptException(); return;
    }
    Napi::Object options = info[0].As<Napi::Object>();
    if (options.Has("model") && options.Get("model").IsString()) {
        m_model_path = options.Get("model").As<Napi::String>();
    } else {
        Napi::TypeError::New(env, "Constructor options must include a 'model' path").ThrowAsJavaScriptException(); return;
    }
    if (options.Has("language") && options.Get("language").IsString()) m_language = options.Get("language").As<Napi::String>(); else m_language = "en";
    if (options.Has("n_threads") && options.Get("n_threads").IsNumber()) m_n_threads = options.Get("n_threads").As<Napi::Number>();
    if (options.Has("use_gpu") && options.Get("use_gpu").IsBoolean()) m_use_gpu = options.Get("use_gpu").As<Napi::Boolean>();
    if (options.Has("step_ms") && options.Get("step_ms").IsNumber()) m_step_ms = options.Get("step_ms").As<Napi::Number>();
    if (options.Has("length_ms") && options.Get("length_ms").IsNumber()) m_length_ms = options.Get("length_ms").As<Napi::Number>();
    if (options.Has("keep_ms") && options.Get("keep_ms").IsNumber()) m_keep_ms = options.Get("keep_ms").As<Napi::Number>();
    if (options.Has("progressive_update") && options.Get("progressive_update").IsBoolean()) m_progressive_update = options.Get("progressive_update").As<Napi::Boolean>();
    if (options.Has("progressive_interval_ms") && options.Get("progressive_interval_ms").IsNumber()) m_progressive_interval_ms = options.Get("progressive_interval_ms").As<Napi::Number>().Int32Value();
    if (options.Has("progressive_initial_ms") && options.Get("progressive_initial_ms").IsNumber()) m_progressive_initial_ms = options.Get("progressive_initial_ms").As<Napi::Number>().Int32Value();
    if (options.Has("progressive_window_tokens") && options.Get("progressive_window_tokens").IsNumber()) m_progressive_window_tokens = options.Get("progressive_window_tokens").As<Napi::Number>().Int32Value();
    if (options.Has("debug") && options.Get("debug").IsBoolean()) {
        m_debug_mode = options.Get("debug").As<Napi::Boolean>();
        g_addon_debug_mode.store(m_debug_mode);
    }
    // VAD parameters (vad_simple)
    if (options.Has("use_vad") && options.Get("use_vad").IsBoolean()) m_use_vad = options.Get("use_vad").As<Napi::Boolean>();
    if (options.Has("vad_thold") && options.Get("vad_thold").IsNumber()) m_vad_thold = options.Get("vad_thold").As<Napi::Number>().FloatValue();
    if (options.Has("freq_thold") && options.Get("freq_thold").IsNumber()) m_freq_thold = options.Get("freq_thold").As<Napi::Number>().FloatValue();

    // VAD parameters (whisper built-in VAD)
    if (options.Has("vad_model") && options.Get("vad_model").IsString()) m_vad_model = options.Get("vad_model").As<Napi::String>();
    if (options.Has("vad_threshold") && options.Get("vad_threshold").IsNumber()) m_vad_threshold = options.Get("vad_threshold").As<Napi::Number>().FloatValue();
    if (options.Has("vad_min_speech_duration_ms") && options.Get("vad_min_speech_duration_ms").IsNumber()) m_vad_min_speech_duration_ms = options.Get("vad_min_speech_duration_ms").As<Napi::Number>();
    if (options.Has("vad_min_silence_duration_ms") && options.Get("vad_min_silence_duration_ms").IsNumber()) m_vad_min_silence_duration_ms = options.Get("vad_min_silence_duration_ms").As<Napi::Number>();
    if (options.Has("vad_max_speech_duration_s") && options.Get("vad_max_speech_duration_s").IsNumber()) m_vad_max_speech_duration_s = options.Get("vad_max_speech_duration_s").As<Napi::Number>().FloatValue();
    if (options.Has("vad_speech_pad_ms") && options.Get("vad_speech_pad_ms").IsNumber()) m_vad_speech_pad_ms = options.Get("vad_speech_pad_ms").As<Napi::Number>();
    if (options.Has("vad_samples_overlap") && options.Get("vad_samples_overlap").IsNumber()) m_vad_samples_overlap = options.Get("vad_samples_overlap").As<Napi::Number>().FloatValue();
    if (options.Has("tinydiarize") && options.Get("tinydiarize").IsBoolean()) m_tinydiarize = options.Get("tinydiarize").As<Napi::Boolean>();
    if (options.Has("max_tokens") && options.Get("max_tokens").IsNumber()) m_max_tokens = options.Get("max_tokens").As<Napi::Number>();
    if (options.Has("audio_ctx") && options.Get("audio_ctx").IsNumber()) m_audio_ctx = options.Get("audio_ctx").As<Napi::Number>();
    if (options.Has("translate") && options.Get("translate").IsBoolean()) m_translate = options.Get("translate").As<Napi::Boolean>();
    if (options.Has("single_segment") && options.Get("single_segment").IsBoolean()) m_single_segment = options.Get("single_segment").As<Napi::Boolean>();
    if (options.Has("no_timestamps") && options.Get("no_timestamps").IsBoolean()) m_no_timestamps = options.Get("no_timestamps").As<Napi::Boolean>();
    if (options.Has("output_tokens") && options.Get("output_tokens").IsBoolean()) m_output_tokens = options.Get("output_tokens").As<Napi::Boolean>();
    // Initial prompt
    if (options.Has("prompt") && options.Get("prompt").IsString()) m_prompt = options.Get("prompt").As<Napi::String>();

    m_wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
}

WhisperStream::~WhisperStream() {
    StreamState current_state = m_state.load();
    if (current_state != StreamState::IDLE) {
        m_state = StreamState::STOPPING;
        m_cv.notify_one();
        if (m_worker_thread.joinable()) {
            m_worker_thread.join();
        }
    }
    if (m_tsfn_callback) {
        m_tsfn_callback.Release();
    }
    if (m_ctx) {
        whisper_free(m_ctx);
    }
}

Napi::Value WhisperStream::Start(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsFunction()) {
        Napi::TypeError::New(env, "start() requires a callback function").ThrowAsJavaScriptException(); return env.Undefined();
    }
    if (m_state.load() != StreamState::IDLE) {
        Napi::Error::New(env, "Stream has already been started").ThrowAsJavaScriptException(); return env.Undefined();
    }
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = m_use_gpu;
    m_ctx = whisper_init_from_file_with_params(m_model_path.c_str(), cparams);
    if (m_ctx == nullptr) {
        Napi::Error::New(env, "Failed to initialize whisper context from model").ThrowAsJavaScriptException(); return env.Undefined();
    }
    m_audio_buffer.clear();
    m_pcmf32_local.clear();
    m_n_samples_processed = 0;
    m_current_callback_offset_samples = 0;
    
    Napi::Function callback = info[0].As<Napi::Function>();
    m_tsfn_callback = Napi::ThreadSafeFunction::New(env, callback, "WhisperStreamCallback", 0, 1);
    
    m_state = StreamState::RUNNING;
    if (m_use_vad) {
        m_worker_thread = std::thread(&WhisperStream::StreamWorkerVAD, this);
    } else {
        m_worker_thread = std::thread(&WhisperStream::StreamWorker, this);
    }
    return env.Undefined();
}

Napi::Value WhisperStream::AddAudio(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    StreamState current_state = m_state.load();
    // Do not accept new audio if stopping or finishing
    if (current_state != StreamState::RUNNING && current_state != StreamState::PAUSED) {
        return env.Undefined();
    }
    if (info.Length() < 1 || !info[0].IsTypedArray()) {
        Napi::TypeError::New(env, "addAudio() requires a Float32Array").ThrowAsJavaScriptException(); return env.Undefined();
    }
    if (m_audio_buffer.size() > WHISPER_SAMPLE_RATE * 30) {
        fprintf(stderr, "Warning: WhisperStream audio buffer is too large, dropping new audio.\n"); return env.Undefined();
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

Napi::Value WhisperStream::Stop(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    StreamState current_state = m_state.load();
    if (current_state == StreamState::IDLE || current_state == StreamState::STOPPING) {
        return env.Undefined();
    }
    m_state = StreamState::STOPPING;
    m_cv.notify_one();

    if (m_worker_thread.joinable()) {
        m_worker_thread.join();
    }

    if (m_tsfn_callback) {
        m_tsfn_callback.Abort(); // Abort any pending calls
        m_tsfn_callback.Release();
        m_tsfn_callback = nullptr;
    }
    if (m_ctx) {
        whisper_free(m_ctx);
        m_ctx = nullptr;
    }
    m_state = StreamState::IDLE;
    return env.Undefined();
}

Napi::Value WhisperStream::Finish(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    StreamState current_state = m_state.load();
    if (current_state == StreamState::RUNNING || current_state == StreamState::PAUSED) {
        m_state = StreamState::FINISHING;
        m_cv.notify_one(); // Wake up worker thread to notice the state change
    }
    return env.Undefined(); // Non-blocking
}

Napi::Value WhisperStream::Pause(const Napi::CallbackInfo& info) {
    if (m_state.load() == StreamState::RUNNING) {
        m_state = StreamState::PAUSED;
    }
    return info.Env().Undefined();
}

Napi::Value WhisperStream::Resume(const Napi::CallbackInfo& info) {
    if (m_state.load() == StreamState::PAUSED) {
        m_state = StreamState::RUNNING;
        m_cv.notify_one();
    }
    return info.Env().Undefined();
}

struct WhisperCallbackUserData {
    WhisperStream* self;
    bool is_progressive;
};

void WhisperStream::OnNewSegmentCallback(struct whisper_context * ctx, struct whisper_state * state, int n_new, void * user_data) {
    WhisperCallbackUserData* user_data_obj = static_cast<WhisperCallbackUserData*>(user_data);
    WhisperStream* self = user_data_obj->self;
    bool is_progressive = user_data_obj->is_progressive;

    const int64_t time_offset_ms = (self->m_current_callback_offset_samples * 1000) / WHISPER_SAMPLE_RATE;
    const int n_segments = whisper_full_n_segments_from_state(state);
    const int s0 = n_segments - n_new;
    std::vector<SegmentData> segments_data;
    for (int i = s0; i < n_segments; ++i) {
        const char* text_cstr = whisper_full_get_segment_text_from_state(state, i);
        if (!text_cstr || strlen(text_cstr) == 0) continue;
        SegmentData data;
        data.start_ms = time_offset_ms + (whisper_full_get_segment_t0_from_state(state, i) * 10);
        data.end_ms = time_offset_ms + (whisper_full_get_segment_t1_from_state(state, i) * 10);
        data.text = std::string(text_cstr);
        data.speaker_turn = whisper_full_get_segment_speaker_turn_next_from_state(state, i);

        if (self->m_output_tokens) {
            const int n_tokens = whisper_full_n_tokens_from_state(state, i);
            if (n_tokens > 0) {
                auto token_data_first_overall = whisper_full_get_token_data_from_state(state, i, 0);
                int64_t segment_t0 = whisper_full_get_segment_t0_from_state(state, i);
                int64_t missing_offset_10ms = segment_t0 - token_data_first_overall.t0;

                int j = 0;
                while (j < n_tokens) {
                    auto token_data_first = whisper_full_get_token_data_from_state(state, i, j);
                    std::string current_text = whisper_token_to_str(self->m_ctx, token_data_first.id);

                    if (is_valid_utf8(current_text)) {
                        StreamTokenData td;
                        td.text = current_text;
                        td.id = token_data_first.id;
                        td.p = token_data_first.p;

                        int64_t t0 = token_data_first.t0 + missing_offset_10ms;
                        int64_t t1 = token_data_first.t1 + missing_offset_10ms;
                        td.start_ms = time_offset_ms + (t0 * 10);
                        td.end_ms = time_offset_ms + (t1 * 10);

                        // Clamp to segment boundaries
                        td.start_ms = std::max(data.start_ms, std::min(data.end_ms, td.start_ms));
                        td.end_ms = std::max(data.start_ms, std::min(data.end_ms, td.end_ms));

                        data.tokens.push_back(td);
                        j++;
                    } else {
                        std::string merged_text = current_text;
                        int64_t start_time = token_data_first.t0;
                        int64_t end_time = token_data_first.t1;
                        int k = j + 1;

                        while (k < n_tokens) {
                            auto token_data_next = whisper_full_get_token_data_from_state(state, i, k);
                            merged_text += whisper_token_to_str(self->m_ctx, token_data_next.id);

                            if (is_valid_utf8(merged_text)) {
                                end_time = token_data_next.t1;
                                break;
                            }
                            k++;
                        }

                        StreamTokenData td;
                        td.text = merged_text;
                        td.id = token_data_first.id;
                        td.p = token_data_first.p;

                        int64_t t0 = start_time + missing_offset_10ms;
                        int64_t t1 = end_time + missing_offset_10ms;
                        td.start_ms = time_offset_ms + (t0 * 10);
                        td.end_ms = time_offset_ms + (t1 * 10);

                        // Clamp to segment boundaries
                        td.start_ms = std::max(data.start_ms, std::min(data.end_ms, td.start_ms));
                        td.end_ms = std::max(data.start_ms, std::min(data.end_ms, td.end_ms));

                        data.tokens.push_back(td);

                        j = k + 1;
                    }
                }
            }
        }
        segments_data.push_back(data);
    }
    if (self->m_tsfn_callback && !segments_data.empty()) {
        auto callback = [segments_data = std::move(segments_data), output_tokens = self->m_output_tokens, is_progressive] (Napi::Env env, Napi::Function jsCallback) {
            for (const auto& data : segments_data) {
                Napi::Object result = Napi::Object::New(env);
                result.Set("start", Napi::Number::New(env, data.start_ms));
                result.Set("end", Napi::Number::New(env, data.end_ms));
                result.Set("text", Napi::String::New(env, data.text));
                result.Set("speaker_turn", Napi::Boolean::New(env, data.speaker_turn));

                if (output_tokens && !data.tokens.empty()) {
                    Napi::Array tokens_array = Napi::Array::New(env, data.tokens.size());
                    for (size_t t = 0; t < data.tokens.size(); ++t) {
                        const auto& tok = data.tokens[t];
                        Napi::Object token_obj = Napi::Object::New(env);
                        token_obj.Set("text", Napi::String::New(env, tok.text));
                        token_obj.Set("id", Napi::Number::New(env, tok.id));
                        token_obj.Set("p", Napi::Number::New(env, tok.p));
                        token_obj.Set("start", Napi::Number::New(env, tok.start_ms));
                        token_obj.Set("end", Napi::Number::New(env, tok.end_ms));
                        tokens_array[t] = token_obj;
                    }
                    result.Set("tokens", tokens_array);
                }

                if (is_progressive) {
                    result.Set("type", "progressive");
                } else {
                    result.Set("type", "segment");
                }

                jsCallback.Call({env.Null(), result});
            }
        };
        self->m_tsfn_callback.NonBlockingCall(callback);
    }
}

void WhisperStream::StreamWorker() {
    m_wparams.print_progress   = false;
    m_wparams.print_realtime   = false;
    m_wparams.print_timestamps = false;
    m_wparams.language         = m_language.c_str();
    m_wparams.n_threads        = m_n_threads;

    WhisperCallbackUserData callback_user_data;
    callback_user_data.self = this;
    callback_user_data.is_progressive = false;

    m_wparams.new_segment_callback = WhisperStream::OnNewSegmentCallback;
    m_wparams.new_segment_callback_user_data = &callback_user_data;
    m_wparams.no_context = true;
    m_wparams.audio_ctx = m_audio_ctx;
    m_wparams.tdrz_enable = m_tinydiarize;
    m_wparams.max_tokens = m_max_tokens;
    m_wparams.translate = m_translate;
    m_wparams.single_segment = m_single_segment;
    m_wparams.no_timestamps = m_no_timestamps;
    m_wparams.token_timestamps = m_output_tokens;
    m_wparams.initial_prompt = m_prompt.empty() ? nullptr : m_prompt.c_str();

    // VAD parameters (whisper built-in VAD)
    if (!m_vad_model.empty()) {
        m_wparams.vad = true;
        m_wparams.vad_model_path = m_vad_model.c_str();
        m_wparams.vad_params.threshold = m_vad_threshold;
        m_wparams.vad_params.min_speech_duration_ms = m_vad_min_speech_duration_ms;
        m_wparams.vad_params.min_silence_duration_ms = m_vad_min_silence_duration_ms;
        m_wparams.vad_params.max_speech_duration_s = m_vad_max_speech_duration_s;
        m_wparams.vad_params.speech_pad_ms = m_vad_speech_pad_ms;
        m_wparams.vad_params.samples_overlap = m_vad_samples_overlap;
    }

    const size_t n_samples_step = (m_step_ms * WHISPER_SAMPLE_RATE) / 1000;
    m_pcmf32_local.clear();
    
    int64_t last_progressive_sample = 0;

    while (true) {
        bool is_stopping = false;
        bool is_finishing = false;

        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait(lock, [this, n_samples_step] {
                StreamState s = m_state.load();
                return s == StreamState::STOPPING || s == StreamState::FINISHING || m_audio_buffer.size() >= n_samples_step;
            });

            StreamState current_state = m_state.load();
            is_stopping = (current_state == StreamState::STOPPING);
            is_finishing = (current_state == StreamState::FINISHING);

            if (is_stopping) {
                break; // Hard stop
            }
            if (current_state == StreamState::PAUSED) {
                continue;
            }
            
            m_pcmf32_local.insert(m_pcmf32_local.end(), m_audio_buffer.begin(), m_audio_buffer.end());
            m_audio_buffer.clear();

            // If finishing, process all remaining audio and exit the loop
            if (is_finishing) {
                lock.unlock(); // Unlock before heavy processing
                if (!m_pcmf32_local.empty()) {
                    m_current_callback_offset_samples = m_n_samples_processed;
                    if (whisper_full(m_ctx, m_wparams, m_pcmf32_local.data(), m_pcmf32_local.size()) != 0) {
                        fprintf(stderr, "whisper_full failed on final audio chunk (finish)\n");
                    }
                }
                break; // Exit loop after processing all remaining audio
            }
        }

        if (m_progressive_update) {
            int64_t current_samples = m_pcmf32_local.size();
            int64_t elapsed_since_start_ms = (current_samples * 1000) / WHISPER_SAMPLE_RATE;
            
            if (elapsed_since_start_ms >= m_progressive_initial_ms) {
                int64_t elapsed_since_last_prog_ms = ((current_samples - last_progressive_sample) * 1000) / WHISPER_SAMPLE_RATE;
                if (elapsed_since_last_prog_ms >= m_progressive_interval_ms) {
                    m_current_callback_offset_samples = m_n_samples_processed;
                    callback_user_data.is_progressive = true;
                    if (whisper_full(m_ctx, m_wparams, m_pcmf32_local.data(), m_pcmf32_local.size()) != 0) {
                        fprintf(stderr, "whisper_full failed in progressive streaming mode\n");
                    }
                    
                    // Note: We do not erase here because standard WhisperStream naturally slides
                    // its window over time using `whisper_full`. Token overlap cropping isn't
                    // necessary since whisper retains its own internal state across chunks.

                    callback_user_data.is_progressive = false;
                    last_progressive_sample = current_samples;
                }
            }
        }
        
        if (!m_progressive_update) {
            while (m_pcmf32_local.size() >= n_samples_step) {
                m_current_callback_offset_samples = m_n_samples_processed;
                if (whisper_full(m_ctx, m_wparams, m_pcmf32_local.data(), n_samples_step) != 0) {
                    fprintf(stderr, "whisper_full failed in streaming mode\n");
                }
                m_n_samples_processed += n_samples_step;
                m_pcmf32_local.erase(m_pcmf32_local.begin(), m_pcmf32_local.begin() + n_samples_step);
                last_progressive_sample -= n_samples_step;
                if (last_progressive_sample < 0) last_progressive_sample = 0;
            }
        }
    }

    // After loop, thread is about to exit. Send 'end' signal.
    if (m_tsfn_callback) {
        m_tsfn_callback.BlockingCall([](Napi::Env env, Napi::Function jsCallback) {
            Napi::Object result = Napi::Object::New(env);
            result.Set("type", "end");
            jsCallback.Call({env.Null(), result});
        });
    }
}

void WhisperStream::StreamWorkerVAD() {
    m_wparams.print_progress   = false;
    m_wparams.print_realtime   = false;
    m_wparams.print_timestamps = false;
    m_wparams.language         = m_language.c_str();
    m_wparams.n_threads        = m_n_threads;

    WhisperCallbackUserData callback_user_data;
    callback_user_data.self = this;
    callback_user_data.is_progressive = false;

    m_wparams.new_segment_callback = WhisperStream::OnNewSegmentCallback;
    m_wparams.new_segment_callback_user_data = &callback_user_data;
    m_wparams.no_context = true;
    m_wparams.audio_ctx = m_audio_ctx;
    m_wparams.tdrz_enable = m_tinydiarize;
    m_wparams.max_tokens = m_max_tokens;
    m_wparams.translate = m_translate;
    m_wparams.single_segment = m_single_segment;
    m_wparams.no_timestamps = m_no_timestamps;
    m_wparams.token_timestamps = m_output_tokens;
    m_wparams.initial_prompt = m_prompt.empty() ? nullptr : m_prompt.c_str();

    // VAD parameters (whisper built-in VAD)
    if (!m_vad_model.empty()) {
        m_wparams.vad = true;
        m_wparams.vad_model_path = m_vad_model.c_str();
        m_wparams.vad_params.threshold = m_vad_threshold;
        m_wparams.vad_params.min_speech_duration_ms = m_vad_min_speech_duration_ms;
        m_wparams.vad_params.min_silence_duration_ms = m_vad_min_silence_duration_ms;
        m_wparams.vad_params.max_speech_duration_s = m_vad_max_speech_duration_s;
        m_wparams.vad_params.speech_pad_ms = m_vad_speech_pad_ms;
        m_wparams.vad_params.samples_overlap = m_vad_samples_overlap;
    }

    const int vad_window_ms = 2000;
    const int n_samples_vad_window = (vad_window_ms * WHISPER_SAMPLE_RATE) / 1000;
    const int vad_last_ms = 500;
    const size_t n_samples_max_len = (25 * WHISPER_SAMPLE_RATE);
    m_pcmf32_local.clear();

    int64_t last_progressive_sample = 0;

    while (true) {
        bool is_finishing = false;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait(lock, [this] {
                StreamState s = m_state.load();
                return s == StreamState::STOPPING || s == StreamState::FINISHING || !m_audio_buffer.empty();
            });

            StreamState current_state = m_state.load();
            if (current_state == StreamState::STOPPING) {
                break; // Hard stop
            }
            if (current_state == StreamState::PAUSED) {
                continue;
            }
            is_finishing = (current_state == StreamState::FINISHING);
            
            m_pcmf32_local.insert(m_pcmf32_local.end(), m_audio_buffer.begin(), m_audio_buffer.end());
            m_audio_buffer.clear();
        }

        bool should_process = false;
        if (is_finishing) {
            // Force processing for the final chunk
            should_process = true;
        } else {
            if (m_progressive_update) {
                int64_t current_samples = m_pcmf32_local.size();
                int64_t elapsed_since_start_ms = (current_samples * 1000) / WHISPER_SAMPLE_RATE;
                
                if (elapsed_since_start_ms >= m_progressive_initial_ms) {
                    int64_t elapsed_since_last_prog_ms = ((current_samples - last_progressive_sample) * 1000) / WHISPER_SAMPLE_RATE;
                    if (elapsed_since_last_prog_ms >= m_progressive_interval_ms) {
                        m_current_callback_offset_samples = m_n_samples_processed;
                        callback_user_data.is_progressive = true;
                        if (whisper_full(m_ctx, m_wparams, m_pcmf32_local.data(), m_pcmf32_local.size()) != 0) {
                            fprintf(stderr, "whisper_full failed in progressive VAD streaming mode\n");
                        }
                        
                        // Note: For progressive updates under StreamWorkerVAD, we are accumulating 
                        // all audio in `m_pcmf32_local` for the final segment transcription. We process 
                        // the entire accumulated buffer up to this point for progressive updates, 
                        // guaranteeing no context loss, and we do NOT erase from `m_pcmf32_local` 
                        // so the final segment is complete.
                        
                        callback_user_data.is_progressive = false;
                        last_progressive_sample = current_samples;
                    }
                }
            }
            
            if ((int)m_pcmf32_local.size() >= n_samples_vad_window) {
                std::vector<float> pcmf32_window(m_pcmf32_local.end() - n_samples_vad_window, m_pcmf32_local.end());
                if (vad_simple(pcmf32_window, WHISPER_SAMPLE_RATE, vad_last_ms, m_vad_thold, m_freq_thold, false)) {
                    should_process = true;
                }
            }
            if (m_pcmf32_local.size() > n_samples_max_len) {
                should_process = true;
            }
        }
        
        if (should_process && !m_pcmf32_local.empty()) {
            m_current_callback_offset_samples = m_n_samples_processed;
            if (whisper_full(m_ctx, m_wparams, m_pcmf32_local.data(), m_pcmf32_local.size()) != 0) {
                fprintf(stderr, "whisper_full failed in VAD streaming mode\n");
            }
            m_n_samples_processed += m_pcmf32_local.size();
            m_pcmf32_local.clear();
            last_progressive_sample = 0;
        }
        
        if (is_finishing) {
            break; // Exit loop after final processing
        }
    }

    // After loop, thread is about to exit. Send 'end' signal.
    if (m_tsfn_callback) {
        m_tsfn_callback.BlockingCall([](Napi::Env env, Napi::Function jsCallback) {
            Napi::Object result = Napi::Object::New(env);
            result.Set("type", "end");
            jsCallback.Call({env.Null(), result});
        });
    }
}

// ============================================================================
// SenseVoice ASR Implementation (in separate file)
// ============================================================================

#include "sense-voice-addon.cpp"
#include "vad-addon.cpp"
#include "nec-addon.cpp"
#include "qwen-asr-addon.cpp"
#include "ss-addon.cpp"

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    // Set custom log callback to suppress spammy INFO logs by default
    whisper_log_set(addon_whisper_log_callback, nullptr);

    exports.Set(
        Napi::String::New(env, "whisper"),
        Napi::Function::New(env, whisper)
    );
    exports.Set(
        Napi::String::New(env, "senseVoice"),
        Napi::Function::New(env, senseVoice)
    );
    exports.Set(
        Napi::String::New(env, "vadDetect"),
        Napi::Function::New(env, vadDetect)
    );
    exports.Set(
        Napi::String::New(env, "qwenASR"),
        Napi::Function::New(env, qwenASR)
    );
    exports.Set(
        Napi::String::New(env, "qwenASRAlign"),
        Napi::Function::New(env, qwenASRAlign)
    );
    exports.Set(
        Napi::String::New(env, "extractSSEmbedding"),
        Napi::Function::New(env, extractSSEmbedding)
    );

    InitNEC(env, exports);
    WhisperStream::Init(env, exports);
    SenseVoiceStream::Init(env, exports);
    VADStream::Init(env, exports);
    QwenASRStream::Init(env, exports);
    return exports;
}

NODE_API_MODULE(whisper, Init)