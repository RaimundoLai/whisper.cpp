#include "napi.h"
#include "common.h"
#include "common-whisper.h"

#include "whisper.h"

#include <string>
#include <thread>
#include <vector>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cfloat>

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

void cb_log_disable(enum ggml_log_level, const char *, void *) {}

struct whisper_result {
    std::vector<segment_data> segments;
    std::string language;
};

class ProgressWorker : public Napi::AsyncWorker {
public:
    ProgressWorker(Napi::Function& callback, whisper_params params, Napi::Function progress_callback, Napi::Env env)
        : Napi::AsyncWorker(callback), params(params), env(env) {
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

        Callback().Call({Env().Null(), final_res});
    }

    void OnProgress(int progress) {
        if (tsfn) {
            auto callback = [progress](Napi::Env env, Napi::Function jsCallback) {
                jsCallback.Call({Napi::Number::New(env, progress)});
            };
            tsfn.BlockingCall(callback);
        }
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
                            td.start_timestamp = to_timestamp(token_data_first.t0, params.comma_in_time);
                            td.end_timestamp = to_timestamp(token_data_first.t1, params.comma_in_time);
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
                            td.start_timestamp = to_timestamp(start_time, params.comma_in_time);
                            td.end_timestamp = to_timestamp(end_time, params.comma_in_time);
                            seg.tokens.push_back(td);

                            j = k + 1; // Update the index of the main loop
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

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    exports.Set(
        Napi::String::New(env, "whisper"),
        Napi::Function::New(env, whisper)
    );
    return exports;
}

NODE_API_MODULE(whisper, Init);