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

std::string char_to_hex(char c) {
    std::stringstream ss;
    ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(static_cast<unsigned char>(c));
    return ss.str();
}


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

struct whisper_params {
    int32_t n_threads    = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_processors = 1;
    int32_t offset_t_ms  = 0;
    int32_t offset_n     = 0;
    int32_t duration_ms  = 0;
    int32_t max_context  = -1;
    int32_t max_len      = 0;
    int32_t best_of      = 5;
    int32_t beam_size    = -1;
    int32_t audio_ctx    = 0;

    float word_thold    = 0.01f;
    float entropy_thold = 2.4f;
    float logprob_thold = -1.0f;

    bool translate      = false;
    bool diarize        = false;
    bool output_txt     = false;
    bool output_vtt     = false;
    bool output_srt     = false;
    bool output_wts     = false;
    bool output_csv     = false;
    bool print_special  = false;
    bool print_colors   = false;
    bool print_progress = false;
    bool no_timestamps  = false;
    bool no_prints      = false;
    bool use_gpu        = true;
    bool flash_attn     = false;
    bool comma_in_time  = true;

    std::string language = "en";
    std::string prompt;
    std::string model    = "../../ggml-large.bin";

    std::vector<std::string> fname_inp = {};
    std::vector<std::string> fname_out = {};

    std::vector<float> pcmf32 = {}; // mono-channel F32 PCM
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

    // print the last n_new segments
    const int s0 = n_segments - n_new;

    if (s0 == 0) {
        printf("\n");
    }

    for (int i = s0; i < n_segments; i++) {
        if (!params.no_timestamps || params.diarize) {
            t0 = whisper_full_get_segment_t0(ctx, i);
            t1 = whisper_full_get_segment_t1(ctx, i);
        }

        if (!params.no_timestamps) {
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

            //printf("is0 = %lld, is1 = %lld, energy0 = %f, energy1 = %f, %s\n", is0, is1, energy0, energy1, speaker.c_str());
        }

        // colorful print bug
        //
        const char * text = whisper_full_get_segment_text(ctx, i);
        printf("%s%s", speaker.c_str(), text);


        // with timestamps or speakers: each segment on new line
        if (!params.no_timestamps || params.diarize) {
            printf("\n");
        }

        fflush(stdout);
    }
}

void cb_log_disable(enum ggml_log_level, const char *, void *) {}

class ProgressWorker : public Napi::AsyncWorker {
 public:
    ProgressWorker(Napi::Function& callback, whisper_params params, Napi::Function progress_callback, Napi::Env env)
        : Napi::AsyncWorker(callback), params(params), env(env) {
        // Create thread-safe function
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
            // Make sure to release the thread-safe function on destruction
            tsfn.Release();
        }
    }

    void Execute() override {
        // Use custom run function with progress callback support
        run_with_progress(params, result);
    }

    void OnOK() override {
        Napi::HandleScope scope(Env());

        // Create the segments array
        Napi::Array segments_array = Napi::Array::New(Env(), result.size());
        for (uint64_t i = 0; i < result.size(); ++i) {
            Napi::Object segment = Napi::Object::New(Env());
            segment.Set("start", Napi::String::New(Env(), result[i][0]));
            segment.Set("end", Napi::String::New(Env(), result[i][1]));
            segment.Set("text", Napi::String::New(Env(), result[i][2]));
            segments_array[i] = segment;
        }

        // Create the final result object
        Napi::Object final_res = Napi::Object::New(Env());
        final_res.Set("segments", segments_array);
        final_res.Set("language", Napi::String::New(Env(), this->detected_language)); 

        Callback().Call({Env().Null(), final_res});
    }

    // Progress callback function - using thread-safe function
    void OnProgress(int progress) {
        if (tsfn) {
            // Use thread-safe function to call JavaScript callback
            auto callback = [progress](Napi::Env env, Napi::Function jsCallback) {
                jsCallback.Call({Napi::Number::New(env, progress)});
            };
            
            tsfn.BlockingCall(callback);
        }
    }

 private:
    whisper_params params;
    std::vector<std::vector<std::string>> result;
    Napi::Env env;
    Napi::ThreadSafeFunction tsfn;
    std::string detected_language = "auto";

    // Custom run function with progress callback support
    int run_with_progress(whisper_params &params, std::vector<std::vector<std::string>> &result) {
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

        // whisper init
        struct whisper_context_params cparams = whisper_context_default_params();
        cparams.use_gpu = params.use_gpu;
        cparams.flash_attn = params.flash_attn;
        struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);

        if (ctx == nullptr) {
            fprintf(stderr, "error: failed to initialize whisper context\n");
            return 3;
        }

        // If params.pcmf32 provides, set params.fname_inp as "buffer"
        if (!params.pcmf32.empty()) {
            fprintf(stderr, "info: using audio buffer as input\n");
            params.fname_inp.clear();
            params.fname_inp.emplace_back("buffer");
        }

        for (int f = 0; f < (int) params.fname_inp.size(); ++f) {
            const auto fname_inp = params.fname_inp[f];
            const auto fname_out = f < (int)params.fname_out.size() && !params.fname_out[f].empty() ? params.fname_out[f] : params.fname_inp[f];

            std::vector<float> pcmf32; // mono-channel F32 PCM
            std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM

            // If params.pcmf32 is empty, read input audio file
            if (params.pcmf32.empty()) {
                if (!::read_audio_data(fname_inp, pcmf32, pcmf32s, params.diarize)) {
                    fprintf(stderr, "error: failed to read audio file '%s'\n", fname_inp.c_str());
                    continue;
                }
            } else {
                pcmf32 = params.pcmf32;
            }

            // Print system info
            if (!params.no_prints) {
                fprintf(stderr, "\n");
                fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                        params.n_threads*params.n_processors, std::thread::hardware_concurrency(), whisper_print_system_info());
            }

            // Print processing info
            if (!params.no_prints) {
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

            // Run inference
            {
                whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

                wparams.strategy = params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;

                wparams.print_realtime   = false;
                wparams.print_progress   = params.print_progress;
                wparams.print_timestamps = !params.no_timestamps;
                wparams.print_special    = params.print_special;
                wparams.translate        = params.translate;
                wparams.language         = params.language.c_str();
                wparams.n_threads        = params.n_threads;
                wparams.n_max_text_ctx   = params.max_context >= 0 ? params.max_context : wparams.n_max_text_ctx;
                wparams.offset_ms        = params.offset_t_ms;
                wparams.duration_ms      = params.duration_ms;

                wparams.token_timestamps = params.output_wts || params.max_len > 0;
                wparams.thold_pt         = params.word_thold;
                wparams.entropy_thold    = params.entropy_thold;
                wparams.logprob_thold    = params.logprob_thold;
                wparams.max_len          = params.output_wts && params.max_len == 0 ? 60 : params.max_len;
                wparams.audio_ctx        = params.audio_ctx;

                wparams.greedy.best_of        = params.best_of;
                wparams.beam_search.beam_size = params.beam_size;

                wparams.initial_prompt   = params.prompt.c_str();

                wparams.no_timestamps    = params.no_timestamps;

                whisper_print_user_data user_data = { &params, &pcmf32s };

                // This callback is called for each new segment
                if (!wparams.print_realtime) {
                    wparams.new_segment_callback           = whisper_print_segment_callback;
                    wparams.new_segment_callback_user_data = &user_data;
                }

                // Set progress callback
                wparams.progress_callback = [](struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, int progress, void * user_data) {
                    ProgressWorker* worker = static_cast<ProgressWorker*>(user_data);
                    worker->OnProgress(progress);
                };
                wparams.progress_callback_user_data = this;

                // Abort mechanism example
                {
                    static bool is_aborted = false; // Note: this should be atomic to avoid data races

                    wparams.encoder_begin_callback = [](struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, void * user_data) {
                        bool is_aborted = *(bool*)user_data;
                        return !is_aborted;
                    };
                    wparams.encoder_begin_callback_user_data = &is_aborted;
                }

                if (whisper_full_parallel(ctx, wparams, pcmf32.data(), pcmf32.size(), params.n_processors) != 0) {
                    fprintf(stderr, "failed to process audio\n");
                    // Clean up context before returning error
                    whisper_free(ctx);
                    return 10;
                }
            }
             // --- Get detected language AFTER processing ---
            int lang_id = whisper_full_lang_id(ctx);
            if (lang_id != -1) {
                 this->detected_language = whisper_lang_str(lang_id);
                 if (!params.no_prints) {
                    fprintf(stderr, "%s: detected language: %s\n", __func__, this->detected_language.c_str());
                 }
            } else {
                 this->detected_language = "unknown"; // Or keep "auto" if detection failed
                 if (!params.no_prints) {
                    fprintf(stderr, "%s: language detection failed\n", __func__);
                 }
            }
            // --- End of language detection ---
        } // End of file processing loop

        const int n_segments = whisper_full_n_segments(ctx);
        result.clear(); // Clear result vector before processing segments
        std::string hex_buffer; // Buffer for incomplete hex across segments
        int64_t pending_t0 = -1; // Timestamp of the start of an incomplete segment

        for (int i = 0; i < n_segments; ++i) {
            const char * text = whisper_full_get_segment_text(ctx, i);
            int64_t current_t0 = whisper_full_get_segment_t0(ctx, i);
            const int64_t current_t1 = whisper_full_get_segment_t1(ctx, i);
            if (current_t0 < 0) {
                fprintf(stderr, "Warning: current_t0 is negative, setting to 0\n");
                current_t0 = 0;
            }
            std::string current_text_hex = string_to_hex(text);
            // Prepend buffer from previous segment, if any
            std::string combined_hex = hex_buffer + current_text_hex;
            hex_buffer.clear(); // Clear buffer, it will be repopulated if combined_hex ends incomplete

            std::string complete_text;
            size_t consumed_hex_len = 0;

            // Process combined_hex to find complete UTF-8 characters
            while (consumed_hex_len + 2 <= combined_hex.length()) {
                int char_length = 0; // Length in hex digits (bytes * 2)
                unsigned int first_byte;
                std::stringstream ss_first;
                // Use substr starting from consumed_hex_len
                ss_first << std::hex << combined_hex.substr(consumed_hex_len, 2);
                ss_first >> first_byte;

                // Determine expected UTF-8 char length based on first byte
                if ((first_byte & 0x80) == 0) { char_length = 2; }         // ASCII (1 byte)
                else if ((first_byte & 0xE0) == 0xC0) { char_length = 4; } // 2-byte UTF-8
                else if ((first_byte & 0xF0) == 0xE0) { char_length = 6; } // 3-byte UTF-8
                else if ((first_byte & 0xF8) == 0xF0) { char_length = 8; } // 4-byte UTF-8
                else {
                    // Invalid start byte. Skip this hex pair.
                    fprintf(stderr, "Warning: Invalid UTF-8 start byte encountered: 0x%02x\n", first_byte);
                    consumed_hex_len += 2;
                    continue;
                }

                // Check if we have enough hex digits left for the complete character
                if (consumed_hex_len + char_length <= combined_hex.length()) {
                    std::string hex_char = combined_hex.substr(consumed_hex_len, char_length);
                    std::string byte_str;
                    bool conversion_ok = true;
                    for (size_t j = 0; j < hex_char.length(); j += 2) {
                        unsigned int byte;
                        std::stringstream ss_byte;
                        ss_byte << std::hex << hex_char.substr(j, 2);
                        if (!(ss_byte >> byte)) {
                            fprintf(stderr, "Warning: Hex conversion failed for: %s\n", hex_char.substr(j, 2).c_str());
                            conversion_ok = false;
                            break;
                        }
                        byte_str += static_cast<char>(byte);
                    }

                    if (conversion_ok) {
                         // Basic validation for continuation bytes (optional but good)
                         bool valid_utf8 = true;
                         if (char_length > 2) {
                             for(size_t k = 1; k < byte_str.length(); ++k) {
                                 if ((static_cast<unsigned char>(byte_str[k]) & 0xC0) != 0x80) {
                                     valid_utf8 = false;
                                     fprintf(stderr, "Warning: Invalid UTF-8 continuation byte at pos %zu in char %s\n", k, hex_char.c_str());
                                     break;
                                 }
                             }
                         }
                         if (valid_utf8) {
                            complete_text += byte_str;
                            consumed_hex_len += char_length;
                         } else {
                             // Skip the invalid sequence start byte pair
                             consumed_hex_len += 2;
                         }
                    } else {
                        // Skip the problematic hex pair where conversion failed
                        consumed_hex_len += 2;
                    }
                } else {
                    // Not enough hex digits left for the character indicated by first_byte.
                    // Break the inner loop, the remainder will be put into hex_buffer.
                    break;
                }
            }

            // Store any remaining hex digits in the buffer for the next iteration
            hex_buffer = combined_hex.substr(consumed_hex_len);

            // Decide whether to store the result or wait for the next segment
            if (!hex_buffer.empty()) {
                // Current segment ends with an incomplete character.
                if (pending_t0 < 0) { // If not already tracking an incomplete segment start
                    pending_t0 = current_t0; // Store the start time of this segment
                }
                // If there was text completed *before* the incomplete part, store it now
                // with the correct timestamp, otherwise wait.
                if (!complete_text.empty()) {
                     int64_t start_time = (pending_t0 >= 0 && pending_t0 != current_t0) ? pending_t0 : current_t0;
                     // We use current_t1 because the *completed* text ends here, even if an incomplete char follows.
                     result.emplace_back(std::vector<std::string>{
                         to_timestamp(start_time, params.comma_in_time),
                         to_timestamp(current_t1, params.comma_in_time), // End time is still current segment's end
                         complete_text
                     });
                     // Since we added text, the *next* segment's text should start from pending_t0
                     // which we just set (or kept). Don't reset pending_t0 here.
                }
                // Do not add the incomplete part to result yet.
            } else {
                // Current segment processing finished completely (no leftover hex).
                if (!complete_text.empty()) {
                    // Use pending_t0 if valid, otherwise use current_t0
                    int64_t start_time = (pending_t0 >= 0) ? pending_t0 : current_t0;
                    result.emplace_back(std::vector<std::string>{
                        to_timestamp(start_time, params.comma_in_time),
                        to_timestamp(current_t1, params.comma_in_time),
                        complete_text
                    });
                }
                // Reset pending_t0 as we have completed the sequence (or there was none)
                pending_t0 = -1;
            }
        }

        // After the loop, check if there's anything left in the buffer
        if (!hex_buffer.empty()) {
            fprintf(stderr, "Warning: Transcription ended with incomplete UTF-8 sequence (hex: %s). Discarding.\n", hex_buffer.c_str());
            // Optionally, you could try to decode it leniently or add placeholder text.
            // If pending_t0 is valid here, it means the very last segment(s) were incomplete.
        }


        whisper_print_timings(ctx);
        whisper_free(ctx);

        return 0;
    }
};

Napi::Value whisper(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() <= 0 || !info[0].IsObject()) {
    Napi::TypeError::New(env, "object expected").ThrowAsJavaScriptException();
  }
  whisper_params params;

  Napi::Object whisper_params = info[0].As<Napi::Object>();
  std::string language = whisper_params.Get("language").As<Napi::String>();
  std::string model = whisper_params.Get("model").As<Napi::String>();
  std::string input = whisper_params.Get("fname_inp").As<Napi::String>();
  bool use_gpu = whisper_params.Get("use_gpu").As<Napi::Boolean>();
  bool flash_attn = whisper_params.Get("flash_attn").As<Napi::Boolean>();
  bool no_prints = whisper_params.Get("no_prints").As<Napi::Boolean>();
  bool no_timestamps = whisper_params.Get("no_timestamps").As<Napi::Boolean>();
  int32_t audio_ctx = whisper_params.Get("audio_ctx").As<Napi::Number>();
  bool comma_in_time = whisper_params.Get("comma_in_time").As<Napi::Boolean>();
  int32_t max_len = whisper_params.Get("max_len").As<Napi::Number>();
  
  // Add support for max_context
  int32_t max_context = -1;
  if (whisper_params.Has("max_context") && whisper_params.Get("max_context").IsNumber()) {
    max_context = whisper_params.Get("max_context").As<Napi::Number>();
  }
  
  // support prompt
  std::string prompt = "";
  if (whisper_params.Has("prompt") && whisper_params.Get("prompt").IsString()) {
    prompt = whisper_params.Get("prompt").As<Napi::String>();
  }
  
  // Add support for print_progress
  bool print_progress = false;
  if (whisper_params.Has("print_progress")) {
    print_progress = whisper_params.Get("print_progress").As<Napi::Boolean>();
  }
  // Add support for progress_callback
  Napi::Function progress_callback;
  if (whisper_params.Has("progress_callback") && whisper_params.Get("progress_callback").IsFunction()) {
    progress_callback = whisper_params.Get("progress_callback").As<Napi::Function>();
  }

  Napi::Value pcmf32Value = whisper_params.Get("pcmf32");
  std::vector<float> pcmf32_vec;
  if (pcmf32Value.IsTypedArray()) {
    Napi::Float32Array pcmf32 = pcmf32Value.As<Napi::Float32Array>();
    size_t length = pcmf32.ElementLength();
    pcmf32_vec.reserve(length);
    for (size_t i = 0; i < length; i++) {
      pcmf32_vec.push_back(pcmf32[i]);
    }
  }

  params.language = language;
  params.model = model;
  params.fname_inp.emplace_back(input);
  params.use_gpu = use_gpu;
  params.flash_attn = flash_attn;
  params.no_prints = no_prints;
  params.no_timestamps = no_timestamps;
  params.audio_ctx = audio_ctx;
  params.pcmf32 = pcmf32_vec;
  params.comma_in_time = comma_in_time;
  params.max_len = max_len;
  params.max_context = max_context;
  params.print_progress = print_progress;
  params.prompt = prompt;

  Napi::Function callback = info[1].As<Napi::Function>();
  // Create a new Worker class with progress callback support
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