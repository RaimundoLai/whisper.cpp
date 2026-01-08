#include "whisper.h"

#include <emscripten.h>
#include <emscripten/bind.h>
#include <iostream>
#include <vector>
#include <string>
#include <thread>

std::thread g_worker;

// Checks if a string is a sequence of one or more valid UTF-8 characters
bool is_valid_utf8(const std::string &s) {
    size_t i = 0;
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

        for (size_t j = 1; j < (size_t)len; ++j) {
            if ((s[i + j] & 0xC0) != 0x80) {
                return false; // Invalid subsequent byte
            }
        }
        i += len;
    }
    return true;
}

// Helper function to convert string to hex for UTF-8 handling
std::string string_to_hex(const char* text) {
    if (text == nullptr) {
        return "";
    }
    std::string hex_stream;
    for (size_t i = 0; text[i] != '\0'; ++i) {
        char buf[3];
        snprintf(buf, sizeof(buf), "%02x", static_cast<unsigned char>(text[i]));
        hex_stream += buf;
    }
    return hex_stream;
}

// Helper function to extract complete UTF-8 characters from hex string
// Returns the extracted text and updates consumed_len
std::string extract_complete_utf8_from_hex(const std::string& combined_hex, size_t& consumed_len) {
    std::string complete_text;
    consumed_len = 0;
    
    while (consumed_len + 2 <= combined_hex.length()) {
        int char_length = 0;
        unsigned int first_byte;
        char hex_pair[3] = {combined_hex[consumed_len], combined_hex[consumed_len + 1], '\0'};
        first_byte = (unsigned int)strtol(hex_pair, nullptr, 16);
        
        if ((first_byte & 0x80) == 0) { char_length = 2; }
        else if ((first_byte & 0xE0) == 0xC0) { char_length = 4; }
        else if ((first_byte & 0xF0) == 0xE0) { char_length = 6; }
        else if ((first_byte & 0xF8) == 0xF0) { char_length = 8; }
        else {
            consumed_len += 2;
            continue;
        }
        
        if (consumed_len + char_length <= combined_hex.length()) {
            std::string hex_char = combined_hex.substr(consumed_len, char_length);
            std::string byte_str;
            bool conversion_ok = true;
            
            for (size_t j = 0; j < hex_char.length(); j += 2) {
                char pair[3] = {hex_char[j], hex_char[j + 1], '\0'};
                unsigned int byte = (unsigned int)strtol(pair, nullptr, 16);
                byte_str += static_cast<char>(byte);
            }
            
            if (conversion_ok) {
                bool valid_utf8 = true;
                if (char_length > 2) {
                    for (size_t k = 1; k < byte_str.length(); ++k) {
                        if ((static_cast<unsigned char>(byte_str[k]) & 0xC0) != 0x80) {
                            valid_utf8 = false;
                            break;
                        }
                    }
                }
                if (valid_utf8) {
                    complete_text += byte_str;
                    consumed_len += char_length;
                } else {
                    consumed_len += 2;
                }
            } else {
                consumed_len += 2;
            }
        } else {
            break;
        }
    }
    return complete_text;
}

static inline int mpow2(int n) {
    int p = 1;
    while (p <= n) p *= 2;
    return p / 2;
}

// 500 -> 00:05.000
// 6000 -> 01:00.000
std::string to_timestamp(int64_t t, bool comma) {
    int64_t msec = t * 10;
    int64_t hr = msec / (1000 * 60 * 60);
    msec = msec - hr * (1000 * 60 * 60);
    int64_t min = msec / (1000 * 60);
    msec = msec - min * (1000 * 60);
    int64_t sec = msec / 1000;
    msec = msec - sec * 1000;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d%s%03d", (int) hr, (int) min, (int) sec, comma ? "," : ".", (int) msec);

    return std::string(buf);
}

static bool output_json(
        struct whisper_context * ctx,
        bool    final,
        int     first_segment,
        int     n_segments) {

    // Helper function: An enhanced JSON string escaping and sanitizing function
    // This version filters out invalid UTF-8 sequences and control characters
    auto escape_json_string = [](const char *s) -> std::string {
    std::string escaped;
    if (s == nullptr) {
        return "";
    }

    const unsigned char *p = reinterpret_cast<const unsigned char *>(s);

    while (*p) {
        unsigned char c = *p;
        
        // Handle special JSON characters that need to be escaped
        switch (c) {
            case '"':  escaped += "\\\""; p++; break;
            case '\\': escaped += "\\\\"; p++; break;
            case '\b': escaped += "\\b";  p++; break;
            case '\f': escaped += "\\f";  p++; break;
            case '\n': escaped += "\\n";  p++; break;
            case '\r': escaped += "\\r";  p++; break;
            case '\t': escaped += "\\t";  p++; break;
            default:
                if (c < 32) {
                    // Ignore other control characters
                    p++;
                } else if (c < 128) {
                    // Standard ASCII characters
                    escaped += c;
                    p++;
                } else {
                    // UTF-8 multi-byte character handling
                    int utf8_len = 0;
                    
                    // Determine the byte length of the UTF-8 character
                    if ((c & 0x80) == 0) {
                        // ASCII (shouldn't get here)
                        utf8_len = 1;
                    } else if ((c & 0xE0) == 0xC0) {
                        // 2-byte UTF-8
                        utf8_len = 2;
                    } else if ((c & 0xF0) == 0xE0) {
                        // 3-byte UTF-8 (Chinese characters are usually in this range)
                        utf8_len = 3;
                    } else if ((c & 0xF8) == 0xF0) {
                        // 4-byte UTF-8
                        utf8_len = 4;
                    } else {
                        // Invalid UTF-8 start byte, skip it
                        p++;
                        continue;
                    }
                    
                    // Verify that the subsequent bytes are valid UTF-8 continuation bytes
                    bool valid_utf8 = true;
                    for (int i = 1; i < utf8_len; i++) {
                        if (p[i] == 0 || (p[i] & 0xC0) != 0x80) {
                            valid_utf8 = false;
                            break;
                        }
                    }
                    
                    if (valid_utf8) {
                        // Add the complete UTF-8 character
                        for (int i = 0; i < utf8_len; i++) {
                            escaped += p[i];
                        }
                        p += utf8_len;
                    } else {
                        // Invalid UTF-8 sequence, skip this byte
                        p++;
                    }
                }
                break;
        }
    }
    return escaped;
};

    int indent = 0;
    std::string output;

    auto doindent = [&]() {
        for (int i = 0; i < indent; i++) output += "\t";
    };

    auto start_arr = [&](const char *name) {
        doindent();
        output += "\"" + std::string(name) + "\": [";
        indent++;
    };

    auto end_arr = [&](bool end) {
        indent--;
        doindent();
        output += (end ? "]" : "],");
    };

    auto start_obj = [&](const char *name) {
        doindent();
        if (name) {
            output += "\"" + std::string(name) + "\": {";
        } else {
            output += "{";
        }
        indent++;
    };

    auto end_obj = [&](bool end) {
        indent--;
        doindent();
        output += (end ? "}" : "},");
    };

    auto start_value = [&](const char *name) {
        doindent();
        output += "\"" + std::string(name) + "\": ";
    };

    auto value_s = [&](const char *name, const char *val, bool end) {
        start_value(name);
        std::string val_escaped = escape_json_string(val);
        output += "\"" + val_escaped + (end ? "\"" : "\",");
    };

    auto end_value = [&](bool end) {
        output += (end ? "" : ",");
    };

    auto value_i = [&](const char *name, const int64_t val, bool end) {
        start_value(name);
        output += std::to_string(val);
        end_value(end);
    };

    auto value_f = [&](const char *name, const float val, bool end) {
        start_value(name);
        output += std::to_string(val);
        end_value(end);
    };

    auto value_b = [&](const char *name, const bool val, bool end) {
        start_value(name);
        output += (val ? "true" : "false");
        end_value(end);
    };

    auto times_o = [&](int64_t t0, int64_t t1, bool end) {
        start_obj("timestamps");
        value_s("from", to_timestamp(t0, true).c_str(), false);
        value_s("to", to_timestamp(t1, true).c_str(), true);
        end_obj(false);
        start_obj("offsets");
        value_i("from", t0 * 10, false);
        value_i("to", t1 * 10, true);
        end_obj(end);
    };

    start_obj(nullptr);
    value_s("systeminfo", whisper_print_system_info(), false);
    start_obj("model");
    value_s("type", whisper_model_type_readable(ctx), false);
    value_b("multilingual", whisper_is_multilingual(ctx), false);
    value_i("vocab", whisper_model_n_vocab(ctx), false);
    start_obj("audio");
    value_i("ctx", whisper_model_n_audio_ctx(ctx), false);
    value_i("state", whisper_model_n_audio_state(ctx), false);
    value_i("head", whisper_model_n_audio_head(ctx), false);
    value_i("layer", whisper_model_n_audio_layer(ctx), true);
    end_obj(false);
    start_obj("text");
    value_i("ctx", whisper_model_n_text_ctx(ctx), false);
    value_i("state", whisper_model_n_text_state(ctx), false);
    value_i("head", whisper_model_n_text_head(ctx), false);
    value_i("layer", whisper_model_n_text_layer(ctx), true);
    end_obj(false);
    value_i("mels", whisper_model_n_mels(ctx), false);
    value_i("ftype", whisper_model_ftype(ctx), true);
    end_obj(false);
    start_obj("result");
    value_s("language", whisper_lang_str(whisper_full_lang_id(ctx)), true);
    end_obj(false);
    start_arr("transcription");

    // For cross-segment UTF-8 handling (similar to addon.cpp)
    std::string hex_buffer;
    int64_t pending_t0 = -1;
    
    // First pass: collect all segments with UTF-8 handling
    struct processed_segment {
        std::string text;
        int64_t t0;
        int64_t t1;
        int original_index; // For token lookup
    };
    std::vector<processed_segment> processed_segments;
    
    for (int i = first_segment; i < n_segments; ++i) {
        const char * text = whisper_full_get_segment_text(ctx, i);
        int64_t current_t0 = whisper_full_get_segment_t0(ctx, i);
        const int64_t current_t1 = whisper_full_get_segment_t1(ctx, i);
        if (current_t0 < 0) current_t0 = 0;
        
        std::string current_text_hex = string_to_hex(text);
        std::string combined_hex = hex_buffer + current_text_hex;
        hex_buffer.clear();
        
        size_t consumed_len = 0;
        std::string complete_text = extract_complete_utf8_from_hex(combined_hex, consumed_len);
        
        hex_buffer = combined_hex.substr(consumed_len);
        
        if (!complete_text.empty()) {
            processed_segment seg;
            seg.text = complete_text;
            seg.t0 = (pending_t0 >= 0) ? pending_t0 : current_t0;
            seg.t1 = current_t1;
            seg.original_index = i;
            processed_segments.push_back(seg);
        }
        
        if (!hex_buffer.empty()) {
            if (pending_t0 < 0) {
                pending_t0 = current_t0;
            }
        } else {
            pending_t0 = -1;
        }
    }
    
    // Handle any remaining hex_buffer
    if (!hex_buffer.empty()) {
        size_t consumed_len = 0;
        std::string remaining_text = extract_complete_utf8_from_hex(hex_buffer, consumed_len);
        if (!remaining_text.empty() && !processed_segments.empty()) {
            processed_segments.back().text += remaining_text;
        }
    }
    
    // Now output all processed segments
    for (size_t seg_idx = 0; seg_idx < processed_segments.size(); ++seg_idx) {
        const auto& seg = processed_segments[seg_idx];
        
        start_obj(nullptr);
        times_o(seg.t0, seg.t1, false);
        value_s("text", seg.text.c_str(), false);

        start_arr("tokens");
        const int n = whisper_full_n_tokens(ctx, seg.original_index);
        int j = 0;
        int output_token_count = 0;
        while (j < n) {
            auto token_first = whisper_full_get_token_data(ctx, seg.original_index, j);
            std::string current_text = whisper_token_to_str(ctx, token_first.id);
            
            if (is_valid_utf8(current_text)) {
                // Token is a valid UTF-8 character, add it directly
                bool is_last = (j == n - 1);
                start_obj(nullptr);
                value_s("text", current_text.c_str(), false);
                if(token_first.t0 > -1 && token_first.t1 > -1) {
                    times_o(token_first.t0, token_first.t1, false);
                }
                value_i("id", token_first.id, false);
                value_f("p", token_first.p, false);
                value_f("t_dtw", token_first.t_dtw, true);
                end_obj(is_last);
                output_token_count++;
                j++;
            } else {
                // Token is an incomplete UTF-8, start merging
                std::string merged_text = current_text;
                int64_t start_time = token_first.t0;
                int64_t end_time = token_first.t1;
                int k = j + 1;
                
                while (k < n) {
                    auto token_next = whisper_full_get_token_data(ctx, seg.original_index, k);
                    merged_text += whisper_token_to_str(ctx, token_next.id);
                    
                    if (is_valid_utf8(merged_text)) {
                        // Becomes a valid UTF-8 character after merging
                        end_time = token_next.t1;
                        break;
                    }
                    k++;
                }
                
                bool is_last = (k >= n - 1);
                start_obj(nullptr);
                value_s("text", merged_text.c_str(), false);
                if(start_time > -1 && end_time > -1) {
                    times_o(start_time, end_time, false);
                }
                value_i("id", token_first.id, false);
                value_f("p", token_first.p, false);
                value_f("t_dtw", token_first.t_dtw, true);
                end_obj(is_last);
                output_token_count++;
                
                j = k + 1; // Update the index of the main loop
            }
        }
        end_arr(true);

        end_obj(seg_idx == (processed_segments.size() - 1));
    }

    end_arr(true);
    end_obj(true);

    if (final) {
        printf("whisper_final:%s\n", output.c_str());
    } else {
        printf("whisper_update:%s\n", output.c_str());
    }

    return true;
}


void whisper_print_segment_callback(struct whisper_context * ctx, struct whisper_state * /*state*/, int n_new, void * user_data) {
    const int n_segments = whisper_full_n_segments(ctx);
    const int s0 = n_segments - n_new;

    if (s0 == 0) {
        printf("\n");
    }

    output_json(ctx, false, s0, n_segments);
}

// Define the progress callback function
void progress_callback(struct whisper_context * ctx, struct whisper_state * state, int progress, void * user_data) {
    printf("whisper_progress:%d%%\n", progress);
}

std::vector<struct whisper_context *> g_contexts(1, nullptr);


EMSCRIPTEN_BINDINGS(whisper) {
    emscripten::function("full_default", emscripten::optional_override([](const std::string & path_model, const emscripten::val & audio, const std::string & model, const std::string & lang, int nthreads, bool translate, int max_len) {
        if (g_contexts[0] != nullptr) {
            printf("whisper_busy:\n");
            return 0;
        }

        g_contexts[0] = whisper_init_from_file_with_params(path_model.c_str(), whisper_context_default_params());

        struct whisper_full_params params = whisper_full_default_params(whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY);

        std::vector<float> pcmf32;

        params.print_realtime    = false;
        params.new_segment_callback = whisper_print_segment_callback;
        params.print_progress    = false;
        params.print_timestamps = false;
        params.print_special     = false;
        params.translate         = translate;
        params.token_timestamps = true;
        params.language          = lang.c_str();
        params.n_threads         = std::min(nthreads, std::min(16, mpow2(std::thread::hardware_concurrency())));
        params.offset_ms         = 0;
        params.progress_callback = progress_callback;
        params.max_len           = max_len;
        params.split_on_word     = false;

        const int n = audio["length"].as<int>();

        emscripten::val heap = emscripten::val::module_property("HEAPU8");
        emscripten::val memory = heap["buffer"];

        pcmf32.resize(n);

        emscripten::val memoryView = audio["constructor"].new_(memory, reinterpret_cast<uintptr_t>(pcmf32.data()), n);
        memoryView.call<void>("set", audio);

        {
            printf("system_info: n_threads = %d / %d | %s\n",
                   params.n_threads, std::thread::hardware_concurrency(), whisper_print_system_info());

            printf("%s: processing %d samples, %.1f sec, %d threads, lang = %s, task = %s ...\n",
                   __func__, int(pcmf32.size()), float(pcmf32.size()) / WHISPER_SAMPLE_RATE,
                   params.n_threads,
                   params.language,
                   params.translate ? "translate" : "transcribe");

            printf("\n");
        }

        {
            g_worker = std::thread([params, pcm = std::move(pcmf32)]() {
                whisper_reset_timings(g_contexts[0]);
                whisper_full(g_contexts[0], params, pcm.data(), pcm.size());
                const int n_segments = whisper_full_n_segments(g_contexts[0]);
                output_json(g_contexts[0], true, 0, n_segments);
                whisper_free(g_contexts[0]);
                g_contexts[0] = nullptr;
            });
        }

        return 0;
    }));
}