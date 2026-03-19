#include "qwen3_asr.h"
#include "timing.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <map>

namespace qwen3_asr {

static int64_t get_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

Qwen3ASR::Qwen3ASR() = default;
Qwen3ASR::~Qwen3ASR() = default;

bool Qwen3ASR::load_model(const std::string & model_path, bool use_gpu, bool debug) {
    int64_t t_start = get_time_ms();
    
    if (!encoder_.load_model(model_path, use_gpu, debug)) {
        error_msg_ = "Failed to load audio encoder: " + encoder_.get_error();
        return false;
    }
    
    if (!decoder_.load_model(model_path, use_gpu, debug)) {
        error_msg_ = "Failed to load text decoder: " + decoder_.get_error();
        return false;
    }
    
    generate_mel_filters(mel_filters_, QWEN_N_MELS, QWEN_N_FFT, QWEN_SAMPLE_RATE);
    
    model_loaded_ = true;
    
    int64_t t_end = get_time_ms();
    fprintf(stderr, "Model loaded in %lld ms\n", (long long)(t_end - t_start));
    
    return true;
}

transcribe_result Qwen3ASR::transcribe(const std::string & audio_path,
                                        const transcribe_params & params) {
    transcribe_result result;
    
    if (!model_loaded_) {
        result.error_msg = "Model not loaded";
        return result;
    }
    
    std::vector<float> samples;
    int sample_rate;
    
    if (!load_wav(audio_path, samples, sample_rate)) {
        result.error_msg = "Failed to load audio file: " + audio_path;
        return result;
    }
    
    if (sample_rate != QWEN_SAMPLE_RATE) {
        result.error_msg = "Audio must be 16kHz, got " + std::to_string(sample_rate) + " Hz";
        return result;
    }
    
    return transcribe_internal(samples.data(), samples.size(), params);
}

transcribe_result Qwen3ASR::transcribe(const float * samples, int n_samples,
                                        const transcribe_params & params) {
    transcribe_result result;
    
    if (!model_loaded_) {
        result.error_msg = "Model not loaded";
        return result;
    }
    
    return transcribe_internal(samples, n_samples, params);
}

static const std::map<std::string, std::vector<int32_t>> kLanguageTaskTokens = {
    {"english", {11528, 6364, 151704}},
    {"en", {11528, 6364, 151704}},
    {"chinese", {11528, 8453, 151704}},
    {"zh", {11528, 8453, 151704}},
    {"cantonese", {11528, 72366, 2367, 151704}},
    {"yue", {11528, 72366, 2367, 151704}},
    {"japanese", {11528, 10769, 151704}},
    {"ja", {11528, 10769, 151704}},
    {"korean", {11528, 16134, 151704}},
    {"ko", {11528, 16134, 151704}},
    {"french", {11528, 8585, 151704}},
    {"fr", {11528, 8585, 151704}},
    {"german", {11528, 5938, 151704}},
    {"de", {11528, 5938, 151704}},
    {"spanish", {11528, 15154, 151704}},
    {"es", {11528, 15154, 151704}},
    {"portuguese", {11528, 42188, 151704}},
    {"pt", {11528, 42188, 151704}},
    {"italian", {11528, 14811, 151704}},
    {"it", {11528, 14811, 151704}},
    {"russian", {11528, 8522, 151704}},
    {"ru", {11528, 8522, 151704}},
    {"arabic", {11528, 34117, 151704}},
    {"ar", {11528, 34117, 151704}},
    {"hindi", {11528, 43980, 151704}},
    {"hi", {11528, 43980, 151704}},
    {"thai", {11528, 26392, 151704}},
    {"th", {11528, 26392, 151704}},
    {"vietnamese", {11528, 48477, 151704}},
    {"vi", {11528, 48477, 151704}},
    {"indonesian", {11528, 58829, 151704}},
    {"id", {11528, 58829, 151704}},
    {"malay", {11528, 79140, 151704}},
    {"ms", {11528, 79140, 151704}},
    {"turkish", {11528, 23734, 151704}},
    {"tr", {11528, 23734, 151704}},
    {"dutch", {11528, 23234, 151704}},
    {"nl", {11528, 23234, 151704}},
    {"swedish", {11528, 30109, 151704}},
    {"sv", {11528, 30109, 151704}},
    {"danish", {11528, 43680, 151704}},
    {"da", {11528, 43680, 151704}},
    {"finnish", {11528, 57853, 151704}},
    {"fi", {11528, 57853, 151704}},
    {"polish", {11528, 31984, 151704}},
    {"pl", {11528, 31984, 151704}},
    {"czech", {11528, 33150, 151704}},
    {"cs", {11528, 33150, 151704}},
    {"greek", {11528, 17860, 151704}},
    {"el", {11528, 17860, 151704}},
    {"hungarian", {11528, 56769, 151704}},
    {"hu", {11528, 56769, 151704}},
    {"romanian", {11528, 73597, 151704}},
    {"ro", {11528, 73597, 151704}},
    {"persian", {11528, 49861, 151704}},
    {"fa", {11528, 49861, 151704}},
    {"filipino", {11528, 62417, 151704}},
    {"fil", {11528, 62417, 151704}},
    {"macedonian", {11528, 56452, 75491, 151704}},
    {"mk", {11528, 56452, 75491, 151704}},
};

transcribe_result Qwen3ASR::transcribe_internal(const float * samples, int n_samples,
                                                 const transcribe_params & params) {
    transcribe_result result;
    int64_t t_total_start = get_time_ms();
    
    int64_t t_mel_start = get_time_ms();
    MelSpectrogram mel;
    {
        QWEN3_TIMER("mel_spectrogram");
        if (!log_mel_spectrogram(samples, n_samples, mel_filters_, mel, params.n_threads)) {
            result.error_msg = "Failed to compute mel spectrogram";
            return result;
        }
    }
    result.t_mel_ms = get_time_ms() - t_mel_start;
    
    if (params.print_progress) {
        fprintf(stderr, "Mel spectrogram: [%d, %d]\n", mel.n_mel, mel.n_len);
    }
    
    int64_t t_encode_start = get_time_ms();
    std::vector<float> audio_features;
    {
        QWEN3_TIMER("audio_encoding");
        if (!encoder_.encode(mel.data.data(), mel.n_mel, mel.n_len, audio_features)) {
            result.error_msg = "Failed to encode audio: " + encoder_.get_error();
            return result;
        }
    }
    result.t_encode_ms = get_time_ms() - t_encode_start;
    
    const auto & text_hparams = encoder_.get_text_hparams();
    int32_t n_audio_frames = audio_features.size() / text_hparams.hidden_size;
    
    if (params.print_progress) {
        fprintf(stderr, "Audio features: [%d, %d]\n", n_audio_frames, text_hparams.hidden_size);
    }
    
    std::vector<int32_t> input_tokens = build_input_tokens(n_audio_frames, params.language);
    
    if (params.print_progress) {
        fprintf(stderr, "Input tokens: %zu\n", input_tokens.size());
    }
    
    int64_t t_decode_start = get_time_ms();
    std::vector<int32_t> output_tokens;
    if (!decode_greedy(input_tokens, audio_features, n_audio_frames, params, output_tokens)) {
        result.error_msg = "Decoding failed: " + error_msg_;
        return result;
    }
    result.t_decode_ms = get_time_ms() - t_decode_start;
    
    result.tokens = output_tokens;
    
    std::string detected_lang = params.language;
    if (detected_lang == "auto") {
        detected_lang = "";
    }
    
    std::string decoded_text = "";
    bool prefix_removed = false;
    const int32_t separator_token = 151704;

    // 1. FIRST PRIORITY: Match known language tokens from dictionary
    size_t match_len = 0;
    std::string matched_lang_key = "";

    for (const auto& kv : kLanguageTaskTokens) {
        const auto& lang_tokens = kv.second;
        size_t check_len = lang_tokens.size() > 0 ? lang_tokens.size() - 1 : 0;
        
        if (check_len > 0 && output_tokens.size() >= check_len) {
            bool match = true;
            for (size_t i = 0; i < check_len; ++i) {
                if (output_tokens[i] != lang_tokens[i]) {
                    match = false;
                    break;
                }
            }
            if (match && check_len > match_len) {
                match_len = check_len;
                matched_lang_key = kv.first; // e.g., "chinese", "indonesian"
            }
        }
    }

    if (match_len > 0) {
        if (detected_lang.empty()) {
            detected_lang = matched_lang_key;
        }
        size_t erase_len = match_len;
        if (output_tokens.size() > match_len && output_tokens[match_len] == separator_token) {
            erase_len++; 
        }
        output_tokens.erase(output_tokens.begin(), output_tokens.begin() + erase_len);
        prefix_removed = true;
    }

    // 2. Fallback: Search for the 151704 separator if dictionary matching failed
    if (!prefix_removed) {
        auto sep_it = std::find(output_tokens.begin(), output_tokens.end(), separator_token);
        if (sep_it != output_tokens.end() && std::distance(output_tokens.begin(), sep_it) <= 10) {
            std::vector<int32_t> lang_tokens(output_tokens.begin(), sep_it);
            std::string parsed_lang = decoder_.decode_tokens(lang_tokens);
            
            output_tokens.erase(output_tokens.begin(), sep_it + 1);
            prefix_removed = true;
            
            while(!parsed_lang.empty() && (parsed_lang[0] == ' ' || parsed_lang[0] == '\n')) {
                parsed_lang.erase(0, 1);
            }
            if (parsed_lang.size() > 9 && parsed_lang.substr(0, 9) == "language ") {
                parsed_lang = parsed_lang.substr(9);
            } else if (parsed_lang.size() > 8 && parsed_lang.substr(0, 8) == "language") {
                parsed_lang = parsed_lang.substr(8);
            }
            
            if (detected_lang.empty()) {
                detected_lang = parsed_lang; 
            }
        }
    }

    decoded_text = decoder_.decode_tokens(output_tokens);

    // 3. Absolute string fallback
    if (!prefix_removed) {
        std::string text_trim = decoded_text;
        while(!text_trim.empty() && (text_trim[0] == ' ' || text_trim[0] == '\n' || text_trim[0] == '\r')) {
            text_trim.erase(0, 1);
        }
        
        if (text_trim.size() > 8 && text_trim.substr(0, 8) == "language") {
            size_t pos = 8;
            while (pos < text_trim.size() && (text_trim[pos] == ' ' || text_trim[pos] == ':')) pos++;
            
            size_t lang_start = pos;
            while (pos < text_trim.size() && 
                  ((text_trim[pos] >= 'a' && text_trim[pos] <= 'z') || 
                   (text_trim[pos] >= 'A' && text_trim[pos] <= 'Z'))) {
                pos++;
            }
            
            if (pos > lang_start) {
                if (detected_lang.empty()) {
                    detected_lang = text_trim.substr(lang_start, pos - lang_start);
                }
                text_trim.erase(0, pos);
                
                while(!text_trim.empty() && (text_trim[0] == ' ' || text_trim[0] == '\n' || text_trim[0] == '\r')) {
                    text_trim.erase(0, 1);
                }
                if (text_trim.find("<asr_text>") == 0) {
                    text_trim.erase(0, 10);
                }
                while(!text_trim.empty() && (text_trim[0] == ' ' || text_trim[0] == '\n' || text_trim[0] == '\r')) {
                    text_trim.erase(0, 1);
                }
                decoded_text = text_trim;
            }
        }
    }
    if (detected_lang.empty() && params.language == "auto") {
        detected_lang = "auto";
    }

    result.text = decoded_text;
    result.language = detected_lang;
    result.success = true;
    
    result.t_total_ms = get_time_ms() - t_total_start;
    
    if (params.print_timing) {
        fprintf(stderr, "\nTiming:\n");
        fprintf(stderr, "  Mel spectrogram: %lld ms\n", (long long)result.t_mel_ms);
        fprintf(stderr, "  Audio encoding:  %lld ms\n", (long long)result.t_encode_ms);
        fprintf(stderr, "  Text decoding:   %lld ms\n", (long long)result.t_decode_ms);
        fprintf(stderr, "  Total:           %lld ms\n", (long long)result.t_total_ms);
        fprintf(stderr, "  Tokens generated: %zu\n", output_tokens.size());
    }
    
    return result;
}

std::vector<int32_t> Qwen3ASR::build_input_tokens(int32_t n_audio_frames,
                                                   const std::string & language) {
    const auto & cfg = decoder_.get_config();
    
    std::vector<int32_t> tokens;
    tokens.reserve(n_audio_frames + 20);
    
    // Chat template format:
    // <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|>...<|audio_end|><|im_end|>\n<|im_start|>assistant\n
    
    // Token IDs from Qwen3 tokenizer:
    // <|im_start|> = 151644
    // <|im_end|> = 151645
    // system = 8948
    // user = 872
    // assistant = 77091
    // \n = 198
    
    const int32_t im_start = 151644;
    const int32_t im_end = 151645;
    const int32_t system_token = 8948;
    const int32_t user_token = 872;
    const int32_t assistant_token = 77091;
    const int32_t newline = 198;
    
    // <|im_start|>system\n<|im_end|>\n
    tokens.push_back(im_start);
    tokens.push_back(system_token);
    tokens.push_back(newline);
    tokens.push_back(im_end);
    tokens.push_back(newline);
    
    // <|im_start|>user\n
    tokens.push_back(im_start);
    tokens.push_back(user_token);
    tokens.push_back(newline);
    
    // <|audio_start|><|audio_pad|>...<|audio_end|>
    tokens.push_back(cfg.audio_start_token_id);
    for (int32_t i = 0; i < n_audio_frames; ++i) {
        tokens.push_back(cfg.audio_pad_token_id);
    }
    tokens.push_back(cfg.audio_end_token_id);
    
    // <|im_end|>\n<|im_start|>assistant\n
    tokens.push_back(im_end);
    tokens.push_back(newline);
    tokens.push_back(im_start);
    tokens.push_back(assistant_token);
    tokens.push_back(newline);
    
    if (!language.empty()) {
      std::string lang_lower = language;
      std::transform(lang_lower.begin(), lang_lower.end(), lang_lower.begin(),
                     ::tolower);
      auto it = kLanguageTaskTokens.find(lang_lower);
      if (it != kLanguageTaskTokens.end()) {
        tokens.insert(tokens.end(), it->second.begin(), it->second.end());
      }
    }
    
    return tokens;
}

bool Qwen3ASR::decode_greedy(const std::vector<int32_t> & input_tokens,
                              const std::vector<float> & audio_features,
                              int32_t n_audio_frames,
                              const transcribe_params & params,
                              std::vector<int32_t> & output_tokens) {
    const auto & cfg = decoder_.get_config();
    
    int32_t n_ctx_needed = input_tokens.size() + params.max_tokens;
    if (!decoder_.init_kv_cache(n_ctx_needed)) {
        error_msg_ = "Failed to initialize KV cache: " + decoder_.get_error();
        return false;
    }
    
    std::vector<float> logits;
    
    // Audio pad tokens start after: <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>
    // That's 8 tokens before the first audio_pad
    int32_t audio_start_pos = 9;
    
    {
        QWEN3_TIMER("decode.initial_forward");
        if (!decoder_.forward_with_audio(
                input_tokens.data(), input_tokens.size(),
                audio_features.data(), n_audio_frames,
                audio_start_pos, 0, logits)) {
            error_msg_ = "Initial forward pass failed: " + decoder_.get_error();
            return false;
        }
    }
    
    int32_t vocab_size = cfg.vocab_size;
    int32_t n_input = input_tokens.size();
    
    int32_t next_token = sample_greedy(logits.data(), vocab_size);
    
    output_tokens.clear();
    output_tokens.push_back(next_token);
    
    if (progress_callback_) {
        progress_callback_(1, params.max_tokens);
    }
    
    if (params.print_progress && output_tokens.size() % 10 == 0) {
        fprintf(stderr, "Token 1: id=%d, text='%s'\\n", next_token, decoder_.decode_token(next_token).c_str());
    }
    
    int32_t n_past = n_input;
    
    while (next_token != cfg.eos_token_id && 
           (int32_t)output_tokens.size() < params.max_tokens) {
        
        std::vector<int32_t> single_token = {next_token};
        
        {
            QWEN3_TIMER("decode.token");
            if (!decoder_.forward(single_token.data(), 1, n_past, logits)) {
                error_msg_ = "Forward pass failed at token " + 
                             std::to_string(output_tokens.size()) + ": " + decoder_.get_error();
                return false;
            }
        }
        
        next_token = sample_greedy(logits.data(), vocab_size);
        output_tokens.push_back(next_token);
        
        n_past += 1;
        
        if (progress_callback_) {
            progress_callback_(output_tokens.size(), params.max_tokens);
        }
        
        if (params.print_progress && output_tokens.size() % 10 == 0) {
            fprintf(stderr, "Generated %zu tokens...\n", output_tokens.size());
        }
    }
    
    if (output_tokens.back() == cfg.eos_token_id) {
        output_tokens.pop_back();
    }
    
    return true;
}

int32_t Qwen3ASR::sample_greedy(const float * logits, int32_t vocab_size) {
    int32_t max_idx = 0;
    float max_val = logits[0];
    
    for (int32_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

void Qwen3ASR::set_progress_callback(progress_callback_t callback) {
    progress_callback_ = std::move(callback);
}

bool load_audio_file(const std::string & path, std::vector<float> & samples, int & sample_rate) {
    return load_wav(path, samples, sample_rate);
}

} // namespace qwen3_asr
