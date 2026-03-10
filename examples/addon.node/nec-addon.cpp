#include <napi.h>
#include "whisper.h"

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <mutex>
#include <iostream>

// NEC special token IDs
static int get_nec_token_ec(whisper_context* ctx) {
    return whisper_model_n_vocab(ctx) - 4;  // <EC>
}

static int get_nec_token_sop(whisper_context* ctx) {
    return whisper_model_n_vocab(ctx) - 3;  // <SOP>
}

static int get_nec_token_empty(whisper_context* ctx) {
    return whisper_model_n_vocab(ctx) - 2;  // <empty>
}

static int get_nec_token_sep(whisper_context* ctx) {
    return whisper_model_n_vocab(ctx) - 1;  // |||
}

// Tokenize text
static void tokenize_append(whisper_context* ctx, const std::string& text, 
                            std::vector<whisper_token>& tokens) {
    if (text.empty()) return;
    
    std::vector<whisper_token> tmp(text.length() * 2 + 16);
    int n = whisper_tokenize(ctx, text.c_str(), tmp.data(), (int)tmp.size());
    if (n > 0) {
        tokens.insert(tokens.end(), tmp.begin(), tmp.begin() + n);
    }
}

// Build NEC decoder input tokens for a SINGLE candidate
// Format: <|startoftranscript|> candidate <EC> transcript <SOP>
static std::vector<whisper_token> build_nec_pipeline_tokens(
    whisper_context* ctx,
    const std::string& candidate,
    const std::string& transcript
) {
    std::vector<whisper_token> tokens;
    
    // <|startoftranscript|>
    whisper_token sot = whisper_token_sot(ctx);
    tokens.push_back(sot);
    
    // NEC tokens
    const int token_ec = get_nec_token_ec(ctx);
    const int token_sop = get_nec_token_sop(ctx);
    
    // Candidate
    tokenize_append(ctx, candidate, tokens);
    
    // Space before <EC> (to match training)
    std::vector<whisper_token> space_tok(4);
    int n_space = whisper_tokenize(ctx, " ", space_tok.data(), 4);
    if (n_space > 0) {
        tokens.push_back(space_tok[0]);
    }
    
    // <EC>
    tokens.push_back(token_ec);
    
    // Transcript
    tokenize_append(ctx, " " + transcript, tokens);
    
    // Space before <SOP>
    if (n_space > 0) {
        tokens.push_back(space_tok[0]);
    }
    
    // <SOP>
    tokens.push_back(token_sop);
    
    return tokens;
}

// Greedy sampling
static whisper_token sample_greedy(float* logits, int n_vocab) {
    whisper_token best = 0;
    float best_val = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

// Parse corrections
static std::vector<std::string> parse_corrections(const std::string& output) {
    std::vector<std::string> corrections;
    std::string remaining = output;
    
    size_t pos;
    size_t start = remaining.find_first_not_of(" \t\n\r");
    size_t end = remaining.find_last_not_of(" \t\n\r");
    if (start != std::string::npos && end != std::string::npos) {
        corrections.push_back(remaining.substr(start, end - start + 1));
    } else {
        corrections.push_back("");
    }
    
    return corrections;
}

// Apply SINGLE correction
static std::string apply_single_correction(
    const std::string& transcript,
    const std::string& candidate,
    const std::string& correction
) {
    if (correction == "<empty>" || correction.empty()) return transcript;
    
    std::string result = transcript;
    size_t pos = result.find(correction);
    if (pos != std::string::npos) {
        result.replace(pos, correction.length(), candidate);
    }
    return result;
}

// --- NEC Class ---

class NEC : public Napi::ObjectWrap<NEC> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    NEC(const Napi::CallbackInfo& info);
    ~NEC();

    whisper_context* GetContext() const { return m_ctx; }

private:
    Napi::Value Correct(const Napi::CallbackInfo& info);
    Napi::Value Free(const Napi::CallbackInfo& info);

    whisper_context* m_ctx = nullptr;
    std::string m_model_path;
};

// --- Worker ---

class NECPipelineWorker : public Napi::AsyncWorker {
public:
    NECPipelineWorker(
        Napi::Function& callback,
        whisper_context* ctx,
        std::vector<float> pcmf32,
        std::string transcript,
        std::vector<std::string> candidates,
        int n_threads,
        int max_tokens,
        bool debug
    ) : Napi::AsyncWorker(callback),
        m_ctx(ctx),
        m_pcmf32(std::move(pcmf32)),
        m_initial_transcript(std::move(transcript)),
        m_candidates(std::move(candidates)),
        m_n_threads(n_threads),
        m_max_tokens(max_tokens),
        m_debug(debug) {}

    void Execute() override {
        if (!m_ctx) {
            SetError("Model context is null");
            return;
        }

        // Initialize state for thread-safe inference
        whisper_state* wstate = whisper_init_state(m_ctx);
        if (!wstate) {
            SetError("Failed to initialize whisper state");
            return;
        }

        m_current_transcript = m_initial_transcript;
        int n_vocab = whisper_model_n_vocab(m_ctx);
        whisper_token eot = whisper_token_eot(m_ctx);

        // 1. Encode Audio (ONCE)
        // Use whisper_pcm_to_mel_with_state + whisper_encode_with_state
        if (m_debug) fprintf(stderr, "[NEC] Audio size: %zu, Encoding...\n", m_pcmf32.size());
        
        if (whisper_pcm_to_mel_with_state(m_ctx, wstate, m_pcmf32.data(), (int)m_pcmf32.size(), m_n_threads) != 0) {
            SetError("Failed to compute mel spectrogram");
            whisper_free_state(wstate);
            return;
        }
        
        if (whisper_encode_with_state(m_ctx, wstate, 0, m_n_threads) != 0) {
            SetError("Failed to encode audio");
            whisper_free_state(wstate);
            return;
        }

        // 2. Pivot Pipeline: Iterate through candidates
        for (const auto& candidate : m_candidates) {
            if (m_debug) fprintf(stderr, "[NEC] Processing candidate: %s\n", candidate.c_str());

            // Build decoder tokens: <SOT> candidate <EC> current_transcript <SOP>
            std::vector<whisper_token> prompt_tokens = build_nec_pipeline_tokens(
                m_ctx, candidate, m_current_transcript
            );

            // Run decoder with prompt
            // Note: input is empty tokens array because we provided prompt? 
            // Run decoder with prompt
            if (whisper_decode_with_state(m_ctx, wstate, prompt_tokens.data(), (int)prompt_tokens.size(), 0, m_n_threads) != 0) {
                SetError("Failed to decode prompt");
                whisper_free_state(wstate);
                return;
            }

            // Autoregressive generation
            std::string generated_text;
            std::vector<whisper_token> generated_tokens;
            int n_past = (int)prompt_tokens.size();
            
            // Logits logic
            int last_token_idx = (int)prompt_tokens.size() - 1;

            for (int k = 0; k < m_max_tokens; k++) {
                float* logits = whisper_get_logits_from_state(wstate);
                if (!logits) break;

                float* target_logits;
                if (k == 0) {
                    target_logits = logits + last_token_idx * n_vocab;
                } else {
                    target_logits = logits;
                }

                whisper_token next = sample_greedy(target_logits, n_vocab);
                
                if (next == eot) break;

                generated_tokens.push_back(next);
                const char* s = whisper_token_to_str(m_ctx, next);
                if (s) generated_text += s;

                // Decode next
                if (whisper_decode_with_state(m_ctx, wstate, &next, 1, n_past, m_n_threads) != 0) break;
                n_past++;
            }

            // Parse correction
            // Optimization: If generated text is empty or just spaces, skip
            if (m_debug) fprintf(stderr, "      -> Output: '%s'\n", generated_text.c_str());
            
            // Assume the output IS the correction string (or "wrong text" that is in the transcript)
            // Logic: The model is trained to output the substring in transcript that matches the entity.
            // If it outputs something, we replace that something with the candidate.
            
            // Parse
            std::vector<std::string> parts = parse_corrections(generated_text);
            if (!parts.empty() && !parts[0].empty() && parts[0] != "<empty>") {
                std::string to_replace = parts[0];
                if (m_debug) fprintf(stderr, "      -> Replacing '%s' with '%s'\n", to_replace.c_str(), candidate.c_str());
                
                // Update transcript
                m_current_transcript = apply_single_correction(m_current_transcript, candidate, to_replace);
                m_corrections_log.push_back({candidate, to_replace}); // Record what happened
            } else {
                if (m_debug) fprintf(stderr, "      -> No correction\n");
            }
        }

        whisper_free_state(wstate);
    }

    void OnOK() override {
        Napi::HandleScope scope(Env());
        Napi::Object result = Napi::Object::New(Env());
        
        result.Set("success", Napi::Boolean::New(Env(), true));
        result.Set("originalTranscript", Napi::String::New(Env(), m_initial_transcript));
        result.Set("correctedTranscript", Napi::String::New(Env(), m_current_transcript));
        
        Napi::Array logs = Napi::Array::New(Env(), m_corrections_log.size());
        for (size_t i = 0; i < m_corrections_log.size(); i++) {
            Napi::Object item = Napi::Object::New(Env());
            item.Set("candidate", Napi::String::New(Env(), m_corrections_log[i].first));
            item.Set("replaced", Napi::String::New(Env(), m_corrections_log[i].second));
            logs[i] = item;
        }
        result.Set("corrections", logs);
        
        Callback().Call({Env().Null(), result});
    }

private:
    whisper_context* m_ctx;
    std::vector<float> m_pcmf32;
    std::string m_initial_transcript;
    std::string m_current_transcript;
    std::vector<std::string> m_candidates;
    int m_n_threads;
    int m_max_tokens;
    bool m_debug;
    
    std::vector<std::pair<std::string, std::string>> m_corrections_log;
};

// --- NEC Implementation ---

Napi::Object NEC::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "NEC", {
        InstanceMethod("correct", &NEC::Correct),
        InstanceMethod("free", &NEC::Free)
    });



    exports.Set("NEC", func);
    return exports;
}

NEC::NEC(const Napi::CallbackInfo& info) : Napi::ObjectWrap<NEC>(info) {
    Napi::Env env = info.Env();
    
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected model path string").ThrowAsJavaScriptException();
        return;
    }
    
    m_model_path = info[0].As<Napi::String>().Utf8Value();
    bool use_gpu = true; 
    if (info.Length() > 1 && info[1].IsObject()) {
        Napi::Object opts = info[1].As<Napi::Object>();
        if (opts.Has("use_gpu")) use_gpu = opts.Get("use_gpu").As<Napi::Boolean>().Value();
    }
    
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = use_gpu;
    
    m_ctx = whisper_init_from_file_with_params(m_model_path.c_str(), cparams);
    if (!m_ctx) {
        Napi::Error::New(env, "Failed to load NEC model").ThrowAsJavaScriptException();
    }
}

NEC::~NEC() {
    if (m_ctx) {
        whisper_free(m_ctx);
        m_ctx = nullptr;
    }
}

Napi::Value NEC::Correct(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (info.Length() < 3) {
        Napi::TypeError::New(env, "Expected: audio, transcript, candidates, [options], callback").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    
    // Args
    Napi::Float32Array pcm = info[0].As<Napi::Float32Array>();
    std::vector<float> pcmf32(pcm.Data(), pcm.Data() + pcm.ElementLength());
    
    std::string transcript = info[1].As<Napi::String>().Utf8Value();
    
    Napi::Array arr = info[2].As<Napi::Array>();
    std::vector<std::string> candidates;
    for (uint32_t i = 0; i < arr.Length(); i++) {
        Napi::Value val = arr.Get(i);
        if (val.IsString()) candidates.push_back(val.As<Napi::String>().Utf8Value());
    }
    
    // Optional options
    int idx = 3;
    int n_threads = 4;
    int max_tokens = 64;
    bool debug = false;
    
    if (info.Length() > 3 && info[3].IsObject()) {
        Napi::Object opts = info[3].As<Napi::Object>();
        if (opts.Has("n_threads")) n_threads = opts.Get("n_threads").As<Napi::Number>().Int32Value();
        if (opts.Has("max_tokens")) max_tokens = opts.Get("max_tokens").As<Napi::Number>().Int32Value();
        if (opts.Has("debug")) debug = opts.Get("debug").As<Napi::Boolean>().Value();
        idx++;
    }
    
    if (info.Length() <= idx || !info[idx].IsFunction()) {
        Napi::TypeError::New(env, "Callback function required").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    
    Napi::Function callback = info[idx].As<Napi::Function>();
    
    NECPipelineWorker* worker = new NECPipelineWorker(
        callback, m_ctx, std::move(pcmf32), transcript, std::move(candidates),
        n_threads, max_tokens, debug
    );
    worker->Queue();
    
    return env.Undefined();
}

Napi::Value NEC::Free(const Napi::CallbackInfo& info) {
    if (m_ctx) {
        whisper_free(m_ctx);
        m_ctx = nullptr;
    }
    return info.Env().Undefined();
}

// Preserve legacy hook for Init (called from addon.cpp)
Napi::Object InitNEC(Napi::Env env, Napi::Object exports) {
    return NEC::Init(env, exports);
}
