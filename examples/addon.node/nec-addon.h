#pragma once

#include <napi.h>
#include "whisper.h"
#include <vector>
#include <string>

// Forward declarations for ss_model
struct ss_model;
struct ss_model * whisper_ss_init_from_file(const char * path_model);
void whisper_ss_free(struct ss_model * model);

// Forward declaration for SS embedding worker
Napi::Value QueueSSEmbeddingWorkerFromOptions(
    Napi::Env env, 
    Napi::Object options, 
    Napi::Function callback, 
    whisper_context* shared_wctx, 
    struct ss_model* shared_ss_ctx
);

// NEC Class definition
class NEC : public Napi::ObjectWrap<NEC> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    NEC(const Napi::CallbackInfo& info);
    ~NEC();

    whisper_context* GetContext() const { return m_ctx; }
    struct ss_model* GetSSContext() const { return m_ss_ctx; }

    Napi::Value ExtractSSEmbedding(const Napi::CallbackInfo& info);

private:
    Napi::Value Correct(const Napi::CallbackInfo& info);
    Napi::Value Free(const Napi::CallbackInfo& info);

    whisper_context* m_ctx = nullptr;
    struct ss_model* m_ss_ctx = nullptr;
    std::string m_model_path;
};

// Legacy hook for Init
Napi::Object InitNEC(Napi::Env env, Napi::Object exports);
