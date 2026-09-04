// model-cache.h — Global model instance cache manager for Native Addon
#pragma once

#include <mutex>
#include <thread>
#include <string>
#include <chrono>
#include <memory>
#include <iostream>
#include <sstream>

enum class ModelType : int {
    WHISPER = 0,
    SENSE_VOICE,
    QWEN3_ASR,
    QWEN3_ALIGNER,
    PARAKEET,
    CRISPASR_SESSION,
    QWEN3_TTS,
    CRISPASR_TTS = QWEN3_TTS,
    MODEL_TYPE_COUNT = 7
};

inline const char* modelTypeToString(ModelType t) {
    switch (t) {
        case ModelType::WHISPER: return "whisper";
        case ModelType::SENSE_VOICE: return "sense_voice";
        case ModelType::QWEN3_ASR: return "qwen3_asr";
        case ModelType::QWEN3_ALIGNER: return "qwen3_aligner";
        case ModelType::PARAKEET: return "parakeet";
        case ModelType::CRISPASR_SESSION: return "crispasr";
        case ModelType::QWEN3_TTS: return "crispasr_tts";
        default: return "unknown";
    }
}

inline bool stringToModelType(const std::string& s, ModelType& outType) {
    if (s == "whisper") { outType = ModelType::WHISPER; return true; }
    if (s == "sense_voice" || s == "sensevoice") { outType = ModelType::SENSE_VOICE; return true; }
    if (s == "qwen3_asr" || s == "qwen_asr" || s == "qwen") { outType = ModelType::QWEN3_ASR; return true; }
    if (s == "qwen3_aligner" || s == "qwen_aligner" || s == "aligner" || s == "align") { outType = ModelType::QWEN3_ALIGNER; return true; }
    if (s == "parakeet") { outType = ModelType::PARAKEET; return true; }
    if (s == "crispasr" || s == "vibevoice" || s == "voxtral") { outType = ModelType::CRISPASR_SESSION; return true; }
    if (s == "qwen3_tts" || s == "qwen_tts" || s == "tts" || s == "crispasr_tts") { outType = ModelType::QWEN3_TTS; return true; }
    return false;
}

struct CacheSlot {
    void* ctx = nullptr;
    std::string model_path;
    std::string secondary_path;  // e.g. codec_model for TTS, backend_name for CrispASR
    bool use_gpu = false;
    bool in_use = false;
    bool pending_release = false;
    int64_t auto_release_ms = 0; // 0 = never
    std::chrono::steady_clock::time_point last_used;

    bool matches(const std::string& path, bool gpu, const std::string& secondary = "") const {
        return ctx != nullptr && model_path == path && use_gpu == gpu &&
               (secondary.empty() || secondary_path == secondary);
    }
};

class ModelCache {
public:
    static ModelCache& instance();

    std::mutex& mutex(ModelType t);

    // Lock global mutex, return ctx if matches. If mismatch, frees old instance and returns nullptr.
    void* acquire(ModelType t, const std::string& path, bool gpu, const std::string& secondary = "");

    // Store newly initialized instance in cache slot
    void store(ModelType t, void* ctx, const std::string& path, bool gpu,
               const std::string& secondary = "", int64_t auto_release_ms = 0);

    // Mark as no longer in use
    void markIdle(ModelType t);

    // Safely free single slot (checking for nullptr and handling exceptions safely)
    void release(ModelType t);

    // Safely free all slots
    void releaseAll();

    // Check auto-release timeout
    void checkAutoRelease();

    // Debug info string
    std::string getInfo() const;

private:
    ModelCache() = default;
    ~ModelCache();

    void freeSlotNoLock(ModelType t, bool async = false);

    CacheSlot slots_[static_cast<int>(ModelType::MODEL_TYPE_COUNT)];
    std::mutex mutexes_[static_cast<int>(ModelType::MODEL_TYPE_COUNT)];
    mutable std::mutex global_mutex_;
};
