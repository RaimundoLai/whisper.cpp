// model-cache.cpp — Global model instance cache manager implementation
#include "model-cache.h"

#include "whisper.h"
#include "qwen3_asr.h"
#include "forced_aligner.h"
#include "../../third-party/CrispASR/src/parakeet.h"
#include "../../third-party/CrispASR/src/qwen3_tts.h"
#include "sense-voice.h"

struct crispasr_session;
extern "C" {
    void crispasr_session_close(crispasr_session* s);
}

ModelCache& ModelCache::instance() {
    static ModelCache inst;
    return inst;
}

ModelCache::~ModelCache() {
    releaseAll();
}

std::mutex& ModelCache::mutex(ModelType t) {
    return mutexes_[static_cast<int>(t)];
}

void* ModelCache::acquire(ModelType t, const std::string& path, bool gpu, const std::string& secondary) {
    int idx = static_cast<int>(t);
    std::lock_guard<std::mutex> lock(global_mutex_);

    if (slots_[idx].matches(path, gpu, secondary)) {
        slots_[idx].in_use = true;
        slots_[idx].last_used = std::chrono::steady_clock::now();
        return slots_[idx].ctx;
    }

    if (slots_[idx].ctx) {
        freeSlotNoLock(t);
    }
    return nullptr;
}

void ModelCache::store(ModelType t, void* ctx, const std::string& path, bool gpu,
                      const std::string& secondary, int64_t auto_release_ms) {
    int idx = static_cast<int>(t);
    std::lock_guard<std::mutex> lock(global_mutex_);

    if (slots_[idx].ctx && slots_[idx].ctx != ctx) {
        freeSlotNoLock(t);
    }

    slots_[idx].ctx = ctx;
    slots_[idx].model_path = path;
    slots_[idx].secondary_path = secondary;
    slots_[idx].use_gpu = gpu;
    slots_[idx].in_use = true;
    slots_[idx].auto_release_ms = auto_release_ms;
    slots_[idx].last_used = std::chrono::steady_clock::now();
}

void ModelCache::markIdle(ModelType t) {
    int idx = static_cast<int>(t);
    std::lock_guard<std::mutex> lock(global_mutex_);

    if (slots_[idx].ctx) {
        slots_[idx].in_use = false;
        slots_[idx].last_used = std::chrono::steady_clock::now();

        if (slots_[idx].pending_release) {
            fprintf(stderr, "[ModelCache] Executing deferred release for %s\n", modelTypeToString(t));
            freeSlotNoLock(t, false);
        }
    }
}

void ModelCache::freeSlotNoLock(ModelType t, bool async) {
    int idx = static_cast<int>(t);
    CacheSlot& slot = slots_[idx];
    if (!slot.ctx) {
        return;
    }

    if (slot.in_use) {
        fprintf(stderr, "[ModelCache] Deferring release for %s (currently in use)\n", modelTypeToString(t));
        slot.pending_release = true;
        return;
    }

    void* ctx = slot.ctx;
    slot.ctx = nullptr; // Nullify immediately to prevent double-free & allow immediate slot clearing!
    slot.in_use = false;
    slot.pending_release = false;
    slot.model_path.clear();
    slot.secondary_path.clear();
    slot.auto_release_ms = 0;

    auto do_free = [ctx, t, this, idx]() {
        std::lock_guard<std::mutex> type_lock(mutexes_[idx]);
        try {
            switch (t) {
                case ModelType::WHISPER:
                    whisper_free(static_cast<struct whisper_context*>(ctx));
                    break;
                case ModelType::SENSE_VOICE:
                    sense_voice_free(static_cast<struct sense_voice_context*>(ctx));
                    break;
                case ModelType::QWEN3_ASR:
                    delete static_cast<qwen3_asr::Qwen3ASR*>(ctx);
                    break;
                case ModelType::QWEN3_ALIGNER:
                    delete static_cast<qwen3_asr::ForcedAligner*>(ctx);
                    break;
                case ModelType::PARAKEET:
                    parakeet_free(static_cast<struct parakeet_context*>(ctx));
                    break;
                case ModelType::CRISPASR_SESSION:
                    crispasr_session_close(static_cast<crispasr_session*>(ctx));
                    break;
                case ModelType::QWEN3_TTS:
                    crispasr_session_close(static_cast<crispasr_session*>(ctx));
                    break;
                default:
                    break;
            }
            fprintf(stderr, "[ModelCache] Successfully freed %s context\n", modelTypeToString(t));
        } catch (...) {
            fprintf(stderr, "[ModelCache] Warning: Exception caught while freeing %s context\n", modelTypeToString(t));
        }
    };

    if (async) {
        std::thread(do_free).detach();
    } else {
        do_free();
    }
}

void ModelCache::release(ModelType t) {
    std::lock_guard<std::mutex> lock(global_mutex_);
    freeSlotNoLock(t, false);
}

void ModelCache::releaseAll() {
    std::lock_guard<std::mutex> lock(global_mutex_);
    for (int i = 0; i < static_cast<int>(ModelType::MODEL_TYPE_COUNT); i++) {
        freeSlotNoLock(static_cast<ModelType>(i), false);
    }
}

void ModelCache::checkAutoRelease() {
    std::lock_guard<std::mutex> lock(global_mutex_);
    auto now = std::chrono::steady_clock::now();

    for (int i = 0; i < static_cast<int>(ModelType::MODEL_TYPE_COUNT); i++) {
        CacheSlot& slot = slots_[i];
        if (slot.ctx && !slot.in_use && slot.auto_release_ms > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - slot.last_used).count();
            if (elapsed >= slot.auto_release_ms) {
                fprintf(stderr, "[ModelCache] Auto-releasing idle model %s after %lld ms\n",
                        modelTypeToString(static_cast<ModelType>(i)), (long long)elapsed);
                freeSlotNoLock(static_cast<ModelType>(i));
            }
        }
    }
}

std::string ModelCache::getInfo() const {
    std::lock_guard<std::mutex> lock(global_mutex_);
    std::stringstream ss;
    ss << "{";
    bool first = true;
    for (int i = 0; i < static_cast<int>(ModelType::MODEL_TYPE_COUNT); i++) {
        const CacheSlot& slot = slots_[i];
        if (slot.ctx) {
            if (!first) ss << ", ";
            first = false;
            ss << "\"" << modelTypeToString(static_cast<ModelType>(i)) << "\": {"
               << "\"path\": \"" << slot.model_path << "\", "
               << "\"use_gpu\": " << (slot.use_gpu ? "true" : "false") << ", "
               << "\"in_use\": " << (slot.in_use ? "true" : "false") << "}";
        }
    }
    ss << "}";
    return ss.str();
}
