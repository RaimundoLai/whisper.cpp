#pragma once

// 1. Rename the original structures to prevent naming collisions
#define ggml_backend_buffer_i ggml_backend_buffer_i_original
#define ggml_backend_buffer ggml_backend_buffer_original

// 2. Include the real, original internal backend header
#include "../../ggml/src/ggml-backend-impl.h"

// 3. Undefine our rename macros
#undef ggml_backend_buffer_i
#undef ggml_backend_buffer

// 4. Define the CrispASR-compatible ggml_backend_buffer_i struct
struct ggml_backend_buffer_i {
    void         (*free_buffer)  (ggml_backend_buffer_t buffer);
    void *       (*get_base)     (ggml_backend_buffer_t buffer);
    enum ggml_status (*init_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
    void         (*memset_tensor)(ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor,     uint8_t value, size_t offset, size_t size);
    void         (*set_tensor)   (ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    void         (*get_tensor)   (ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
    
    // Augmented 2D tensor fields for CrispASR compatibility
    void         (*set_tensor_2d)(ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
    void         (*get_tensor_2d)(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);

    bool         (*cpy_tensor)   (ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst);
    void         (*clear)        (ggml_backend_buffer_t buffer, uint8_t value);
    void         (*reset)        (ggml_backend_buffer_t buffer);
};

// 5. Define our property-based compatible wrapper for ggml_backend_buffer to handle layout/calling safety
struct ggml_backend_buffer {
    #ifdef __clang__
    __declspec(property(get = get_iface)) struct ggml_backend_buffer_i iface;
    __declspec(property(get = get_buft)) ggml_backend_buffer_type_t buft;
    __declspec(property(get = get_context)) void * context;
    __declspec(property(get = get_size)) size_t size;
    __declspec(property(get = get_usage)) enum ggml_backend_buffer_usage usage;

    struct ggml_backend_buffer_i get_iface() const {
        const struct ggml_backend_buffer_original * orig = (const struct ggml_backend_buffer_original *)this;
        struct ggml_backend_buffer_i compat;
        compat.free_buffer   = orig->iface.free_buffer;
        compat.get_base      = orig->iface.get_base;
        compat.init_tensor   = orig->iface.init_tensor;
        compat.memset_tensor = orig->iface.memset_tensor;
        compat.set_tensor    = orig->iface.set_tensor;
        compat.get_tensor    = orig->iface.get_tensor;
        compat.cpy_tensor    = orig->iface.cpy_tensor;
        compat.clear         = orig->iface.clear;
        compat.reset         = orig->iface.reset;
        compat.set_tensor_2d = nullptr; // Safe stub
        compat.get_tensor_2d = nullptr; // Safe stub
        return compat;
    }

    ggml_backend_buffer_type_t get_buft() const {
        return ((const struct ggml_backend_buffer_original *)this)->buft;
    }

    void * get_context() const {
        return ((const struct ggml_backend_buffer_original *)this)->context;
    }

    size_t get_size() const {
        return ((const struct ggml_backend_buffer_original *)this)->size;
    }

    enum ggml_backend_buffer_usage get_usage() const {
        return ((const struct ggml_backend_buffer_original *)this)->usage;
    }
    #endif
};

// 6. Inline compatible wrapper for ggml_backend_buffer_init to convert between struct types
static inline ggml_backend_buffer_t ggml_backend_buffer_init_compat(
    ggml_backend_buffer_type_t buft,
    struct ggml_backend_buffer_i iface,
    void * context,
    size_t size)
{
    struct ggml_backend_buffer_i_original orig;
    orig.free_buffer   = iface.free_buffer;
    orig.get_base      = iface.get_base;
    orig.init_tensor   = iface.init_tensor;
    orig.memset_tensor = iface.memset_tensor;
    orig.set_tensor    = iface.set_tensor;
    orig.get_tensor    = iface.get_tensor;
    orig.cpy_tensor    = iface.cpy_tensor;
    orig.clear         = iface.clear;
    orig.reset         = iface.reset;
    return ggml_backend_buffer_init(buft, orig, context, size);
}

// 7. Hijack the ggml_backend_buffer_init call to use the compatibility wrapper
#define ggml_backend_buffer_init ggml_backend_buffer_init_compat
