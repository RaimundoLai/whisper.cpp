#include "nec-addon.h"
#include <string>
#include <vector>
#include <thread>
#include <iostream>
#include <fstream>
#include <cmath>
#include "ggml.h"

// -------------------------------------------------------------------------
// Standalone SS Module GGML implementation
// -------------------------------------------------------------------------


struct ss_transformer_layer {
    struct ggml_tensor * norm1_weight = nullptr;
    struct ggml_tensor * norm1_bias   = nullptr;
    struct ggml_tensor * qkv_weight   = nullptr; // [3 * 512, 512]
    struct ggml_tensor * qkv_bias     = nullptr; // [3 * 512]
    struct ggml_tensor * proj_weight  = nullptr; // [512, 512]
    struct ggml_tensor * proj_bias    = nullptr; // [512]
    struct ggml_tensor * norm2_weight = nullptr;
    struct ggml_tensor * norm2_bias   = nullptr;
    struct ggml_tensor * ffn1_weight  = nullptr; // [ffn_dim=2048, 512]
    struct ggml_tensor * ffn1_bias    = nullptr; // [ffn_dim=2048]
    struct ggml_tensor * ffn2_weight  = nullptr; // [512, ffn_dim=2048]
    struct ggml_tensor * ffn2_bias    = nullptr; // [512]
};

struct ss_model {
    int hidden_dim = 512;
    int ffn_dim    = 2048;
    int n_heads    = 8;
    int n_layers   = 2;

    // CNN downsample layer
    struct ggml_tensor * cnn_weight = nullptr; // [512, 512, 3]
    struct ggml_tensor * cnn_bias   = nullptr; // [512]
    struct ggml_tensor * cnn_norm_w = nullptr; // [512]
    struct ggml_tensor * cnn_norm_b = nullptr; // [512]

    // Transformer Encoder Layers
    struct ss_transformer_layer layers[2];

    // GGML context
    struct ggml_context * ctx = nullptr;
    
    // Memory buffer for tensor weights
    std::vector<uint8_t> buffer;
};

// Error reporting wrapper
#define SS_CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "%s: %s\n", __func__, msg); \
            return nullptr; \
        } \
    } while (0)

struct ss_model * whisper_ss_init_from_file(const char * path_model) {
    std::string path_str = path_model;
    std::ifstream is(path_str, std::ios::binary);
    SS_CHECK(is.is_open(), "failed to open ss-model.bin");

    // File format:
    // Magic: uint32 (0x53534D4C) 'SSML'
    // hidden_dim, ffn_dim, n_heads: 3 * int32
    
    uint32_t magic;
    is.read((char *)&magic, sizeof(magic));
    SS_CHECK(magic == 0x53534D4C, "invalid magic number for ss model");

    struct ss_model * model = new ss_model;
    is.read((char *)&model->hidden_dim, sizeof(model->hidden_dim));
    int32_t dummy_ffn_dim;
    is.read((char *)&dummy_ffn_dim,     sizeof(dummy_ffn_dim)); // Written as 512 by python, but actual is 2048
    is.read((char *)&model->n_heads,    sizeof(model->n_heads));
    
    // Build tensor definitions
    struct ggml_init_params params = {
        /* .mem_size   = */ 64 * 1024 * 1024, // 64 MB for graph and definitions
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };
    model->ctx = ggml_init(params);
    SS_CHECK(model->ctx, "ggml_init failed");
    
    // The python export gives: ndims (int), shape (int[]), name_len (int), name (str), data
    int n_tensors = 0;
    while (true) {
        int32_t ndims;
        if (!is.read((char *)&ndims, sizeof(ndims))) break;

        int32_t ne[4] = {1, 1, 1, 1};
        for (int i = 0; i < ndims; ++i) {
            is.read((char *)&ne[i], sizeof(ne[i]));
        }

        int32_t name_len;
        is.read((char *)&name_len, sizeof(name_len));

        std::string name;
        name.resize(name_len);
        is.read(&name[0], name_len);

        // Python export saves shape as [dim0, dim1...] (row-major). 
        // GGML expects [width, height, channels] (column-major logic for 2D/3D matrix ops).
        // Standard linear weight PyTorch [out_feautres, in_features] -> GGML ne[0]=in, ne[1]=out.
        // For 1D it's just ne[0]
        size_t n_elements = 1;
        for (int i = 0; i < ndims; ++i) n_elements *= ne[i];
        
        size_t bpe = sizeof(float);
        
        // Save current pos to load directly into buffer later
        size_t pos = is.tellg();
        is.seekg(n_elements * bpe, std::ios::cur);
        n_tensors++;
    }

    // Now allocate real buffer
    is.clear();
    is.seekg(0, std::ios::end);
    size_t file_size = is.tellg();
    is.seekg(0, std::ios::beg);
    
    // fast forward header: magic + 3 ints
    is.seekg(sizeof(magic) + 3 * sizeof(int32_t), std::ios::beg);
    
    model->buffer.resize(file_size);
    
    for (int i = 0; i < n_tensors; ++i) {
        int32_t ndims;
        is.read((char *)&ndims, sizeof(ndims));

        int64_t ne[4] = {1, 1, 1, 1};
        for (int j = 0; j < ndims; ++j) {
            int32_t d;
            is.read((char *)&d, sizeof(d));
            ne[ndims - 1 - j] = d; // Reserve shape for GGML
        }
        
        int32_t name_len;
        is.read((char *)&name_len, sizeof(name_len));
        std::string name;
        name.resize(name_len);
        is.read(&name[0], name_len);
        
        struct ggml_tensor * tensor = nullptr;
        if (ndims == 1) tensor = ggml_new_tensor_1d(model->ctx, GGML_TYPE_F32, ne[0]);
        else if (ndims == 2) tensor = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32, ne[0], ne[1]);
        else if (ndims == 3) tensor = ggml_new_tensor_3d(model->ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2]);
        else tensor = ggml_new_tensor_4d(model->ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);

        ggml_set_name(tensor, name.c_str());
        
        // Point data to our pre-allocated vector
        size_t byte_size = ggml_nbytes(tensor);
        tensor->data = model->buffer.data() + model->buffer.size() - byte_size; // Put at end (temp strategy, real alloc down below)
        
        // Real logic: we just malloc memory for the tensor data
        tensor->data = malloc(byte_size);
        is.read((char *)tensor->data, byte_size);

        // Map to model struct pointers based on name
        if (name == "cnn.conv.weight") model->cnn_weight = tensor;
        else if (name == "cnn.conv.bias") model->cnn_bias = tensor;
        else if (name == "cnn.norm.weight") model->cnn_norm_w = tensor;
        else if (name == "cnn.norm.bias") model->cnn_norm_b = tensor;
        else {
            bool found = false;
            for (int l = 0; l < 2; ++l) {
                std::string prefix = "transformer.layers." + std::to_string(l) + ".";
                if (name == prefix + "self_attn.in_proj_weight") { model->layers[l].qkv_weight = tensor; found = true; break; }
                if (name == prefix + "self_attn.in_proj_bias") { model->layers[l].qkv_bias = tensor; found = true; break; }
                if (name == prefix + "self_attn.out_proj.weight") { model->layers[l].proj_weight = tensor; found = true; break; }
                if (name == prefix + "self_attn.out_proj.bias") { model->layers[l].proj_bias = tensor; found = true; break; }
                if (name == prefix + "linear1.weight") { model->layers[l].ffn1_weight = tensor; found = true; break; }
                if (name == prefix + "linear1.bias") { model->layers[l].ffn1_bias = tensor; found = true; break; }
                if (name == prefix + "linear2.weight") { model->layers[l].ffn2_weight = tensor; found = true; break; }
                if (name == prefix + "linear2.bias") { model->layers[l].ffn2_bias = tensor; found = true; break; }
                if (name == prefix + "norm1.weight") { model->layers[l].norm1_weight = tensor; found = true; break; }
                if (name == prefix + "norm1.bias") { model->layers[l].norm1_bias = tensor; found = true; break; }
                if (name == prefix + "norm2.weight") { model->layers[l].norm2_weight = tensor; found = true; break; }
                if (name == prefix + "norm2.bias") { model->layers[l].norm2_bias = tensor; found = true; break; }
            }
            if (!found) {
                fprintf(stderr, "Unknown tensor name: %s\n", name.c_str());
            }
        }
    }
    is.close();
    return model;
}

void whisper_ss_free(struct ss_model * model) {
    if (!model) return;
    
    // Free malloc'd data
    if (model->cnn_weight) free(model->cnn_weight->data);
    if (model->cnn_bias) free(model->cnn_bias->data);
    if (model->cnn_norm_w) free(model->cnn_norm_w->data);
    if (model->cnn_norm_b) free(model->cnn_norm_b->data);
    
    for (int l = 0; l < 2; l++) {
        if (model->layers[l].qkv_weight) free(model->layers[l].qkv_weight->data);
        if (model->layers[l].qkv_bias) free(model->layers[l].qkv_bias->data);
        if (model->layers[l].proj_weight) free(model->layers[l].proj_weight->data);
        if (model->layers[l].proj_bias) free(model->layers[l].proj_bias->data);
        if (model->layers[l].ffn1_weight) free(model->layers[l].ffn1_weight->data);
        if (model->layers[l].ffn1_bias) free(model->layers[l].ffn1_bias->data);
        if (model->layers[l].ffn2_weight) free(model->layers[l].ffn2_weight->data);
        if (model->layers[l].ffn2_bias) free(model->layers[l].ffn2_bias->data);
        if (model->layers[l].norm1_weight) free(model->layers[l].norm1_weight->data);
        if (model->layers[l].norm1_bias) free(model->layers[l].norm1_bias->data);
        if (model->layers[l].norm2_weight) free(model->layers[l].norm2_weight->data);
        if (model->layers[l].norm2_bias) free(model->layers[l].norm2_bias->data);
    }

    if (model->ctx) ggml_free(model->ctx);
    delete model;
}

// -------------------------------------------------------------------------
// Represents a single segmented result
// -------------------------------------------------------------------------
struct SSEmbeddingResult {
    float start;
    float end;
    std::vector<float> embedding;
};

// Full inference function inside C++
static int whisper_ss_get_embedding(struct ss_model * model, struct whisper_context * wctx, const float * samples, int n_samples, std::vector<float>& out_embedding) {
    if (!model || !wctx) return -1;
    out_embedding.resize(512);

    // 1. Run standard whisper base encoder
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_progress   = false;
    wparams.print_realtime   = false;
    wparams.print_timestamps = false;
    wparams.no_context       = true;
    
    whisper_state * wstate = whisper_init_state(wctx);
    
    if (whisper_pcm_to_mel_with_state(wctx, wstate, samples, n_samples, /*n_threads=*/ 4) != 0) {
        fprintf(stderr, "Failed to compute mel spectrogram\n");
        whisper_free_state(wstate);
        return -1;
    }
    
    if (whisper_encode_with_state(wctx, wstate, /*offset=*/ 0, /*n_threads=*/ 4) != 0) {
        fprintf(stderr, "Failed to encode audio\n");
        whisper_free_state(wstate);
        return -1;
    }
    
    // Get encoder features: [n_audio_ctx, 512] -> [seq_len, 512]
    int n_mels_frames = whisper_n_len_from_state(wstate); // total mel frames
    int seq_len = n_mels_frames / 2; // CNN stride 2 in whisper base

    // Copy encoder output to CPU buffer (GPU-safe)
    std::vector<float> enc_buf(seq_len * 512);
    int copied = whisper_copy_embd_enc_from_state(wstate, enc_buf.data(), seq_len * 512);
    if (copied <= 0) {
        whisper_free_state(wstate);
        return -1;
    }

    // 2. Build GGML Compute Graph for SS Module
    int n_threads = 4;
    size_t buf_size = 128 * 1024 * 1024; // 128MB for activations
    void * buf = malloc(buf_size);
    struct ggml_init_params ggml_params = {
        /* .mem_size   = */ buf_size,
        /* .mem_buffer = */ buf,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx0 = ggml_init(ggml_params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    // Inputs: whisper encoder features [seq_len, 512]. (C-array layout: [seq_len, 512], meaning ne[0]=512, ne[1]=seq_len in GGML)
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 512, seq_len);
    memcpy(x->data, enc_buf.data(), seq_len * 512 * sizeof(float));

    auto safe_reshape_2d = [&](struct ggml_tensor * a, int64_t ne0, int64_t ne1, const char * name) {
        if (ggml_nelements(a) != ne0 * ne1) {
            fprintf(stderr, "ggml_reshape_2d assertion failed at %s! a->nelements=%d, expected %d*%d=%d\n", name, (int)ggml_nelements(a), (int)ne0, (int)ne1, (int)(ne0*ne1));
        }
        return ggml_reshape_2d(ctx0, a, ne0, ne1);
    };

    // Transpose x for Conv1d expectation: GGML conv_1d expects ne[0]=Length, ne[1]=IC.
    struct ggml_tensor * x_t = ggml_cont(ctx0, ggml_transpose(ctx0, x));

    // --- CNN Downsample ---
    // nn.Conv1d(512, 512, kernel=3, stride=2, padding=1)
    // IMPORTANT: ggml_conv_1d (via ggml_im2col) strictly requires the kernel to be F16 on CPU backends.
    struct ggml_tensor * cnn_w = ggml_cast(ctx0, model->cnn_weight, GGML_TYPE_F16); // [3, 512, 512] in GGML view
    struct ggml_tensor * cnn_out = ggml_conv_1d(ctx0, cnn_w, x_t, /*stride=*/ 2, /*padding=*/ 1, /*dilation=*/ 1);
    
    // cnn_out is returned as [Length, Channels]. Transpose back to [Channels, Length]
    cnn_out = ggml_cont(ctx0, ggml_transpose(ctx0, cnn_out));

    // Now cnn_out has ne[0]=512, ne[1]=OL.
    // Add bias
    struct ggml_tensor * b = ggml_repeat(ctx0, safe_reshape_2d(model->cnn_bias, 512, 1, "cnn_bias"), cnn_out);
    cnn_out = ggml_add(ctx0, cnn_out, b);
    
    // CNN LayerNorm
    // cnn_out is [512, OL]. ggml_norm works along ne[0] (which is Channels).
    cnn_out = ggml_norm(ctx0, cnn_out, 1e-5f);
    struct ggml_tensor * cnn_norm_w = ggml_repeat(ctx0, safe_reshape_2d(model->cnn_norm_w, 512, 1, "cnn_norm_w"), cnn_out);
    struct ggml_tensor * cnn_norm_b = ggml_repeat(ctx0, safe_reshape_2d(model->cnn_norm_b, 512, 1, "cnn_norm_b"), cnn_out);
    cnn_out = ggml_add(ctx0, ggml_mul(ctx0, cnn_out, cnn_norm_w), cnn_norm_b);
    
    int cur_seq_len = cnn_out->ne[1]; 
    struct ggml_tensor * current_h = cnn_out;

    // --- Transformer Encoder Layers ---
    for (int l = 0; l < 2; l++) {
        struct ss_transformer_layer & layer = model->layers[l];

        // 1. Self Attention (Input shape [512, cur_seq_len])
        struct ggml_tensor * qkv = ggml_mul_mat(ctx0, layer.qkv_weight, current_h); // [1536, cur_seq_len]
        struct ggml_tensor * qkv_b = ggml_repeat(ctx0, safe_reshape_2d(layer.qkv_bias, 512*3, 1, "qkv_bias"), qkv);
        qkv = ggml_add(ctx0, qkv, qkv_b);
        
        // Split Q, K, V
        int head_dim = model->hidden_dim / model->n_heads; // 512/8 = 64
        size_t type_size = ggml_type_size(GGML_TYPE_F32);

        struct ggml_tensor * Q = ggml_view_3d(ctx0, qkv, 
            head_dim, model->n_heads, cur_seq_len, 
            head_dim * type_size, qkv->nb[1], 
            0);

        struct ggml_tensor * K = ggml_view_3d(ctx0, qkv, 
            head_dim, model->n_heads, cur_seq_len, 
            head_dim * type_size, qkv->nb[1], 
            512 * type_size);

        struct ggml_tensor * V = ggml_view_3d(ctx0, qkv, 
            head_dim, model->n_heads, cur_seq_len, 
            head_dim * type_size, qkv->nb[1], 
            1024 * type_size);

        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3)); 
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        // V needs to be transposed to [seq_len, head_dim, n_heads] for ggml_mul_mat with [seq_len, seq_len, n_heads]
        // Old: [head_dim(0), n_heads(1), seq_len(2)]
        // We want new shape: ne[0]=seq_len(old 2), ne[1]=head_dim(old 0), ne[2]=n_heads(old 1)
        // ggml_permute axes specify which NEW axis the old axis goes to.
        // Old 0 -> New 1  => axis0 = 1
        // Old 1 -> New 2  => axis1 = 2
        // Old 2 -> New 0  => axis2 = 0
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3));

        struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
        KQ = ggml_scale(ctx0, KQ, 1.0f / sqrtf((float)head_dim));
        struct ggml_tensor * KQ_soft = ggml_soft_max(ctx0, KQ);
        struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft);

        KQV = ggml_cont(ctx0, ggml_permute(ctx0, KQV, 0, 2, 1, 3)); 
        KQV = safe_reshape_2d(KQV, 512, cur_seq_len, "kqv_reshape");
        
        struct ggml_tensor * attn_out = ggml_mul_mat(ctx0, layer.proj_weight, KQV);
        struct ggml_tensor * proj_b = ggml_repeat(ctx0, safe_reshape_2d(layer.proj_bias, 512, 1, "proj_bias"), attn_out);
        attn_out = ggml_add(ctx0, attn_out, proj_b);
        
        // Residual + Norm 1 (PyTorch LayerNorm)
        struct ggml_tensor * h1 = ggml_add(ctx0, current_h, attn_out);
        h1 = ggml_norm(ctx0, h1, 1e-5f);
        struct ggml_tensor * norm1_w = ggml_repeat(ctx0, safe_reshape_2d(layer.norm1_weight, 512, 1, "norm1_weight"), h1);
        struct ggml_tensor * norm1_b = ggml_repeat(ctx0, safe_reshape_2d(layer.norm1_bias, 512, 1, "norm1_bias"), h1);
        h1 = ggml_add(ctx0, ggml_mul(ctx0, h1, norm1_w), norm1_b);

        // FFN
        struct ggml_tensor * ffn1 = ggml_mul_mat(ctx0, layer.ffn1_weight, h1);
        struct ggml_tensor * ffb1 = ggml_repeat(ctx0, safe_reshape_2d(layer.ffn1_bias, model->ffn_dim, 1, "ffn1_bias"), ffn1);
        ffn1 = ggml_add(ctx0, ffn1, ffb1);
        ffn1 = ggml_relu(ctx0, ffn1); // PyTorch nn.TransformerEncoderLayer default activation is "relu"

        struct ggml_tensor * ffn2 = ggml_mul_mat(ctx0, layer.ffn2_weight, ffn1);
        struct ggml_tensor * ffb2 = ggml_repeat(ctx0, safe_reshape_2d(layer.ffn2_bias, 512, 1, "ffn2_bias"), ffn2);
        ffn2 = ggml_add(ctx0, ffn2, ffb2);
        
        // Residual + Norm 2
        struct ggml_tensor * h2 = ggml_add(ctx0, h1, ffn2);
        h2 = ggml_norm(ctx0, h2, 1e-5f);
        struct ggml_tensor * norm2_w = ggml_repeat(ctx0, safe_reshape_2d(layer.norm2_weight, 512, 1, "norm2_weight"), h2);
        struct ggml_tensor * norm2_b = ggml_repeat(ctx0, safe_reshape_2d(layer.norm2_bias, 512, 1, "norm2_bias"), h2);
        h2 = ggml_add(ctx0, ggml_mul(ctx0, h2, norm2_w), norm2_b);
        
        current_h = h2;
    }

    // --- Global Mean Pooling ---
    // Transpose current_h to [cur_seq_len, 512] -> ne[0] = cur_seq_len, ne[1] = 512
    struct ggml_tensor * current_h_t = ggml_cont(ctx0, ggml_transpose(ctx0, current_h));
    
    // Create avg_w [cur_seq_len] and fill with 1.0f / cur_seq_len
    struct ggml_tensor * avg_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, cur_seq_len);
    float * avg_data = (float *) avg_w->data;
    for (int i = 0; i < cur_seq_len; i++) {
        avg_data[i] = 1.0f / cur_seq_len;
    }
    
    // ggml_mul_mat(ctx, A, B) -> C = B * A^T
    // B = current_h_t [cur_seq_len, 512] => ne[0]=cur_seq_len, ne[1]=512
    // A = avg_w [cur_seq_len, 1] => ne[0]=cur_seq_len, ne[1]=1
    // Result C => ne[0] = 1, ne[1] = 512.
    struct ggml_tensor * pool = ggml_mul_mat(ctx0, avg_w, current_h_t);

    // reshape to [512] for norm
    pool = ggml_reshape_1d(ctx0, pool, 512);

    // --- L2 Normalization ---
    // Using ggml_rms_norm (which is x / sqrt(mean(x^2))). L2 norm is x / sqrt(sum(x^2)).
    // RMSNorm = x / sqrt(sum(x^2)/N). So L2Norm = RMSNorm / sqrt(N). N = 512.
    struct ggml_tensor * l2 = ggml_rms_norm(ctx0, pool, 1e-12f);
    l2 = ggml_scale(ctx0, l2, 1.0f / sqrtf(model->hidden_dim));

    // Build compute graph
    ggml_build_forward_expand(gf, l2);
    
    // Compute
    ggml_graph_compute_with_ctx(ctx0, gf, n_threads);

    // Copy result
    float * result_data = (float *)l2->data;
    memcpy(out_embedding.data(), result_data, 512 * sizeof(float));

    // Cleanup
    ggml_free(ctx0);
    free(buf);
    whisper_free_state(wstate);

    return 0;
}

// -------------------------------------------------------------------------
// Asynchronous Worker for SS Module Embedding Extraction
// -------------------------------------------------------------------------
class SSEmbeddingWorker : public Napi::AsyncWorker {
public:
    SSEmbeddingWorker(Napi::Function& callback, 
                      const std::string& ss_model_path, 
                      const std::string& base_model_path, 
                      whisper_context* shared_wctx,
                      struct ss_model* shared_ss_ctx,
                      const std::string& vad_model_path,
                      bool use_gpu,
                      whisper_vad_params vad_params,
                      const std::vector<float>& pcmf32)
        : Napi::AsyncWorker(callback), 
          m_ss_path(ss_model_path), 
          m_base_path(base_model_path), 
          m_shared_wctx(shared_wctx),
          m_shared_ss_ctx(shared_ss_ctx),
          m_owns_wctx(shared_wctx == nullptr),
          m_owns_ss_ctx(shared_ss_ctx == nullptr),
          m_vad_path(vad_model_path),
          m_use_gpu(use_gpu),
          m_vad_params(vad_params),
          m_pcmf32(pcmf32) {}

    void Execute() override {
        // 1. Initialize SS model (or use shared)
        struct ss_model * ss_ctx = m_shared_ss_ctx;
        if (!ss_ctx) {
            ss_ctx = whisper_ss_init_from_file(m_ss_path.c_str());
            if (!ss_ctx) {
                SetError("Failed to initialize SS Model from " + m_ss_path);
                return;
            }
        }

        // 2. Initialize Whisper Base Context (or use shared)
        struct whisper_context * wctx = m_shared_wctx;
        if (!wctx) {
            struct whisper_context_params cparams = whisper_context_default_params();
            cparams.use_gpu = m_use_gpu; // Configurable GPU
            wctx = whisper_init_from_file_with_params(m_base_path.c_str(), cparams);

            if (!wctx) {
                whisper_ss_free(ss_ctx);
                SetError("Failed to initialize whisper base model from " + m_base_path);
                return;
            }
        }

        // 3. Process Segmentation (VAD or Whole)
        if (!m_vad_path.empty()) {
            // VAD is enabled
            struct whisper_vad_context_params vad_cparams = whisper_vad_default_context_params();
            vad_cparams.use_gpu = false; // Use CPU for VAD to avoid GPU/CUDA/Metal issues
            whisper_vad_context* vad_ctx = whisper_vad_init_from_file_with_params(m_vad_path.c_str(), vad_cparams);
            
            if (!vad_ctx) {
                if (m_owns_wctx) whisper_free(wctx);
                whisper_ss_free(ss_ctx);
                SetError("Failed to initialize VAD context from " + m_vad_path);
                return;
            }

            struct whisper_vad_segments* segments = whisper_vad_segments_from_samples(
                vad_ctx, m_vad_params, m_pcmf32.data(), m_pcmf32.size());
            
            if (segments) {
                int n_segments = whisper_vad_segments_n_segments(segments);
                for (int i = 0; i < n_segments; i++) {
                    float t0 = whisper_vad_segments_get_segment_t0(segments, i) / 100.0f;
                    float t1 = whisper_vad_segments_get_segment_t1(segments, i) / 100.0f;
                    
                    int start_sample = std::max(0, (int)(t0 * WHISPER_SAMPLE_RATE));
                    int end_sample = std::min((int)m_pcmf32.size(), (int)(t1 * WHISPER_SAMPLE_RATE));
                    
                    // Min length guard
                    if (end_sample - start_sample < WHISPER_SAMPLE_RATE * 0.1) continue;

                    SSEmbeddingResult res;
                    res.start = t0;
                    res.end = t1;
                    int ret = whisper_ss_get_embedding(ss_ctx, wctx, m_pcmf32.data() + start_sample, end_sample - start_sample, res.embedding);
                    if (ret == 0) m_results.push_back(res);
                }
                whisper_vad_free_segments(segments);
            }
            whisper_vad_free(vad_ctx);
        } else {
            // Whole audio processing
            SSEmbeddingResult res;
            res.start = 0.0f;
            res.end = (float)m_pcmf32.size() / WHISPER_SAMPLE_RATE;
            int ret = whisper_ss_get_embedding(ss_ctx, wctx, m_pcmf32.data(), (int)m_pcmf32.size(), res.embedding);
            if (ret == 0) m_results.push_back(res);
        }

        // 4. Cleanup
        if (m_owns_wctx && wctx) {
            whisper_free(wctx);
        }
        if (m_owns_ss_ctx && ss_ctx) {
            whisper_ss_free(ss_ctx);
        }
    }

    void OnOK() override {
        Napi::Env env = Env();
        Napi::HandleScope scope(env);

        Napi::Array resultsArray = Napi::Array::New(env, m_results.size());

        for (size_t i = 0; i < m_results.size(); i++) {
            Napi::Object obj = Napi::Object::New(env);
            
            Napi::ArrayBuffer buffer = Napi::ArrayBuffer::New(env, m_results[i].embedding.size() * sizeof(float));
            memcpy(buffer.Data(), m_results[i].embedding.data(), m_results[i].embedding.size() * sizeof(float));
            Napi::Float32Array arr = Napi::Float32Array::New(env, m_results[i].embedding.size(), buffer, 0);

            obj.Set("start", Napi::Number::New(env, m_results[i].start));
            obj.Set("end", Napi::Number::New(env, m_results[i].end));
            obj.Set("embedding", arr);
            
            resultsArray[i] = obj;
        }

        Napi::Object result = Napi::Object::New(env);
        result.Set("segments", resultsArray);

        Callback().Call({env.Null(), result});
    }

private:
    std::string m_ss_path;
    std::string m_base_path;
    whisper_context* m_shared_wctx;
    struct ss_model* m_shared_ss_ctx;
    bool m_owns_wctx;
    bool m_owns_ss_ctx;
    std::string m_vad_path;
    bool m_use_gpu;
    whisper_vad_params m_vad_params;
    
    std::vector<float> m_pcmf32;
    std::vector<SSEmbeddingResult> m_results;
};

// -------------------------------------------------------------------------
// Napi Wrapper exposed to JavaScript
// Usage: extractSSEmbedding({ 
//    model: "ss-model.bin", 
//    base_model: "whisper.bin", 
//    pcmf32: Float32Array,
//    vad_model: "ggml-silero.bin", // Optional
//    use_gpu: true,
//    threshold: 0.9, ...
// }, callback)
// -------------------------------------------------------------------------
Napi::Value QueueSSEmbeddingWorkerFromOptions(Napi::Env env, Napi::Object options, Napi::Function callback, whisper_context* shared_wctx, struct ss_model* shared_ss_ctx) {
    // Parse options
    std::string model_path = "";
    if (shared_ss_ctx == nullptr) {
        if (!options.Has("model") || !options.Get("model").IsString()) {
            Napi::TypeError::New(env, "Option 'model' must be a string if SS model is not pre-loaded.").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        model_path = options.Get("model").As<Napi::String>();
    } else {
        if (options.Has("model") && options.Get("model").IsString()) {
            model_path = options.Get("model").As<Napi::String>();
        }
    }

    std::string base_model_path = "";
    if (shared_wctx == nullptr) {
        if (options.Has("base_model")) {
            Napi::Value base_model_val = options.Get("base_model");
            if (base_model_val.IsString()) {
                base_model_path = base_model_val.As<Napi::String>();
            } else if (base_model_val.IsObject()) {
                // Attempt to unwrap it as NEC object
                Napi::Object obj = base_model_val.As<Napi::Object>();
                NEC* nec_instance = Napi::ObjectWrap<NEC>::Unwrap(obj);
                if (nec_instance) {
                    shared_wctx = nec_instance->GetContext();
                    shared_ss_ctx = nec_instance->GetSSContext();
                } else {
                    Napi::TypeError::New(env, "Option 'base_model' object is not a valid NEC instance.").ThrowAsJavaScriptException();
                    return env.Undefined();
                }
            } else {
                Napi::TypeError::New(env, "Option 'base_model' must be a string or a valid NEC instance.").ThrowAsJavaScriptException();
                return env.Undefined();
            }
        } else {
            Napi::TypeError::New(env, "Option 'base_model' is required.").ThrowAsJavaScriptException();
            return env.Undefined();
        }
    }

    if (!options.Has("pcmf32") || !options.Get("pcmf32").IsTypedArray()) {
        Napi::TypeError::New(env, "Option 'pcmf32' must be a Float32Array.").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Float32Array pcmf32_arr = options.Get("pcmf32").As<Napi::Float32Array>();
    std::vector<float> pcmf32;
    pcmf32.assign(pcmf32_arr.Data(), pcmf32_arr.Data() + pcmf32_arr.ElementLength());

    // GPU and VAD
    bool use_gpu = true;
    if (options.Has("use_gpu") && options.Get("use_gpu").IsBoolean()) {
        use_gpu = options.Get("use_gpu").As<Napi::Boolean>();
    }

    std::string vad_model_path = "";
    if (options.Has("vad_model") && options.Get("vad_model").IsString()) {
        vad_model_path = options.Get("vad_model").As<Napi::String>();
    }

    whisper_vad_params vad_params = whisper_vad_default_params();
    if (options.Has("threshold") && options.Get("threshold").IsNumber()) {
        vad_params.threshold = options.Get("threshold").As<Napi::Number>().FloatValue();
    }
    if (options.Has("min_speech_duration_ms") && options.Get("min_speech_duration_ms").IsNumber()) {
        vad_params.min_speech_duration_ms = options.Get("min_speech_duration_ms").As<Napi::Number>();
    }
    if (options.Has("min_silence_duration_ms") && options.Get("min_silence_duration_ms").IsNumber()) {
        vad_params.min_silence_duration_ms = options.Get("min_silence_duration_ms").As<Napi::Number>();
    }
    if (options.Has("speech_pad_ms") && options.Get("speech_pad_ms").IsNumber()) {
        vad_params.speech_pad_ms = options.Get("speech_pad_ms").As<Napi::Number>();
    }

    SSEmbeddingWorker* worker = new SSEmbeddingWorker(
        callback, model_path, base_model_path, shared_wctx, shared_ss_ctx, vad_model_path, use_gpu, vad_params, pcmf32);
    worker->Queue();

    return env.Undefined();
}

Napi::Value extractSSEmbedding(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2 || !info[0].IsObject() || !info[1].IsFunction()) {
        Napi::TypeError::New(env, "Usage: extractSSEmbedding(options, callback)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Object options = info[0].As<Napi::Object>();
    Napi::Function callback = info[1].As<Napi::Function>();

    return QueueSSEmbeddingWorkerFromOptions(env, options, callback, nullptr, nullptr);
}
