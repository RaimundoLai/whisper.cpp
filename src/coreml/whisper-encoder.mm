#if !__has_feature(objc_arc)
#error This file must be compiled with automatic reference counting enabled (-fobjc-arc)
#endif

#import "whisper-encoder.h"
#import "whisper-encoder-impl.h"

#import <CoreML/CoreML.h>

#include <stdlib.h>

#if __cplusplus
extern "C" {
#endif

struct whisper_coreml_context {
    const void * data;
};

struct whisper_coreml_context * whisper_coreml_init(const char * path_model) {
    NSString * path_model_str = [[NSString alloc] initWithUTF8String:path_model];

    NSURL * url_model = [NSURL fileURLWithPath: path_model_str];

    // select which device to run the Core ML model on
    MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
    // Use CPU + Neural Engine to avoid macOS 15 MPSGraph bytecode parsing bug.
    // MLComputeUnitsAll includes the GPU/MPS path which generates broken bytecode
    // for certain model conversions on macOS 15 Sequoia, causing crashing
    // CPU + ANE still provides hardware acceleration via the Neural Engine.
    config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

    const void * data = CFBridgingRetain([[whisper_encoder_impl alloc] initWithContentsOfURL:url_model configuration:config error:nil]);

    if (data == NULL) {
        return NULL;
    }

    whisper_coreml_context * ctx = new whisper_coreml_context;

    ctx->data = data;

    return ctx;
}

void whisper_coreml_free(struct whisper_coreml_context * ctx) {
    CFRelease(ctx->data);
    delete ctx;
}

void whisper_coreml_encode(
        const whisper_coreml_context * ctx,
                             int64_t   n_ctx,
                             int64_t   n_mel,
                               float * mel,
                               float * out) {
    MLMultiArray * inMultiArray = [
        [MLMultiArray alloc] initWithDataPointer: mel
                                           shape: @[@1, @(n_mel), @(n_ctx)]
                                        dataType: MLMultiArrayDataTypeFloat32
                                         strides: @[@(n_ctx*n_mel), @(n_ctx), @1]
                                     deallocator: nil
                                           error: nil
    ];

    @autoreleasepool {
        NSError * error = nil;
        whisper_encoder_implOutput * outCoreML = [(__bridge id) ctx->data predictionFromLogmel_data:inMultiArray error:&error];

        if (!outCoreML) {
            fprintf(stderr, "%s: CoreML prediction failed: %s\n", __func__, [[error localizedDescription] UTF8String]);
            return;
        }
        if (out == NULL) {
            fprintf(stderr, "%s: Output pointer is NULL\n", __func__);
            return;
        }
        if (outCoreML.output.dataPointer == NULL) {
            fprintf(stderr, "%s: CoreML output dataPointer is NULL\n", __func__);
            return;
        }

        if (outCoreML.output.dataType == MLMultiArrayDataTypeFloat32) {
            memcpy(out, outCoreML.output.dataPointer, outCoreML.output.count * sizeof(float));
        } else if (outCoreML.output.dataType == MLMultiArrayDataTypeFloat16) {
            uint16_t *f16_data = (uint16_t *)outCoreML.output.dataPointer;
            for (NSInteger i = 0; i < outCoreML.output.count; i++) {
                uint16_t h = f16_data[i];
                uint32_t sign = (h >> 15) & 0x00000001;
                uint32_t exp  = (h >> 10) & 0x0000001f;
                uint32_t mant =  h        & 0x000003ff;

                if (exp == 0) {
                    if (mant == 0) {
                        exp = 0;
                    } else {
                        exp = 127 - 15 + 1;
                        while ((mant & 0x00000400) == 0) {
                            mant <<= 1;
                            exp--;
                        }
                        mant &= 0x000003ff;
                    }
                } else if (exp == 0x1f) {
                    exp = 255;
                } else {
                    exp = exp + (127 - 15);
                }

                uint32_t v = (sign << 31) | (exp << 23) | (mant << 13);
                memcpy(&out[i], &v, sizeof(float));
            }
        } else {
            fprintf(stderr, "%s: Unsupported CoreML output dataType\n", __func__);
        }
    }
}

#if __cplusplus
}
#endif
