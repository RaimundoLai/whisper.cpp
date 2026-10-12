// Shared segmentation policy for the chunked-VAD streaming workers
// (SenseVoiceStream, QwenASRStream, CrispASRStream).
//
// Provides three latency optimisations that are independent of the ASR engine:
//   1. Adaptive silence threshold: the longer an utterance runs, the shorter the
//      pause required to close it, so long speech is cut at natural pauses
//      instead of at the hard maximum length.
//   2. Speculative final (tail decode): shortly after speech stops, decode the
//      exact audio the final pass would decode. If no speech resumes before the
//      utterance closes, the cached result is emitted as the final segment, so
//      the final arrives without an additional decode.
//   3. Early commit: when two consecutive progressive hypotheses agree on a
//      prefix that ends with sentence punctuation, emit that prefix as a final
//      segment and drop its audio from the working buffer.
//
// This header is engine-agnostic and has no Node-API dependency so it can be
// unit tested on its own (see __test__/stream-segmenter-test.cpp).

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace stream_seg {

struct Word {
    std::string text;
    int64_t t0_ms = 0; // absolute stream time
    int64_t t1_ms = 0; // absolute stream time
    float p = 1.0f;
};

struct Hypothesis {
    int64_t start_ms = 0;
    int64_t end_ms = 0;
    std::string text;
    std::vector<Word> words;
    // True only when word timestamps come from the model/aligner rather than
    // being interpolated. Early commit requires real timestamps.
    bool words_timed = false;
    // Engine-specific extras carried through unchanged.
    std::string language;
    std::string emotion;
    std::string event;
};

struct Options {
    bool adaptive_silence = false;
    int adaptive_mid_ms = 5000;
    int adaptive_soft_ms = 10000;
    int adaptive_mid_silence_ms = 350;
    int adaptive_soft_silence_ms = 200;

    bool speculative_final = false;
    int tail_delay_ms = 300;

    bool early_commit = false;
    int early_commit_min_ms = 1500;
    int early_commit_min_remaining_ms = 200;
};

// Reads the shared options from a Napi::Object-like value. Templated so the
// header does not depend on napi.h.
template <typename Obj>
inline void parse_options(Options& o, const Obj& options) {
    auto read_bool = [&](const char* key, bool& out) {
        if (options.Has(key) && options.Get(key).IsBoolean()) {
            out = options.Get(key).ToBoolean().Value();
        }
    };
    auto read_int = [&](const char* key, int& out) {
        if (options.Has(key) && options.Get(key).IsNumber()) {
            out = options.Get(key).ToNumber().Int32Value();
        }
    };
    read_bool("adaptive_silence", o.adaptive_silence);
    read_int("adaptive_mid_ms", o.adaptive_mid_ms);
    read_int("adaptive_soft_ms", o.adaptive_soft_ms);
    read_int("adaptive_mid_silence_ms", o.adaptive_mid_silence_ms);
    read_int("adaptive_soft_silence_ms", o.adaptive_soft_silence_ms);
    read_bool("speculative_final", o.speculative_final);
    read_int("tail_delay_ms", o.tail_delay_ms);
    read_bool("early_commit", o.early_commit);
    read_int("early_commit_min_ms", o.early_commit_min_ms);
    read_int("early_commit_min_remaining_ms", o.early_commit_min_remaining_ms);
}

// Returns T such that the utterance closes when silence_chunks > T.
// Without adaptive mode this is the configured base count.
inline int silence_threshold(const Options& o, int base_chunks, int chunk_ms, int speech_chunks) {
    if (!o.adaptive_silence || chunk_ms <= 0) {
        return base_chunks;
    }
    const int64_t speech_ms = static_cast<int64_t>(speech_chunks) * chunk_ms;
    int target_ms;
    if (speech_ms >= o.adaptive_soft_ms) {
        target_ms = o.adaptive_soft_silence_ms;
    } else if (speech_ms >= o.adaptive_mid_ms) {
        target_ms = o.adaptive_mid_silence_ms;
    } else {
        return base_chunks;
    }
    // Close after ceil(target / chunk) silent chunks, i.e. T = that count - 1.
    int needed = (target_ms + chunk_ms - 1) / chunk_ms;
    int t = std::max(1, needed - 1);
    return std::min(base_chunks, t);
}

// Holds the decode result of the speculative tail pass. It is reusable as the
// final result only if the final pass would decode the identical audio window.
template <typename Result>
struct TailCache {
    bool valid = false;
    size_t pcm_size = 0;
    int64_t start_ms = 0;
    int64_t end_ms = 0;
    Result result{};

    void reset() {
        valid = false;
        pcm_size = 0;
        result = Result{};
    }

    bool matches(size_t size, int64_t start, int64_t end) const {
        return valid && pcm_size == size && start_ms == start && end_ms == end;
    }
};

// Number of trailing samples the final pass trims, given the silent samples
// held at the end of the speech buffer. Mirrors the existing workers: keep at
// most pad_samples of trailing silence.
inline size_t trailing_trim(size_t silence_samples, size_t pad_samples) {
    return silence_samples > pad_samples ? silence_samples - pad_samples : 0;
}

// ---------------------------------------------------------------------------
// Early commit planning
// ---------------------------------------------------------------------------

struct CommitPlan {
    size_t text_end = 0;    // byte offset in Hypothesis::text after the punctuation
    size_t n_words = 0;     // words [0, n_words) belong to the committed head
    int64_t commit_end_ms = 0;
    int64_t cut_ms = 0;     // absolute time at which the remaining audio starts
};

// Returns the byte length of a sentence-ending punctuation mark at pos, or 0.
inline size_t sentence_punct_len(const std::string& s, size_t pos) {
    if (pos >= s.size()) return 0;
    const unsigned char c = static_cast<unsigned char>(s[pos]);
    if (c == '.' || c == '?' || c == '!') return 1;
    if (pos + 3 <= s.size()) {
        const unsigned char c1 = static_cast<unsigned char>(s[pos + 1]);
        const unsigned char c2 = static_cast<unsigned char>(s[pos + 2]);
        if (c == 0xE3 && c1 == 0x80 && c2 == 0x82) return 3; // U+3002 ideographic full stop
        if (c == 0xEF && c1 == 0xBC && (c2 == 0x9F || c2 == 0x81)) return 3; // fullwidth ? and !
    }
    return 0;
}

inline bool is_ascii_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

// SentencePiece word-boundary marker U+2581, used by SenseVoice tokens.
inline bool is_sentencepiece_space(const std::string& s, size_t pos) {
    return pos + 3 <= s.size() && static_cast<unsigned char>(s[pos]) == 0xE2 &&
           static_cast<unsigned char>(s[pos + 1]) == 0x96 && static_cast<unsigned char>(s[pos + 2]) == 0x81;
}

inline std::string trim_copy(const std::string& s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && is_ascii_space(s[b])) ++b;
    while (e > b && is_ascii_space(s[e - 1])) --e;
    return s.substr(b, e - b);
}

inline std::string squash_spaces(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (!is_ascii_space(c)) out.push_back(c);
    }
    return out;
}

// Returns the offset just past a run of sentence punctuation starting at pos,
// or npos when there is none. An ASCII '.' only counts when it is followed by
// whitespace, so decimals and abbreviations glued to the next token are kept.
inline size_t consume_sentence_end(const std::string& s, size_t pos) {
    size_t cur = pos;
    bool found = false;
    while (cur < s.size()) {
        size_t len = sentence_punct_len(s, cur);
        if (len == 0) break;
        cur += len;
        found = true;
    }
    if (!found) return std::string::npos;
    if (s[pos] == '.' && cur < s.size() && !is_ascii_space(s[cur]) && !is_sentencepiece_space(s, cur)) {
        return std::string::npos;
    }
    return cur;
}

// Finds the latest sentence boundary that is stable between the previous and
// current hypotheses and has at least one word after it.
inline std::optional<CommitPlan> plan_commit(const Hypothesis& h, const std::string& prev_text, const Options& o) {
    if (!o.early_commit || !h.words_timed || h.words.size() < 2 || prev_text.empty()) {
        return std::nullopt;
    }

    // Map every word to its end offset in the text. Abort when a word cannot be
    // located, because a wrong mapping would cut audio at the wrong place.
    std::vector<size_t> word_end(h.words.size(), 0);
    size_t cursor = 0;
    const size_t max_skip = 16;
    for (size_t k = 0; k < h.words.size(); ++k) {
        const std::string w = trim_copy(h.words[k].text);
        if (w.empty()) {
            word_end[k] = cursor;
            continue;
        }
        size_t pos = h.text.find(w, cursor);
        if (pos == std::string::npos || pos - cursor > max_skip) {
            return std::nullopt;
        }
        cursor = pos + w.size();
        word_end[k] = cursor;
    }

    const std::string prev_squashed = squash_spaces(prev_text);
    for (size_t k = h.words.size() - 1; k-- > 0;) {
        // The punctuation may be part of the word itself or follow it.
        size_t end = std::string::npos;
        const size_t we = word_end[k];
        size_t after = we;
        while (after < h.text.size() && is_ascii_space(h.text[after])) ++after;
        size_t trailing = consume_sentence_end(h.text, after);
        if (trailing != std::string::npos) {
            end = trailing;
        } else if (we > 0) {
            // Walk back over the punctuation run that ends the word.
            size_t start = we;
            while (start > 0) {
                size_t step = 0;
                if (start >= 3 && sentence_punct_len(h.text, start - 3) == 3) step = 3;
                else if (sentence_punct_len(h.text, start - 1) == 1) step = 1;
                if (step == 0) break;
                start -= step;
            }
            if (start < we && consume_sentence_end(h.text, start) != std::string::npos) {
                end = we;
            }
        }
        if (end == std::string::npos) continue;

        const Word& head_last = h.words[k];
        const Word& tail_first = h.words[k + 1];
        if (head_last.t1_ms - h.start_ms < o.early_commit_min_ms) {
            // Earlier boundaries are even shorter.
            break;
        }

        const std::string head = h.text.substr(0, end);
        const std::string head_squashed = squash_spaces(head);
        if (head_squashed.empty() || prev_squashed.compare(0, head_squashed.size(), head_squashed) != 0) {
            continue;
        }

        int64_t gap = tail_first.t0_ms - head_last.t1_ms;
        int64_t cut = head_last.t1_ms + std::max<int64_t>(0, gap) / 2;
        if (cut <= h.start_ms || cut >= h.end_ms - o.early_commit_min_remaining_ms) {
            continue;
        }

        // Require real content after the boundary.
        if (trim_copy(h.text.substr(end)).empty()) {
            continue;
        }

        CommitPlan plan;
        plan.text_end = end;
        plan.n_words = k + 1;
        plan.commit_end_ms = head_last.t1_ms;
        plan.cut_ms = cut;
        return plan;
    }
    return std::nullopt;
}

inline void split_hypothesis(const Hypothesis& h, const CommitPlan& plan, Hypothesis& head, Hypothesis& tail) {
    head = Hypothesis{};
    tail = Hypothesis{};
    head.start_ms = h.start_ms;
    head.end_ms = plan.commit_end_ms;
    head.text = trim_copy(h.text.substr(0, plan.text_end));
    head.words.assign(h.words.begin(), h.words.begin() + plan.n_words);
    head.words_timed = h.words_timed;
    head.language = h.language;
    head.emotion = h.emotion;
    head.event = h.event;

    tail.start_ms = plan.cut_ms;
    tail.end_ms = h.end_ms;
    tail.text = trim_copy(h.text.substr(plan.text_end));
    tail.words.assign(h.words.begin() + plan.n_words, h.words.end());
    tail.words_timed = h.words_timed;
    tail.language = h.language;
    tail.emotion = h.emotion;
    tail.event = h.event;
}

} // namespace stream_seg
