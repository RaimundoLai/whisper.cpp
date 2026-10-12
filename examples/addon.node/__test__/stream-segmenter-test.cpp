// Standalone tests for stream-segmenter.h.
// Build and run:
//   c++ -std=c++17 -I.. stream-segmenter-test.cpp -o /tmp/stream-segmenter-test && /tmp/stream-segmenter-test

#include "../stream-segmenter.h"

#include <cassert>
#include <cstdio>

using namespace stream_seg;

static Word w(const char* text, int64_t t0, int64_t t1) {
    Word word;
    word.text = text;
    word.t0_ms = t0;
    word.t1_ms = t1;
    return word;
}

static Options commit_options() {
    Options o;
    o.early_commit = true;
    o.early_commit_min_ms = 1000;
    o.early_commit_min_remaining_ms = 200;
    return o;
}

static void test_silence_threshold() {
    std::printf("silence_threshold\n");
    Options o;
    // Disabled: base count unchanged.
    assert(silence_threshold(o, 5, 100, 200) == 5);

    o.adaptive_silence = true;
    // Short speech keeps the base threshold.
    assert(silence_threshold(o, 5, 100, 10) == 5);
    // 5-10 s: 350 ms -> ceil(3.5) = 4 chunks -> T = 3.
    assert(silence_threshold(o, 5, 100, 60) == 3);
    // >= 10 s: 200 ms -> 2 chunks -> T = 1.
    assert(silence_threshold(o, 5, 100, 100) == 1);
    // 32 ms chunks: 200 ms -> 7 chunks -> T = 6.
    assert(silence_threshold(o, 15, 32, 400) == 6);
    // Never longer than the base threshold.
    assert(silence_threshold(o, 2, 100, 60) == 2);
    // Never zero.
    o.adaptive_soft_silence_ms = 10;
    assert(silence_threshold(o, 5, 100, 100) == 1);
}

static void test_trailing_trim() {
    std::printf("trailing_trim\n");
    assert(trailing_trim(8000, 4800) == 3200);
    assert(trailing_trim(4800, 4800) == 0);
    assert(trailing_trim(1000, 4800) == 0);
}

static void test_tail_cache() {
    std::printf("TailCache\n");
    TailCache<int> cache;
    assert(!cache.matches(10, 0, 100));
    cache.valid = true;
    cache.pcm_size = 10;
    cache.start_ms = 0;
    cache.end_ms = 100;
    cache.result = 7;
    assert(cache.matches(10, 0, 100));
    assert(!cache.matches(11, 0, 100));
    assert(!cache.matches(10, 0, 101));
    cache.reset();
    assert(!cache.valid && cache.result == 0);
}

static Hypothesis english_hypothesis() {
    Hypothesis h;
    h.start_ms = 0;
    h.end_ms = 4000;
    h.text = "Hello there. How are you";
    h.words = {w("Hello", 0, 500), w("there.", 550, 1200), w("How", 1500, 1800), w("are", 1850, 2100), w("you", 2150, 2500)};
    h.words_timed = true;
    return h;
}

static void test_commit_english() {
    std::printf("plan_commit english\n");
    Options o = commit_options();
    Hypothesis h = english_hypothesis();

    auto plan = plan_commit(h, "Hello there. How", o);
    assert(plan.has_value());
    assert(plan->n_words == 2);
    assert(h.text.substr(0, plan->text_end) == "Hello there.");
    assert(plan->commit_end_ms == 1200);
    assert(plan->cut_ms == 1350);

    Hypothesis head, tail;
    split_hypothesis(h, *plan, head, tail);
    assert(head.text == "Hello there.");
    assert(head.start_ms == 0 && head.end_ms == 1200);
    assert(head.words.size() == 2);
    assert(tail.text == "How are you");
    assert(tail.start_ms == 1350 && tail.end_ms == 4000);
    assert(tail.words.size() == 3);
}

static void test_commit_requires_stability() {
    std::printf("plan_commit stability\n");
    Options o = commit_options();
    Hypothesis h = english_hypothesis();
    assert(!plan_commit(h, "", o).has_value());
    assert(!plan_commit(h, "Hello their. How", o).has_value());
    // Previous hypothesis without the punctuation yet.
    assert(!plan_commit(h, "Hello there", o).has_value());
    // Whitespace differences are ignored.
    assert(plan_commit(h, "Hello  there.How", o).has_value());
}

static void test_commit_guards() {
    std::printf("plan_commit guards\n");
    Options o = commit_options();
    Hypothesis h = english_hypothesis();

    Options disabled = o;
    disabled.early_commit = false;
    assert(!plan_commit(h, h.text, disabled).has_value());

    Hypothesis untimed = h;
    untimed.words_timed = false;
    assert(!plan_commit(untimed, h.text, o).has_value());

    Options long_min = o;
    long_min.early_commit_min_ms = 2000;
    assert(!plan_commit(h, h.text, long_min).has_value());

    // Sentence end at the very end has no following word.
    Hypothesis ended = h;
    ended.text = "Hello there.";
    ended.words = {w("Hello", 0, 500), w("there.", 550, 1200)};
    assert(!plan_commit(ended, ended.text, o).has_value());

    // Remaining audio too short.
    Hypothesis short_tail = h;
    short_tail.end_ms = 1500;
    assert(!plan_commit(short_tail, h.text, o).has_value());

    // Unmappable words abort the plan.
    Hypothesis mismatch = h;
    mismatch.words[2].text = "Who";
    assert(!plan_commit(mismatch, h.text, o).has_value());

    // Decimal point is not a sentence end.
    Hypothesis decimal;
    decimal.start_ms = 0;
    decimal.end_ms = 4000;
    decimal.text = "It costs 3.5 dollars today";
    decimal.words = {w("It", 0, 300), w("costs", 300, 1100), w("3.5", 1200, 1600), w("dollars", 1700, 2200), w("today", 2300, 2800)};
    decimal.words_timed = true;
    assert(!plan_commit(decimal, decimal.text, o).has_value());

    // SentencePiece boundary after a period counts as whitespace.
    Hypothesis piece;
    piece.start_ms = 0;
    piece.end_ms = 4000;
#define SPM "\xE2\x96\x81"
    piece.text = SPM "Hi" SPM "there." SPM "How" SPM "are";
    piece.words = {w(SPM "Hi", 0, 500), w(SPM "there.", 600, 1300),
                   w(SPM "How", 1500, 1800), w(SPM "are", 1850, 2100)};
#undef SPM
    piece.words_timed = true;
    auto piece_plan = plan_commit(piece, piece.text, o);
    assert(piece_plan.has_value() && piece_plan->n_words == 2);
}

static void test_commit_cjk_aligner_words() {
    std::printf("plan_commit cjk\n");
    Options o = commit_options();
    // Aligner words carry no punctuation; it lives only in the text.
    Hypothesis h;
    h.start_ms = 1000;
    h.end_ms = 5000;
    h.text = "\xE4\xBD\xA0\xE5\xA5\xBD\xE3\x80\x82\xE6\x88\x91\xE5\x80\x91\xE9\x96\x8B\xE5\xA7\x8B";
    h.words = {
        w("\xE4\xBD\xA0", 1000, 1500), w("\xE5\xA5\xBD", 1500, 2200),
        w("\xE6\x88\x91", 2400, 2700), w("\xE5\x80\x91", 2700, 3000),
        w("\xE9\x96\x8B", 3000, 3300), w("\xE5\xA7\x8B", 3300, 3600)};
    h.words_timed = true;

    auto plan = plan_commit(h, h.text, o);
    assert(plan.has_value());
    assert(plan->n_words == 2);
    assert(plan->cut_ms == 2300);
    Hypothesis head, tail;
    split_hypothesis(h, *plan, head, tail);
    assert(head.text == "\xE4\xBD\xA0\xE5\xA5\xBD\xE3\x80\x82");
    assert(tail.text == "\xE6\x88\x91\xE5\x80\x91\xE9\x96\x8B\xE5\xA7\x8B");
    assert(tail.start_ms == 2300);

    // Fullwidth question mark.
    Hypothesis q = h;
    q.text = "\xE4\xBD\xA0\xE5\xA5\xBD\xEF\xBC\x9F\xE6\x88\x91\xE5\x80\x91\xE9\x96\x8B\xE5\xA7\x8B";
    assert(plan_commit(q, q.text, o).has_value());
    // Comma is not a sentence end.
    Hypothesis comma = h;
    comma.text = "\xE4\xBD\xA0\xE5\xA5\xBD\xEF\xBC\x8C\xE6\x88\x91\xE5\x80\x91\xE9\x96\x8B\xE5\xA7\x8B";
    assert(!plan_commit(comma, comma.text, o).has_value());
}

static void test_commit_picks_latest_boundary() {
    std::printf("plan_commit latest boundary\n");
    Options o = commit_options();
    Hypothesis h;
    h.start_ms = 0;
    h.end_ms = 6000;
    h.text = "One. Two. Three";
    h.words = {w("One.", 0, 1100), w("Two.", 1300, 2500), w("Three", 2700, 3200)};
    h.words_timed = true;
    auto plan = plan_commit(h, h.text, o);
    assert(plan.has_value() && plan->n_words == 2);
    // Only the first boundary is stable.
    auto earlier = plan_commit(h, "One. Too", o);
    assert(earlier.has_value() && earlier->n_words == 1);
}

int main() {
    test_silence_threshold();
    test_trailing_trim();
    test_tail_cache();
    test_commit_english();
    test_commit_requires_stability();
    test_commit_guards();
    test_commit_cjk_aligner_words();
    test_commit_picks_latest_boundary();
    std::printf("All stream-segmenter tests passed\n");
    return 0;
}
