#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <fstream>
#include <iostream>
#include <iterator>
#include <exception>
#include <limits>
#include <atomic>
#include <mutex>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

static constexpr const char * DEFAULT_TRAIN_TEXT = "TinyStories-train.txt";
static constexpr const char * DEFAULT_VALID_TEXT = "TinyStories-valid.txt";
static constexpr const char * DEFAULT_TRAIN_TOKENS = "TinyStories-train.regex.u16";
static constexpr const char * DEFAULT_VALID_TOKENS = "TinyStories-valid.regex.u16";
static constexpr const char * DEFAULT_CHECKPOINT = "tinystories_metal_llm_v2.bin";
static constexpr const char * DEFAULT_LOG = "tinystories_metal_train_v2.tsv";
static constexpr const char * SPECIAL = "<|endoftext|>";

enum class TokenizerKind {
    RegexBpe,
    SuperBpe,
    BltPatch,
};

static TokenizerKind parse_tokenizer_kind(const std::string & name) {
    if (name == "regex-bpe" || name == "bpe" || name == "regex") {
        return TokenizerKind::RegexBpe;
    }
    if (name == "superbpe" || name == "super-bpe" || name == "boundary-relaxed-bpe") {
        return TokenizerKind::SuperBpe;
    }
    if (name == "blt" || name == "byte-latent" || name == "patch") {
        return TokenizerKind::BltPatch;
    }
    throw std::runtime_error("unknown tokenizer kind: " + name);
}

static const char * tokenizer_kind_name(TokenizerKind kind) {
    switch (kind) {
        case TokenizerKind::RegexBpe: return "regex-bpe";
        case TokenizerKind::SuperBpe: return "superbpe";
        case TokenizerKind::BltPatch: return "blt";
    }
    return "unknown";
}

static std::string lowercase_ascii(std::string s) {
    for (char & ch : s) {
        if (ch >= 'A' && ch <= 'Z') {
            ch = char(ch - 'A' + 'a');
        }
    }
    return s;
}

static bool is_openwebtext_dataset(const std::string & dataset) {
    const std::string d = lowercase_ascii(dataset);
    return d == "openwebtext" || d == "open-webtext" || d == "owt";
}

static int default_tokenizer_vocab_size(const std::string & dataset) {
    return is_openwebtext_dataset(dataset) ? 32000 : 10000;
}

static std::string vocab_suffix(int vocab_size) {
    if (vocab_size % 1000 == 0) {
        return std::to_string(vocab_size / 1000) + "k";
    }
    return std::to_string(vocab_size);
}

static std::string tokenizer_suffix(TokenizerKind kind) {
    switch (kind) {
        case TokenizerKind::RegexBpe: return "bpe";
        case TokenizerKind::SuperBpe: return "superbpe";
        case TokenizerKind::BltPatch: return "blt";
    }
    return "bpe";
}

static std::string default_tokenizer_dir(TokenizerKind kind, const std::string & dataset = "tinystories", int vocab_size = 10000) {
    if (is_openwebtext_dataset(dataset)) {
        return "openwebtext_" + tokenizer_suffix(kind) + "_" + vocab_suffix(vocab_size);
    }
    switch (kind) {
        case TokenizerKind::RegexBpe: return "tinystories_bpe_10k";
        case TokenizerKind::SuperBpe: return "tinystories_superbpe_10k";
        case TokenizerKind::BltPatch: return "tinystories_blt_10k";
    }
    return "tinystories_bpe_10k";
}

static std::string default_train_text(const std::string & dataset) {
    return is_openwebtext_dataset(dataset) ? "OpenWebText-train.txt" : DEFAULT_TRAIN_TEXT;
}

static std::string default_valid_text(const std::string & dataset) {
    return is_openwebtext_dataset(dataset) ? "OpenWebText-valid.txt" : DEFAULT_VALID_TEXT;
}

static std::string default_train_tokens(TokenizerKind kind, const std::string & dataset = "tinystories") {
    if (is_openwebtext_dataset(dataset)) {
        switch (kind) {
            case TokenizerKind::RegexBpe: return "OpenWebText-train.regex.u16";
            case TokenizerKind::SuperBpe: return "OpenWebText-train.superbpe.u16";
            case TokenizerKind::BltPatch: return "OpenWebText-train.blt.u16";
        }
    }
    switch (kind) {
        case TokenizerKind::RegexBpe: return "TinyStories-train.regex.u16";
        case TokenizerKind::SuperBpe: return "TinyStories-train.superbpe.u16";
        case TokenizerKind::BltPatch: return "TinyStories-train.blt.u16";
    }
    return DEFAULT_TRAIN_TOKENS;
}

static std::string default_valid_tokens(TokenizerKind kind, const std::string & dataset = "tinystories") {
    if (is_openwebtext_dataset(dataset)) {
        switch (kind) {
            case TokenizerKind::RegexBpe: return "OpenWebText-valid.regex.u16";
            case TokenizerKind::SuperBpe: return "OpenWebText-valid.superbpe.u16";
            case TokenizerKind::BltPatch: return "OpenWebText-valid.blt.u16";
        }
    }
    switch (kind) {
        case TokenizerKind::RegexBpe: return "TinyStories-valid.regex.u16";
        case TokenizerKind::SuperBpe: return "TinyStories-valid.superbpe.u16";
        case TokenizerKind::BltPatch: return "TinyStories-valid.blt.u16";
    }
    return DEFAULT_VALID_TOKENS;
}

static std::string default_checkpoint_path(const std::string & dataset, TokenizerKind kind) {
    if (is_openwebtext_dataset(dataset)) {
        return "openwebtext_metal_" + tokenizer_suffix(kind) + "_v1.bin";
    }
    return DEFAULT_CHECKPOINT;
}

static std::string default_log_path(const std::string & dataset, TokenizerKind kind) {
    if (is_openwebtext_dataset(dataset)) {
        return "openwebtext_metal_" + tokenizer_suffix(kind) + "_train_v1.tsv";
    }
    return DEFAULT_LOG;
}

static uint64_t pack_pair(uint32_t a, uint32_t b) {
    return (uint64_t(a) << 32) | uint64_t(b);
}

static bool file_exists(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

static std::string get_arg(int argc, char ** argv, const std::string & name, const std::string & def) {
    for (int i = 2; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return def;
}

static bool get_arg_if_present(int argc, char ** argv, const std::string & name, std::string & value) {
    for (int i = 2; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            value = argv[i + 1];
            return true;
        }
    }
    return false;
}

static int get_arg_i(int argc, char ** argv, const std::string & name, int def) {
    return std::stoi(get_arg(argc, argv, name, std::to_string(def)));
}

static uint64_t get_arg_u64(int argc, char ** argv, const std::string & name, uint64_t def) {
    return std::stoull(get_arg(argc, argv, name, std::to_string(def)));
}

static float get_arg_f(int argc, char ** argv, const std::string & name, float def) {
    return std::stof(get_arg(argc, argv, name, std::to_string(def)));
}

static bool has_arg(int argc, char ** argv, const std::string & name) {
    for (int i = 2; i < argc; ++i) {
        if (argv[i] == name) {
            return true;
        }
    }
    return false;
}

static bool get_arg_bool(int argc, char ** argv, const std::string & name, bool def) {
    if (!has_arg(argc, argv, name)) {
        return def;
    }
    const std::string value = get_arg(argc, argv, name, def ? "1" : "0");
    return value == "1" || value == "true" || value == "yes" || value == "on";
}

static uint64_t get_arg_bytes(int argc, char ** argv, const std::string & name, uint64_t def) {
    std::string value = get_arg(argc, argv, name, std::to_string(def));
    if (value.empty()) {
        return def;
    }
    char suffix = value.back();
    uint64_t multiplier = 1;
    if (suffix == 'k' || suffix == 'K' || suffix == 'm' || suffix == 'M' ||
        suffix == 'g' || suffix == 'G' || suffix == 't' || suffix == 'T') {
        value.pop_back();
        switch (suffix) {
            case 'k': case 'K': multiplier = 1024ull; break;
            case 'm': case 'M': multiplier = 1024ull * 1024ull; break;
            case 'g': case 'G': multiplier = 1024ull * 1024ull * 1024ull; break;
            case 't': case 'T': multiplier = 1024ull * 1024ull * 1024ull * 1024ull; break;
        }
    }
    return uint64_t(std::stod(value) * double(multiplier));
}

static bool is_space(uint8_t b) {
    return b == ' ' || b == '\n' || b == '\r' || b == '\t' || b == 0x0b || b == 0x0c;
}

static size_t utf8_advance(const uint8_t * data, size_t n, size_t i) {
    if (i >= n) {
        return i;
    }
    const uint8_t b = data[i];
    size_t len = 1;
    if ((b & 0xe0) == 0xc0) {
        len = 2;
    } else if ((b & 0xf0) == 0xe0) {
        len = 3;
    } else if ((b & 0xf8) == 0xf0) {
        len = 4;
    }
    return std::min(n, i + len);
}

static bool is_patch_stop_byte(uint8_t b) {
    return b == '\n' || b == '.' || b == '!' || b == '?' || b == ';' || b == ':';
}

static void dynamic_byte_patches(
    const uint8_t * data,
    size_t n,
    int max_patch_bytes,
    std::vector<std::vector<uint8_t>> & out) {
    const size_t max_len = size_t(std::max(1, max_patch_bytes));
    size_t i = 0;
    while (i < n) {
        const size_t start = i;
        size_t last = i;
        while (i < n && i - start < max_len) {
            const uint8_t b = data[i];
            const size_t next = utf8_advance(data, n, i);
            if (next - start > max_len && i > start) {
                break;
            }
            i = next;
            last = i;
            if (is_patch_stop_byte(b)) {
                break;
            }
            if (i < n && i - start >= 4 && is_space(data[i - 1]) && !is_space(data[i])) {
                break;
            }
        }
        if (last == start) {
            last = utf8_advance(data, n, start);
            i = last;
        }
        out.emplace_back(data + start, data + last);
    }
}

static NSRegularExpression * gpt2_pretoken_regex() {
    static NSRegularExpression * regex = nil;
    static std::once_flag once;
    std::call_once(once, [] {
        NSError * error = nil;
        NSString * pattern = @"'(?:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
        regex = [NSRegularExpression regularExpressionWithPattern:pattern options:0 error:&error];
        if (!regex) {
            NSString * message = error ? [error localizedDescription] : @"unknown regex error";
            throw std::runtime_error(std::string("failed to compile GPT-2 pretokenization regex: ") +
                                     [message UTF8String]);
        }
    });
    return regex;
}

template <typename Fn>
static void for_each_gpt2_pretoken(const uint8_t * data, size_t n, Fn && fn) {
    if (n == 0) {
        return;
    }
    @autoreleasepool {
        NSString * text = [[NSString alloc] initWithBytes:data length:n encoding:NSUTF8StringEncoding];
        if (!text) {
            for (size_t i = 0; i < n; ++i) {
                fn(data + i, 1);
            }
            return;
        }
        NSRegularExpression * regex = gpt2_pretoken_regex();
        NSRange range = NSMakeRange(0, [text length]);
        [regex enumerateMatchesInString:text options:0 range:range usingBlock:^(NSTextCheckingResult * match, NSMatchingFlags, BOOL *) {
            if (match.range.length == 0) {
                return;
            }
            NSString * piece = [text substringWithRange:match.range];
            NSData * bytes = [piece dataUsingEncoding:NSUTF8StringEncoding];
            if ([bytes length] > 0) {
                fn(reinterpret_cast<const uint8_t *>([bytes bytes]), [bytes length]);
            }
        }];
    }
}

template <typename Fn>
static void for_each_non_special_segment(const std::string & text, Fn && fn, uint64_t * special_count = nullptr) {
    size_t start = 0;
    while (start < text.size()) {
        const size_t pos = text.find(SPECIAL, start);
        const size_t end = pos == std::string::npos ? text.size() : pos;
        if (end > start) {
            fn(reinterpret_cast<const uint8_t *>(text.data() + start), end - start);
        }
        if (pos == std::string::npos) {
            break;
        }
        if (special_count) {
            ++*special_count;
        }
        start = pos + std::strlen(SPECIAL);
    }
}

static void ensure_directory(const std::string & dir) {
    if (::mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
        throw std::runtime_error("failed to create output dir: " + dir);
    }
}

static int normalize_thread_count(int requested) {
    if (requested > 0) {
        return std::max(1, requested);
    }
    const unsigned hw = std::thread::hardware_concurrency();
    return std::max(1u, hw == 0 ? 4u : hw);
}

struct TextMMap {
    int fd = -1;
    size_t bytes = 0;
    const char * data = nullptr;

    explicit TextMMap(const std::string & path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("failed to open text input: " + path);
        }
        struct stat st {};
        if (::fstat(fd, &st) != 0) {
            throw std::runtime_error("failed to stat text input: " + path);
        }
        if (st.st_size <= 0) {
            throw std::runtime_error("text input is empty: " + path);
        }
        bytes = size_t(st.st_size);
        void * p = ::mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) {
            throw std::runtime_error("failed to mmap text input: " + path);
        }
        data = reinterpret_cast<const char *>(p);
    }

    TextMMap(const TextMMap &) = delete;
    TextMMap & operator=(const TextMMap &) = delete;

    ~TextMMap() {
        if (data) {
            ::munmap((void *)data, bytes);
        }
        if (fd >= 0) {
            ::close(fd);
        }
    }
};

struct ByteRange {
    size_t begin = 0;
    size_t end = 0;
    bool special_after = false;
};

static std::vector<ByteRange> split_document_ranges(const char * data, size_t n, uint64_t & special_count) {
    std::vector<ByteRange> ranges;
    ranges.reserve(1 << 20);
    special_count = 0;
    const char * special_begin = SPECIAL;
    const char * special_end = SPECIAL + std::strlen(SPECIAL);
    const char * cur = data;
    const char * end = data + n;
    while (cur < end) {
        const char * found = std::search(cur, end, special_begin, special_end);
        if (found > cur) {
            ranges.push_back({size_t(cur - data), size_t(found - data), found != end});
        }
        if (found == end) {
            break;
        }
        ++special_count;
        cur = found + std::strlen(SPECIAL);
    }
    return ranges;
}

static uint64_t restrict_ranges_to_byte_budget(
    std::vector<ByteRange> & ranges,
    uint64_t byte_budget,
    uint64_t & special_count) {
    if (byte_budget == 0) {
        uint64_t total = 0;
        uint64_t specials = 0;
        for (const ByteRange & r : ranges) {
            total += uint64_t(r.end - r.begin);
            if (r.special_after) {
                ++specials;
            }
        }
        special_count = specials;
        return total;
    }

    uint64_t total = 0;
    uint64_t specials = 0;
    size_t keep = 0;
    for (const ByteRange & r : ranges) {
        const uint64_t doc_bytes = uint64_t(r.end - r.begin);
        if (keep > 0 && total + doc_bytes > byte_budget) {
            break;
        }
        total += doc_bytes;
        if (r.special_after) {
            ++specials;
        }
        ++keep;
        if (total >= byte_budget) {
            break;
        }
    }
    ranges.resize(keep);
    special_count = specials;
    return total;
}

static void report_parallel_progress(
    const std::string & label,
    std::atomic<uint64_t> & processed,
    std::atomic<uint64_t> & next_report,
    uint64_t delta_bytes,
    size_t units_done) {
    const uint64_t value = processed.fetch_add(delta_bytes, std::memory_order_relaxed) + delta_bytes;
    uint64_t threshold = next_report.load(std::memory_order_relaxed);
    while (value >= threshold) {
        if (next_report.compare_exchange_weak(
                threshold,
                threshold + 256ull * 1024ull * 1024ull,
                std::memory_order_relaxed)) {
            std::cerr << label << " " << (double(value) / (1024.0 * 1024.0 * 1024.0))
                      << " GiB; units=" << units_done << "\n";
            break;
        }
    }
}

struct BpeCandidate {
    uint32_t rank = 0;
    int left_pos = 0;
    int right_pos = 0;
    uint32_t left_id = 0;
    uint32_t right_id = 0;
};

struct BpeCandidateGreater {
    bool operator()(const BpeCandidate & a, const BpeCandidate & b) const {
        if (a.rank != b.rank) {
            return a.rank > b.rank;
        }
        return a.left_pos > b.left_pos;
    }
};

static void apply_bpe_merges_in_place(
    std::vector<uint32_t> & ids,
    const std::unordered_map<uint64_t, uint32_t> & merge_to_id,
    const std::unordered_map<uint64_t, uint32_t> & merge_rank) {
    const int n = int(ids.size());
    if (n < 2 || merge_rank.empty()) {
        return;
    }

    std::vector<int> prev(size_t(n), -1);
    std::vector<int> next(size_t(n), -1);
    std::vector<uint8_t> active(size_t(n), 1);
    for (int i = 0; i < n; ++i) {
        prev[size_t(i)] = i - 1;
        next[size_t(i)] = (i + 1 < n) ? i + 1 : -1;
    }

    std::priority_queue<BpeCandidate, std::vector<BpeCandidate>, BpeCandidateGreater> queue;
    auto push_pair = [&](int left) {
        if (left < 0 || !active[size_t(left)]) {
            return;
        }
        const int right = next[size_t(left)];
        if (right < 0 || !active[size_t(right)]) {
            return;
        }
        const uint64_t pair = pack_pair(ids[size_t(left)], ids[size_t(right)]);
        auto rank_it = merge_rank.find(pair);
        if (rank_it == merge_rank.end()) {
            return;
        }
        queue.push({rank_it->second, left, right, ids[size_t(left)], ids[size_t(right)]});
    };

    for (int i = 0; i + 1 < n; ++i) {
        push_pair(i);
    }

    while (!queue.empty()) {
        const BpeCandidate cand = queue.top();
        queue.pop();
        if (!active[size_t(cand.left_pos)] || !active[size_t(cand.right_pos)] ||
            next[size_t(cand.left_pos)] != cand.right_pos ||
            ids[size_t(cand.left_pos)] != cand.left_id ||
            ids[size_t(cand.right_pos)] != cand.right_id) {
            continue;
        }
        const uint64_t pair = pack_pair(cand.left_id, cand.right_id);
        auto id_it = merge_to_id.find(pair);
        if (id_it == merge_to_id.end()) {
            continue;
        }

        const int left = cand.left_pos;
        const int right = cand.right_pos;
        const int before = prev[size_t(left)];
        const int after = next[size_t(right)];
        ids[size_t(left)] = id_it->second;
        active[size_t(right)] = 0;
        next[size_t(left)] = after;
        if (after >= 0) {
            prev[size_t(after)] = left;
        }

        push_pair(before);
        push_pair(left);
    }

    std::vector<uint32_t> compact;
    compact.reserve(ids.size());
    for (int i = 0; i >= 0; i = next[size_t(i)]) {
        if (active[size_t(i)]) {
            compact.push_back(ids[size_t(i)]);
        }
    }
    ids.swap(compact);
}

struct Tokenizer {
    TokenizerKind kind = TokenizerKind::RegexBpe;
    std::vector<std::vector<uint8_t>> vocab;
    std::unordered_map<std::string, uint32_t> bytes_to_id;
    std::unordered_map<uint64_t, uint32_t> merge_to_id;
    std::unordered_map<uint64_t, uint32_t> merge_rank;
    uint32_t special_id = 256;
    int max_patch_bytes = 8;

    void load(const std::string & vocab_path, const std::string & merges_path, TokenizerKind tokenizer_kind = TokenizerKind::RegexBpe) {
        kind = tokenizer_kind;
        load_vocab(vocab_path);
        if (kind == TokenizerKind::BltPatch) {
            merge_to_id.clear();
            merge_rank.clear();
        } else {
            load_merges(merges_path);
        }
    }

    void load_dir(const std::string & dir, TokenizerKind tokenizer_kind) {
        const std::string vocab_path = dir + "/vocab.json";
        const std::string merges_path = dir + "/merges.tsv";
        load(vocab_path, merges_path, tokenizer_kind);
    }

    void load_vocab(const std::string & path) {
        std::ifstream in(path);
        if (!in) {
            throw std::runtime_error("failed to open vocab: " + path);
        }
        vocab.clear();
        bytes_to_id.clear();

        std::string line;
        while (std::getline(in, line)) {
            const size_t q1 = line.find('"');
            if (q1 == std::string::npos) {
                continue;
            }
            const size_t q2 = line.find('"', q1 + 1);
            if (q2 == std::string::npos) {
                continue;
            }
            const int id = std::stoi(line.substr(q1 + 1, q2 - q1 - 1));
            const size_t lb = line.find('[', q2);
            const size_t rb = line.find(']', lb);
            if (lb == std::string::npos || rb == std::string::npos) {
                continue;
            }
            if ((int)vocab.size() <= id) {
                vocab.resize(id + 1);
            }
            std::vector<uint8_t> bytes;
            size_t p = lb + 1;
            while (p < rb) {
                while (p < rb && (line[p] == ' ' || line[p] == ',')) {
                    ++p;
                }
                if (p >= rb) {
                    break;
                }
                size_t e = p;
                while (e < rb && line[e] >= '0' && line[e] <= '9') {
                    ++e;
                }
                if (e > p) {
                    bytes.push_back(uint8_t(std::stoi(line.substr(p, e - p))));
                }
                p = e + 1;
            }
            vocab[id] = bytes;
        }

        for (uint32_t id = 0; id < vocab.size(); ++id) {
            std::string key(reinterpret_cast<const char *>(vocab[id].data()), vocab[id].size());
            bytes_to_id[key] = id;
            if (key == SPECIAL) {
                special_id = id;
            }
        }
        if (vocab.empty()) {
            throw std::runtime_error("vocab was empty: " + path);
        }
    }

    void load_merges(const std::string & path) {
        std::ifstream in(path);
        if (!in) {
            throw std::runtime_error("failed to open merges: " + path);
        }
        merge_to_id.clear();
        merge_rank.clear();

        std::string line;
        uint32_t rank = 0;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            uint32_t new_id = 0, left = 0, right = 0;
            if (std::sscanf(line.c_str(), "%u\t%u\t%u", &new_id, &left, &right) == 3) {
                const uint64_t key = pack_pair(left, right);
                merge_to_id[key] = new_id;
                merge_rank[key] = rank++;
            }
        }
    }

    void encode_regex_span(const uint8_t * data, size_t n, std::vector<uint32_t> & out) const {
        for_each_gpt2_pretoken(data, n, [&](const uint8_t * piece, size_t len) {
            encode_pretoken(piece, len, out);
        });
    }

    void encode_blt_span(const uint8_t * data, size_t n, std::vector<uint32_t> & out) const {
        std::vector<std::vector<uint8_t>> patches;
        dynamic_byte_patches(data, n, max_patch_bytes, patches);
        for (const auto & patch : patches) {
            std::string key(reinterpret_cast<const char *>(patch.data()), patch.size());
            auto it = bytes_to_id.find(key);
            if (it != bytes_to_id.end()) {
                out.push_back(it->second);
            } else {
                for (uint8_t b : patch) {
                    out.push_back(uint32_t(b));
                }
            }
        }
    }

    void encode_pretoken(const uint8_t * data, size_t n, std::vector<uint32_t> & out) const {
        if (n == 0) {
            return;
        }
        std::vector<uint32_t> ids;
        ids.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            ids.push_back(data[i]);
        }

        apply_bpe_merges_in_place(ids, merge_to_id, merge_rank);

        out.insert(out.end(), ids.begin(), ids.end());
    }

    void encode_span(const uint8_t * data, size_t n, std::vector<uint32_t> & out) const {
        if (n == 0) {
            return;
        }
        if (kind == TokenizerKind::RegexBpe) {
            encode_regex_span(data, n, out);
        } else if (kind == TokenizerKind::SuperBpe) {
            encode_pretoken(data, n, out);
        } else {
            encode_blt_span(data, n, out);
        }
    }

    std::vector<uint32_t> encode(const std::string & text) const {
        std::vector<uint32_t> out;
        size_t start = 0;
        while (start < text.size()) {
            const size_t pos = text.find(SPECIAL, start);
            const size_t end = pos == std::string::npos ? text.size() : pos;
            const auto * data = reinterpret_cast<const uint8_t *>(text.data() + start);
            encode_span(data, end - start, out);
            if (pos == std::string::npos) {
                break;
            }
            out.push_back(special_id);
            start = pos + std::strlen(SPECIAL);
        }
        return out;
    }

    std::string decode(const std::vector<uint32_t> & ids) const {
        std::string bytes;
        for (uint32_t id : ids) {
            if (id < vocab.size()) {
                bytes.append(reinterpret_cast<const char *>(vocab[id].data()), vocab[id].size());
            }
        }
        return bytes;
    }
};

static void write_token_ids16(std::ofstream & out, const std::vector<uint32_t> & ids) {
    std::vector<uint16_t> ids16;
    ids16.reserve(ids.size());
    for (uint32_t id : ids) {
        if (id > 65535) {
            throw std::runtime_error("token id does not fit in u16");
        }
        ids16.push_back(uint16_t(id));
    }
    out.write(reinterpret_cast<const char *>(ids16.data()), ids16.size() * sizeof(uint16_t));
}

static void encode_file(const Tokenizer & tok, const std::string & input, const std::string & output, int tokenizer_threads) {
    TextMMap text(input);
    uint64_t special_count = 0;
    std::vector<ByteRange> ranges = split_document_ranges(text.data, text.bytes, special_count);
    const int n_threads = std::min<int>(normalize_thread_count(tokenizer_threads), std::max<size_t>(1, ranges.size()));
    std::vector<std::string> part_paths;
    part_paths.resize(size_t(n_threads));
    std::vector<uint64_t> part_tokens(size_t(n_threads), 0);
    std::vector<std::thread> workers;
    std::vector<std::exception_ptr> errors;
    errors.resize(size_t(n_threads));
    std::mutex error_mutex;
    std::atomic<uint64_t> processed{0};
    std::atomic<uint64_t> next_report{256ull * 1024ull * 1024ull};

    std::cerr << "encoding " << ranges.size() << " documents from " << input
              << " with " << n_threads << " threads\n";
    for (int t = 0; t < n_threads; ++t) {
        part_paths[size_t(t)] = output + ".part" + std::to_string(t);
        const size_t lo = ranges.size() * size_t(t) / size_t(n_threads);
        const size_t hi = ranges.size() * size_t(t + 1) / size_t(n_threads);
        workers.emplace_back([&, t, lo, hi] {
            try {
                @autoreleasepool {
                    std::ofstream out(part_paths[size_t(t)], std::ios::binary | std::ios::trunc);
                    if (!out) {
                        throw std::runtime_error("failed to open token part output: " + part_paths[size_t(t)]);
                    }
                    uint64_t local_tokens = 0;
                    uint64_t local_bytes = 0;
                    for (size_t i = lo; i < hi; ++i) {
                        const ByteRange r = ranges[i];
                        std::vector<uint32_t> ids;
                        ids.reserve(std::min<size_t>(r.end - r.begin, 1ull << 20));
                        tok.encode_span(reinterpret_cast<const uint8_t *>(text.data + r.begin), r.end - r.begin, ids);
                        write_token_ids16(out, ids);
                        local_tokens += ids.size();
                        if (r.special_after) {
                            const uint16_t sid = uint16_t(tok.special_id);
                            out.write(reinterpret_cast<const char *>(&sid), sizeof(sid));
                            ++local_tokens;
                        }
                        local_bytes += r.end - r.begin;
                        if (local_bytes >= 16ull * 1024ull * 1024ull) {
                            report_parallel_progress("encoded", processed, next_report, local_bytes, i + 1);
                            local_bytes = 0;
                        }
                    }
                    if (local_bytes > 0) {
                        report_parallel_progress("encoded", processed, next_report, local_bytes, hi);
                    }
                    part_tokens[size_t(t)] = local_tokens;
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!errors[size_t(t)]) {
                    errors[size_t(t)] = std::current_exception();
                }
            }
        });
    }
    for (std::thread & worker : workers) {
        worker.join();
    }
    for (const auto & err : errors) {
        if (err) {
            std::rethrow_exception(err);
        }
    }

    std::ofstream final_out(output, std::ios::binary | std::ios::trunc);
    if (!final_out) {
        throw std::runtime_error("failed to open token output: " + output);
    }
    uint64_t n_tok = 0;
    for (int t = 0; t < n_threads; ++t) {
        std::ifstream part(part_paths[size_t(t)], std::ios::binary);
        if (!part) {
            throw std::runtime_error("failed to read token part output: " + part_paths[size_t(t)]);
        }
        final_out << part.rdbuf();
        n_tok += part_tokens[size_t(t)];
        part.close();
        std::remove(part_paths[size_t(t)].c_str());
    }
    std::cerr << "wrote " << n_tok << " tokens to " << output << "\n";
}

static std::string json_escape_string(const std::string & s) {
    std::string escaped;
    escaped.reserve(s.size());
    for (char ch : s) {
        switch (ch) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (uint8_t(ch) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", uint8_t(ch));
                    escaped += buf;
                } else {
                    escaped.push_back(ch);
                }
                break;
        }
    }
    return escaped;
}

static void write_tokenizer_vocab_json(const std::string & path, const std::vector<std::vector<uint8_t>> & vocab) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write vocab: " + path);
    }
    out << "{\n";
    for (size_t id = 0; id < vocab.size(); ++id) {
        out << "  \"" << id << "\": [";
        for (size_t i = 0; i < vocab[id].size(); ++i) {
            if (i) {
                out << ",";
            }
            out << int(vocab[id][i]);
        }
        out << "]" << (id + 1 == vocab.size() ? "\n" : ",\n");
    }
    out << "}\n";
}

struct MergeRecord {
    uint32_t new_id = 0;
    uint32_t left = 0;
    uint32_t right = 0;
};

struct BpeWord {
    std::vector<uint32_t> ids;
    uint64_t count = 0;
};

static bool token_bytes_greater(
    const std::vector<uint8_t> & a,
    const std::vector<uint8_t> & b) {
    return std::lexicographical_compare(b.begin(), b.end(), a.begin(), a.end());
}

static bool pair_bytes_greater(
    uint64_t a,
    uint64_t b,
    const std::vector<std::vector<uint8_t>> & vocab) {
    const uint32_t al = uint32_t(a >> 32);
    const uint32_t ar = uint32_t(a);
    const uint32_t bl = uint32_t(b >> 32);
    const uint32_t br = uint32_t(b);
    if (vocab[al] != vocab[bl]) {
        return token_bytes_greater(vocab[al], vocab[bl]);
    }
    return token_bytes_greater(vocab[ar], vocab[br]);
}

static void build_known_merge_maps(
    const std::vector<MergeRecord> & merges,
    std::unordered_map<uint64_t, uint32_t> & merge_to_id,
    std::unordered_map<uint64_t, uint32_t> & merge_rank) {
    merge_to_id.clear();
    merge_rank.clear();
    for (uint32_t i = 0; i < merges.size(); ++i) {
        const uint64_t key = pack_pair(merges[i].left, merges[i].right);
        merge_to_id[key] = merges[i].new_id;
        merge_rank[key] = i;
    }
}

static void apply_ranked_merges(
    std::vector<uint32_t> & ids,
    const std::unordered_map<uint64_t, uint32_t> & merge_to_id,
    const std::unordered_map<uint64_t, uint32_t> & merge_rank) {
    apply_bpe_merges_in_place(ids, merge_to_id, merge_rank);
}

static void collect_regex_pretoken_counts(
    const std::string & input_path,
    std::unordered_map<std::string, uint64_t> & counts,
    uint64_t & bytes_read,
    uint64_t & special_count,
    int tokenizer_threads,
    uint64_t tokenizer_train_bytes) {
    (void)gpt2_pretoken_regex();
    TextMMap input(input_path);
    std::vector<ByteRange> ranges = split_document_ranges(input.data, input.bytes, special_count);
    bytes_read = restrict_ranges_to_byte_budget(ranges, tokenizer_train_bytes, special_count);
    counts.clear();
    counts.reserve(1 << 20);

    const int n_threads = std::min<int>(normalize_thread_count(tokenizer_threads), std::max<size_t>(1, ranges.size()));
    std::vector<std::unordered_map<std::string, uint64_t>> local_counts{size_t(n_threads)};
    std::vector<std::thread> workers;
    workers.reserve(size_t(n_threads));
    std::atomic<uint64_t> processed{0};
    std::atomic<uint64_t> next_report{256ull * 1024ull * 1024ull};
    std::cerr << "BPE pretokenizing " << ranges.size() << " documents with " << n_threads << " threads";
    if (tokenizer_train_bytes > 0) {
        std::cerr << " byte_budget=" << tokenizer_train_bytes;
    }
    std::cerr << "\n";

    for (int t = 0; t < n_threads; ++t) {
        const size_t lo = ranges.size() * size_t(t) / size_t(n_threads);
        const size_t hi = ranges.size() * size_t(t + 1) / size_t(n_threads);
        workers.emplace_back([&, t, lo, hi] {
            @autoreleasepool {
                auto & local = local_counts[size_t(t)];
                local.reserve(1 << 16);
                uint64_t local_bytes = 0;
                for (size_t i = lo; i < hi; ++i) {
                    const ByteRange r = ranges[i];
                    const auto * data = reinterpret_cast<const uint8_t *>(input.data + r.begin);
                    for_each_gpt2_pretoken(data, r.end - r.begin, [&](const uint8_t * piece, size_t len) {
                        ++local[std::string(reinterpret_cast<const char *>(piece), len)];
                    });
                    local_bytes += r.end - r.begin;
                    if (local_bytes >= 8ull * 1024ull * 1024ull) {
                        report_parallel_progress("BPE pretokenized", processed, next_report, local_bytes, i + 1);
                        local_bytes = 0;
                    }
                }
                if (local_bytes > 0) {
                    report_parallel_progress("BPE pretokenized", processed, next_report, local_bytes, hi);
                }
            }
        });
    }
    for (std::thread & worker : workers) {
        worker.join();
    }

    for (auto & local : local_counts) {
        for (auto & item : local) {
            counts[item.first] += item.second;
        }
    }
    std::cerr << "BPE pretokenized " << (double(bytes_read) / (1024.0 * 1024.0 * 1024.0))
              << " GiB; unique pretokens=" << counts.size() << "\n";
}

static void append_stage1_encoded_raw_segment(
    const uint8_t * data,
    size_t n,
    const std::unordered_map<uint64_t, uint32_t> & merge_to_id,
    const std::unordered_map<uint64_t, uint32_t> & merge_rank,
    BpeWord & word) {
    word.ids.reserve(word.ids.size() + n);
    for (size_t i = 0; i < n; ++i) {
        word.ids.push_back(uint32_t(data[i]));
    }
    apply_ranked_merges(word.ids, merge_to_id, merge_rank);
}

static std::vector<BpeWord> collect_superbpe_stage2_words(
    const std::string & input_path,
    const std::unordered_map<uint64_t, uint32_t> & merge_to_id,
    const std::unordered_map<uint64_t, uint32_t> & merge_rank,
    uint64_t & bytes_read,
    uint64_t & special_count,
    int tokenizer_threads,
    uint64_t tokenizer_train_bytes) {
    TextMMap input(input_path);
    std::vector<ByteRange> ranges = split_document_ranges(input.data, input.bytes, special_count);
    bytes_read = restrict_ranges_to_byte_budget(ranges, tokenizer_train_bytes, special_count);

    const int n_threads = std::min<int>(normalize_thread_count(tokenizer_threads), std::max<size_t>(1, ranges.size()));
    std::vector<std::vector<BpeWord>> local_words{size_t(n_threads)};
    std::vector<std::thread> workers;
    workers.reserve(size_t(n_threads));
    std::atomic<uint64_t> processed{0};
    std::atomic<uint64_t> next_report{256ull * 1024ull * 1024ull};
    std::cerr << "SuperBPE stage2 preparing " << ranges.size() << " documents with " << n_threads << " threads";
    if (tokenizer_train_bytes > 0) {
        std::cerr << " byte_budget=" << tokenizer_train_bytes;
    }
    std::cerr << "\n";

    for (int t = 0; t < n_threads; ++t) {
        const size_t lo = ranges.size() * size_t(t) / size_t(n_threads);
        const size_t hi = ranges.size() * size_t(t + 1) / size_t(n_threads);
        workers.emplace_back([&, t, lo, hi] {
            auto & local = local_words[size_t(t)];
            local.reserve(hi - lo);
            uint64_t local_bytes = 0;
            for (size_t i = lo; i < hi; ++i) {
                const ByteRange r = ranges[i];
                BpeWord word;
                word.count = 1;
                append_stage1_encoded_raw_segment(
                    reinterpret_cast<const uint8_t *>(input.data + r.begin),
                    r.end - r.begin,
                    merge_to_id,
                    merge_rank,
                    word);
                if (!word.ids.empty()) {
                    local.push_back(std::move(word));
                }
                local_bytes += r.end - r.begin;
                if (local_bytes >= 8ull * 1024ull * 1024ull) {
                    report_parallel_progress("SuperBPE stage2 prepared", processed, next_report, local_bytes, i + 1);
                    local_bytes = 0;
                }
            }
            if (local_bytes > 0) {
                report_parallel_progress("SuperBPE stage2 prepared", processed, next_report, local_bytes, hi);
            }
        });
    }
    for (std::thread & worker : workers) {
        worker.join();
    }

    std::vector<BpeWord> words;
    size_t total = 0;
    for (const auto & local : local_words) {
        total += local.size();
    }
    words.reserve(total);
    for (auto & local : local_words) {
        std::move(local.begin(), local.end(), std::back_inserter(words));
    }
    std::cerr << "SuperBPE stage2 prepared " << (double(bytes_read) / (1024.0 * 1024.0 * 1024.0))
              << " GiB; documents=" << words.size() << "\n";
    return words;
}

static std::vector<BpeWord> counts_to_bpe_words(
    const std::unordered_map<std::string, uint64_t> & counts,
    const std::unordered_map<uint64_t, uint32_t> & merge_to_id = {},
    const std::unordered_map<uint64_t, uint32_t> & merge_rank = {}) {
    std::vector<BpeWord> words;
    words.reserve(counts.size());
    for (const auto & item : counts) {
        BpeWord word;
        word.count = item.second;
        word.ids.reserve(item.first.size());
        for (uint8_t b : item.first) {
            word.ids.push_back(uint32_t(b));
        }
        if (!merge_rank.empty()) {
            apply_ranked_merges(word.ids, merge_to_id, merge_rank);
        }
        if (!word.ids.empty()) {
            words.push_back(std::move(word));
        }
    }
    return words;
}

struct FlatPairTable {
    std::vector<uint64_t> keys;
    std::vector<uint64_t> counts;
    std::vector<int32_t> heads;
    std::vector<uint8_t> used;
    size_t filled = 0;
    size_t mask = 0;

    static uint64_t mix(uint64_t x) {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    }

    void init(size_t expected) {
        size_t cap = 1024;
        const size_t want = std::max<size_t>(expected * 2, 1024);
        while (cap < want) {
            cap <<= 1;
        }
        keys.assign(cap, 0);
        counts.assign(cap, 0);
        heads.assign(cap, -1);
        used.assign(cap, 0);
        filled = 0;
        mask = cap - 1;
    }

    size_t capacity() const {
        return keys.size();
    }

    size_t find(uint64_t key) const {
        if (keys.empty()) {
            return npos();
        }
        size_t i = size_t(mix(key)) & mask;
        while (used[i]) {
            if (keys[i] == key) {
                return i;
            }
            i = (i + 1) & mask;
        }
        return npos();
    }

    static size_t npos() {
        return std::numeric_limits<size_t>::max();
    }

    void rehash(size_t new_cap) {
        size_t cap = 1024;
        while (cap < new_cap) {
            cap <<= 1;
        }
        FlatPairTable next;
        next.init(cap / 2);
        for (size_t i = 0; i < keys.size(); ++i) {
            if (!used[i]) {
                continue;
            }
            const size_t j = next.get_or_insert_no_rehash(keys[i]);
            next.counts[j] = counts[i];
            next.heads[j] = heads[i];
        }
        *this = std::move(next);
    }

    size_t get_or_insert_no_rehash(uint64_t key) {
        size_t i = size_t(mix(key)) & mask;
        while (used[i]) {
            if (keys[i] == key) {
                return i;
            }
            i = (i + 1) & mask;
        }
        used[i] = 1;
        keys[i] = key;
        counts[i] = 0;
        heads[i] = -1;
        ++filled;
        return i;
    }

    size_t get_or_insert(uint64_t key) {
        if (keys.empty()) {
            init(1024);
        }
        if ((filled + 1) * 10 >= keys.size() * 7) {
            rehash(keys.size() * 2);
        }
        return get_or_insert_no_rehash(key);
    }

    void add_count(uint64_t key, uint64_t delta) {
        const size_t i = get_or_insert(key);
        counts[i] += delta;
    }
};

static void train_bpe_words(
    std::vector<BpeWord> & words,
    std::vector<std::vector<uint8_t>> & vocab,
    std::vector<MergeRecord> & merges,
    int target_vocab_size,
    const std::string & label,
    int tokenizer_threads) {
    size_t total_tokens = 0;
    for (const BpeWord & word : words) {
        total_tokens += word.ids.size();
    }
    if (total_tokens > size_t(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error(label + " has too many token positions for the in-memory BPE trainer");
    }

    std::vector<uint32_t> token;
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    std::vector<int32_t> bucket_prev;
    std::vector<int32_t> bucket_next;
    std::vector<uint32_t> weight;
    token.reserve(total_tokens);
    prev.reserve(total_tokens);
    next.reserve(total_tokens);
    bucket_prev.reserve(total_tokens);
    bucket_next.reserve(total_tokens);
    weight.reserve(total_tokens);

    for (const BpeWord & word : words) {
        int32_t last = -1;
        const uint32_t w = uint32_t(std::min<uint64_t>(word.count, uint64_t(std::numeric_limits<uint32_t>::max())));
        for (uint32_t id : word.ids) {
            const int32_t pos = int32_t(token.size());
            token.push_back(id);
            prev.push_back(last);
            next.push_back(-1);
            bucket_prev.push_back(-1);
            bucket_next.push_back(-1);
            weight.push_back(w);
            if (last >= 0) {
                next[size_t(last)] = pos;
            }
            last = pos;
        }
    }
    words.clear();
    words.shrink_to_fit();
    std::cerr << label << " flattened_positions=" << token.size() << "\n";

    const int n_threads = std::min<int>(normalize_thread_count(tokenizer_threads), std::max<size_t>(1, token.size()));
    const size_t reserve_pairs = std::max<size_t>(1024, total_tokens / 8);

    auto is_active = [&](int32_t pos) {
        return pos >= 0 && next[size_t(pos)] != -2;
    };

    auto has_pair = [&](int32_t pos) {
        return is_active(pos) && next[size_t(pos)] >= 0 && is_active(next[size_t(pos)]);
    };

    std::vector<FlatPairTable> local_tables;
    local_tables.resize(size_t(n_threads));
    std::vector<std::thread> workers;
    workers.reserve(size_t(n_threads));
    for (int t = 0; t < n_threads; ++t) {
        const size_t lo = token.size() * size_t(t) / size_t(n_threads);
        const size_t hi = token.size() * size_t(t + 1) / size_t(n_threads);
        workers.emplace_back([&, t, lo, hi] {
            FlatPairTable & local = local_tables[size_t(t)];
            local.init(std::max<size_t>(1024, (hi - lo) / 8));
            for (size_t p = lo; p < hi; ++p) {
                const int32_t pos = int32_t(p);
                if (!has_pair(pos)) {
                    continue;
                }
                const uint64_t pair = pack_pair(token[p], token[size_t(next[p])]);
                local.add_count(pair, weight[p]);
            }
        });
    }
    for (std::thread & worker : workers) {
        worker.join();
    }
    workers.clear();

    FlatPairTable table;
    table.init(reserve_pairs);
    for (FlatPairTable & local : local_tables) {
        for (size_t i = 0; i < local.keys.size(); ++i) {
            if (local.used[i] && local.counts[i] > 0) {
                table.add_count(local.keys[i], local.counts[i]);
            }
        }
        FlatPairTable empty;
        local = std::move(empty);
    }
    local_tables.clear();
    local_tables.shrink_to_fit();
    std::cerr << label << " indexed_unique_pairs=" << table.filled
              << " table_capacity=" << table.capacity() << "\n";

    std::vector<std::atomic<int32_t>> atomic_heads(table.capacity());
    for (auto & h : atomic_heads) {
        h.store(-1, std::memory_order_relaxed);
    }

    workers.reserve(size_t(n_threads));
    for (int t = 0; t < n_threads; ++t) {
        const size_t lo = token.size() * size_t(t) / size_t(n_threads);
        const size_t hi = token.size() * size_t(t + 1) / size_t(n_threads);
        workers.emplace_back([&, lo, hi] {
            for (size_t p = lo; p < hi; ++p) {
                const int32_t pos = int32_t(p);
                if (!has_pair(pos)) {
                    continue;
                }
                const uint64_t pair = pack_pair(token[p], token[size_t(next[p])]);
                const size_t idx = table.find(pair);
                if (idx == FlatPairTable::npos()) {
                    continue;
                }
                bucket_prev[p] = -1;
                const int32_t old_head = atomic_heads[idx].exchange(pos, std::memory_order_relaxed);
                bucket_next[p] = old_head;
                if (old_head >= 0) {
                    bucket_prev[size_t(old_head)] = pos;
                }
            }
        });
    }
    for (std::thread & worker : workers) {
        worker.join();
    }
    workers.clear();
    for (size_t i = 0; i < table.capacity(); ++i) {
        table.heads[i] = atomic_heads[i].load(std::memory_order_relaxed);
    }
    atomic_heads.clear();
    std::vector<std::atomic<int32_t>> empty_heads;
    atomic_heads.swap(empty_heads);
    std::cerr << label << " occurrence_buckets_ready\n";

    // Addressable (indexed) binary max-heap over table slots. Each live pair
    // (count > 0) occupies exactly ONE heap entry, identified by its table slot
    // index; heap_pos maps a slot back to its position in the heap. Count changes
    // update the entry in place (sift up/down) instead of pushing a duplicate, so
    // the heap never accumulates the stale entries that made a lazy-deletion
    // priority_queue grow without bound in the superword stage. All operations
    // are O(log live_pairs) on the genuine live-pair count.
    const size_t NPOS = FlatPairTable::npos();
    std::vector<size_t> heap;          // heap[i] = table slot index
    std::vector<size_t> heap_pos;      // heap_pos[slot] = index into heap, or NPOS

    // The pair currently being merged is popped and "frozen": its count is driven
    // to zero by the unregister calls during the merge, but we keep it out of the
    // heap meanwhile so those decrements don't repeatedly re-insert and sift it.
    bool has_frozen = false;
    uint64_t frozen_key = 0;

    // Strict weak ordering matching the old PairHeapLess: lower priority means
    // smaller count, ties broken so the lexicographically greater pair wins (for
    // deterministic, reference-compatible merge selection).
    auto prio_less = [&](size_t a, size_t b) -> bool {
        const uint64_t ca = table.counts[a];
        const uint64_t cb = table.counts[b];
        if (ca != cb) {
            return ca < cb;
        }
        const uint64_t pa = table.keys[a];
        const uint64_t pb = table.keys[b];
        if (pa == pb) {
            return false;
        }
        return pair_bytes_greater(pb, pa, vocab);
    };

    auto heap_sift_up = [&](size_t i) {
        while (i > 0) {
            const size_t parent = (i - 1) / 2;
            if (!prio_less(heap[parent], heap[i])) {
                break;
            }
            std::swap(heap[i], heap[parent]);
            heap_pos[heap[i]] = i;
            heap_pos[heap[parent]] = parent;
            i = parent;
        }
    };

    auto heap_sift_down = [&](size_t i) {
        const size_t n = heap.size();
        for (;;) {
            const size_t l = 2 * i + 1;
            const size_t r = 2 * i + 2;
            size_t m = i;
            if (l < n && prio_less(heap[m], heap[l])) {
                m = l;
            }
            if (r < n && prio_less(heap[m], heap[r])) {
                m = r;
            }
            if (m == i) {
                break;
            }
            std::swap(heap[i], heap[m]);
            heap_pos[heap[i]] = i;
            heap_pos[heap[m]] = m;
            i = m;
        }
    };

    auto heap_insert = [&](size_t idx) {
        const size_t i = heap.size();
        heap.push_back(idx);
        heap_pos[idx] = i;
        heap_sift_up(i);
    };

    auto heap_remove_at = [&](size_t p) {
        heap_pos[heap[p]] = NPOS;
        const size_t last = heap.back();
        heap.pop_back();
        if (p < heap.size()) {
            heap[p] = last;
            heap_pos[last] = p;
            heap_sift_down(p);
            heap_sift_up(p);
        }
    };

    // Reconcile a slot's heap membership with its current count. Skips the frozen
    // (currently-merging) pair so its in-progress decrements cost nothing.
    auto heap_update = [&](size_t idx) {
        if (idx == NPOS) {
            return;
        }
        if (has_frozen && table.keys[idx] == frozen_key) {
            return;
        }
        const size_t p = heap_pos[idx];
        if (table.counts[idx] > 0) {
            if (p == NPOS) {
                heap_insert(idx);
            } else {
                heap_sift_down(p);
                heap_sift_up(p);
            }
        } else if (p != NPOS) {
            heap_remove_at(p);
        }
    };

    // Build (or rebuild, after a table rehash remaps slot indices) the heap from
    // the live table entries in O(live_pairs) via bottom-up heapify.
    auto heap_rebuild_from_table = [&]() {
        heap.clear();
        heap_pos.assign(table.capacity(), NPOS);
        for (size_t i = 0; i < table.keys.size(); ++i) {
            if (table.used[i] && table.counts[i] > 0) {
                if (has_frozen && table.keys[i] == frozen_key) {
                    continue;
                }
                heap_pos[i] = heap.size();
                heap.push_back(i);
            }
        }
        for (size_t i = heap.size() / 2; i-- > 0;) {
            heap_sift_down(i);
        }
    };

    heap_rebuild_from_table();
    std::cerr << label << " heap_entries=" << heap.size() << "\n";

    auto link_pair_occurrence = [&](size_t idx, int32_t pos) {
        const int32_t old_head = table.heads[idx];
        bucket_prev[size_t(pos)] = -1;
        bucket_next[size_t(pos)] = old_head;
        if (old_head >= 0) {
            bucket_prev[size_t(old_head)] = pos;
        }
        table.heads[idx] = pos;
    };

    auto find_or_insert_pair = [&](uint64_t pair) {
        size_t idx = table.find(pair);
        if (idx != FlatPairTable::npos()) {
            return idx;
        }
        if ((table.filled + 1) * 10 >= table.keys.size() * 7) {
            // Rehash remaps every slot index, invalidating heap/heap_pos, so
            // rebuild both from the (preserved) counts afterwards.
            table.rehash(table.keys.size() * 2);
            heap_rebuild_from_table();
        }
        return table.get_or_insert_no_rehash(pair);
    };

    auto register_pair = [&](int32_t pos) {
        if (!has_pair(pos)) {
            return;
        }
        const uint64_t pair = pack_pair(token[size_t(pos)], token[size_t(next[size_t(pos)])]);
        const size_t idx = find_or_insert_pair(pair);
        table.counts[idx] += weight[size_t(pos)];
        heap_update(idx);
        link_pair_occurrence(idx, pos);
    };

    auto unregister_pair = [&](int32_t pos) {
        if (!has_pair(pos)) {
            return;
        }
        const uint64_t pair = pack_pair(token[size_t(pos)], token[size_t(next[size_t(pos)])]);
        const size_t idx = table.find(pair);
        if (idx == FlatPairTable::npos()) {
            return;
        }
        const uint64_t old_count = table.counts[idx];
        const uint64_t delta = weight[size_t(pos)];
        const uint64_t new_count = old_count > delta ? old_count - delta : 0;
        table.counts[idx] = new_count;
        heap_update(idx);

        const int32_t bp = bucket_prev[size_t(pos)];
        const int32_t bn = bucket_next[size_t(pos)];
        if (bp >= 0) {
            bucket_next[size_t(bp)] = bn;
        } else {
            if (table.heads[idx] == pos) {
                table.heads[idx] = bn;
            }
        }
        if (bn >= 0) {
            bucket_prev[size_t(bn)] = bp;
        }
        bucket_prev[size_t(pos)] = -1;
        bucket_next[size_t(pos)] = -1;
    };

    std::vector<int32_t> bucket_positions;

    while ((int)vocab.size() < target_vocab_size) {
        if (heap.empty()) {
            std::cerr << label << " stopped early: no mergeable pairs\n";
            break;
        }
        // heap[0] is the live maximum — counts are always current, so there are
        // no stale entries to skip. Pop it and freeze it for the merge below.
        size_t best_idx = heap[0];
        const uint64_t best_pair = table.keys[best_idx];
        const uint64_t best_count = table.counts[best_idx];
        heap_remove_at(0);
        has_frozen = true;
        frozen_key = best_pair;

        const uint32_t left = uint32_t(best_pair >> 32);
        const uint32_t right = uint32_t(best_pair);
        const uint32_t new_id = uint32_t(vocab.size());
        std::vector<uint8_t> merged;
        merged.reserve(vocab[left].size() + vocab[right].size());
        merged.insert(merged.end(), vocab[left].begin(), vocab[left].end());
        merged.insert(merged.end(), vocab[right].begin(), vocab[right].end());
        vocab.push_back(std::move(merged));
        merges.push_back({new_id, left, right});

        bucket_positions.clear();
        for (int32_t p = table.heads[best_idx]; p >= 0; p = bucket_next[size_t(p)]) {
            bucket_positions.push_back(p);
        }
        uint64_t merged_occurrences = 0;
        for (int32_t pos : bucket_positions) {
            if (has_pair(pos) &&
                token[size_t(pos)] == left &&
                token[size_t(next[size_t(pos)])] == right) {
                const int32_t right_pos = next[size_t(pos)];
                const int32_t before = prev[size_t(pos)];
                const int32_t after = next[size_t(right_pos)];

                if (before >= 0) {
                    unregister_pair(before);
                }
                unregister_pair(pos);
                if (after >= 0) {
                    unregister_pair(right_pos);
                }

                token[size_t(pos)] = new_id;
                next[size_t(pos)] = after;
                if (after >= 0) {
                    prev[size_t(after)] = pos;
                }
                prev[size_t(right_pos)] = -1;
                next[size_t(right_pos)] = -2;

                if (before >= 0) {
                    register_pair(before);
                }
                register_pair(pos);
                merged_occurrences += weight[size_t(pos)];
            }
        }

        if (merged_occurrences == 0) {
            table.counts[best_idx] = 0;
            table.heads[best_idx] = -1;
            has_frozen = false;
            vocab.pop_back();
            merges.pop_back();
            continue;
        }

        // Unfreeze and reconcile the merged pair. Its count is normally driven to
        // zero by the unregisters above (so it stays out of the heap); the final
        // heap_update is a safety net that re-inserts it if any count remains.
        has_frozen = false;
        const size_t merged_idx = table.find(best_pair);
        if (merged_idx != NPOS) {
            heap_update(merged_idx);
        }

        const int log_every = label.find("superword") != std::string::npos ? 25 : 250;
        if ((vocab.size() % size_t(log_every)) == 0 || (int)vocab.size() == target_vocab_size) {
            std::cerr << label << " vocab=" << vocab.size()
                      << " last_count=" << best_count
                      << " merged_weight=" << merged_occurrences
                      << " heap_entries=" << heap.size()
                      << "\n";
        }
    }
}

static void write_merges_tsv(const std::string & path, const std::vector<MergeRecord> & merges) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write merges: " + path);
    }
    out << "# special_token_id\t256\n";
    out << "# new_token_id\tleft_token_id\tright_token_id\n";
    for (const MergeRecord & m : merges) {
        out << m.new_id << '\t' << m.left << '\t' << m.right << '\n';
    }
}

static void write_bpe_metadata_json(
    const std::string & path,
    const std::string & input_path,
    TokenizerKind kind,
    uint64_t bytes_read,
    uint64_t special_count,
    uint64_t tokenizer_train_bytes,
    int max_vocab_size,
    int actual_vocab_size,
    int normal_stage_vocab_size,
    double superbpe_stage2_fraction,
    double elapsed_seconds) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write metadata: " + path);
    }
    out << "{\n";
    out << "  \"input_path\": \"" << json_escape_string(input_path) << "\",\n";
    out << "  \"bytes_read\": " << bytes_read << ",\n";
    if (tokenizer_train_bytes > 0) {
        out << "  \"tokenizer_train_bytes_limit\": " << tokenizer_train_bytes << ",\n";
    }
    out << "  \"max_vocab_size\": " << max_vocab_size << ",\n";
    out << "  \"actual_vocab_size\": " << actual_vocab_size << ",\n";
    out << "  \"base_byte_tokens\": 256,\n";
    out << "  \"special_token\": \"" << SPECIAL << "\",\n";
    out << "  \"special_token_id\": 256,\n";
    out << "  \"special_token_occurrences_in_input\": " << special_count << ",\n";
    out << "  \"merge_count\": " << std::max(0, actual_vocab_size - 257) << ",\n";
    out << "  \"tokenizer\": \"" << tokenizer_kind_name(kind) << "\",\n";
    out << "  \"pretokenization\": \"GPT-2 regex: '(?:[sdmt]|ll|ve|re)| ?\\\\p{L}+| ?\\\\p{N}+| ?[^\\\\s\\\\p{L}\\\\p{N}]+|\\\\s+(?!\\\\S)|\\\\s+\",\n";
    if (kind == TokenizerKind::SuperBpe) {
        out << "  \"superbpe_normal_stage_vocab_size\": " << normal_stage_vocab_size << ",\n";
        out << "  \"superbpe_stage2_fraction\": " << superbpe_stage2_fraction << ",\n";
        out << "  \"superbpe_stage2_training\": \"resume BPE from the stage-1 vocabulary with pretokenization disabled inside each <|endoftext|>-delimited document\",\n";
    }
    out << "  \"elapsed_seconds\": " << elapsed_seconds << "\n";
    out << "}\n";
}

static void build_bpe_vocab(
    const std::string & input_path,
    const std::string & output_dir,
    TokenizerKind kind,
    int max_vocab_size,
    double superbpe_transition_frac,
    int superbpe_transition_vocab,
    int tokenizer_threads,
    uint64_t tokenizer_train_bytes) {
    const auto started = std::chrono::steady_clock::now();
    if (max_vocab_size <= 257) {
        throw std::runtime_error("BPE vocab size must be greater than 257");
    }
    if (kind != TokenizerKind::RegexBpe && kind != TokenizerKind::SuperBpe) {
        throw std::runtime_error("build_bpe_vocab called for non-BPE tokenizer");
    }

    std::vector<std::vector<uint8_t>> vocab;
    vocab.reserve(size_t(max_vocab_size));
    for (int i = 0; i < 256; ++i) {
        vocab.push_back({uint8_t(i)});
    }
    vocab.emplace_back(SPECIAL, SPECIAL + std::strlen(SPECIAL));
    std::vector<MergeRecord> merges;
    merges.reserve(size_t(max_vocab_size - 257));

    uint64_t bytes_read = 0;
    uint64_t special_count = 0;
    std::unordered_map<std::string, uint64_t> counts;
    collect_regex_pretoken_counts(input_path, counts, bytes_read, special_count, tokenizer_threads, tokenizer_train_bytes);

    int normal_stage_vocab_size = max_vocab_size;
    if (kind == TokenizerKind::SuperBpe) {
        if (superbpe_transition_vocab > 0) {
            normal_stage_vocab_size = superbpe_transition_vocab;
        } else {
            const double clamped_transition = std::min(0.999, std::max(0.0, superbpe_transition_frac));
            normal_stage_vocab_size = int(std::round(double(max_vocab_size) * clamped_transition));
        }
        normal_stage_vocab_size = std::max(258, std::min(max_vocab_size, normal_stage_vocab_size));
    }

    std::vector<BpeWord> words = counts_to_bpe_words(counts);
    std::cerr << "BPE training units=" << words.size()
              << " target_vocab=" << normal_stage_vocab_size << "\n";
    train_bpe_words(words, vocab, merges, normal_stage_vocab_size,
                    kind == TokenizerKind::SuperBpe ? "SuperBPE normal stage" : "BPE",
                    tokenizer_threads);

    if (kind == TokenizerKind::SuperBpe && (int)vocab.size() < max_vocab_size) {
        std::unordered_map<uint64_t, uint32_t> merge_to_id;
        std::unordered_map<uint64_t, uint32_t> merge_rank;
        build_known_merge_maps(merges, merge_to_id, merge_rank);
        counts.clear();
        uint64_t stage2_bytes_read = 0;
        uint64_t stage2_special_count = 0;
        words = collect_superbpe_stage2_words(input_path, merge_to_id, merge_rank,
                                              stage2_bytes_read, stage2_special_count,
                                              tokenizer_threads, tokenizer_train_bytes);
        bytes_read = stage2_bytes_read;
        special_count = stage2_special_count;
        std::cerr << "SuperBPE stage2 documents=" << words.size()
                  << " final_target_vocab=" << max_vocab_size << "\n";
        train_bpe_words(words, vocab, merges, max_vocab_size, "SuperBPE superword stage",
                        tokenizer_threads);
    }

    ensure_directory(output_dir);
    write_tokenizer_vocab_json(output_dir + "/vocab.json", vocab);
    write_merges_tsv(output_dir + "/merges.tsv", merges);
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    write_bpe_metadata_json(output_dir + "/metadata.json", input_path, kind, bytes_read, special_count,
                            tokenizer_train_bytes,
                            max_vocab_size, int(vocab.size()), normal_stage_vocab_size,
                            1.0 - double(normal_stage_vocab_size) / double(max_vocab_size), elapsed);
    std::cerr << "done: wrote " << tokenizer_kind_name(kind)
              << " vocab=" << vocab.size() << " merges=" << merges.size()
              << " to " << output_dir << " in " << elapsed << "s\n";
}

static void write_blt_metadata_json(
    const std::string & path,
    const std::string & input_path,
    uint64_t bytes_read,
    uint64_t special_count,
    int max_vocab_size,
    int max_patch_bytes,
    int actual_vocab_size,
    double elapsed_seconds) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write metadata: " + path);
    }
    out << "{\n";
    out << "  \"input_path\": \"" << json_escape_string(input_path) << "\",\n";
    out << "  \"bytes_read\": " << bytes_read << ",\n";
    out << "  \"max_vocab_size\": " << max_vocab_size << ",\n";
    out << "  \"actual_vocab_size\": " << actual_vocab_size << ",\n";
    out << "  \"base_byte_tokens\": 256,\n";
    out << "  \"special_token\": \"" << SPECIAL << "\",\n";
    out << "  \"special_token_id\": 256,\n";
    out << "  \"special_token_occurrences_in_input\": " << special_count << ",\n";
    out << "  \"tokenizer\": \"blt\",\n";
    out << "  \"max_patch_bytes\": " << max_patch_bytes << ",\n";
    out << "  \"patching\": \"dynamic byte patching with byte fallback; frequent train patches are assigned ids above 256\",\n";
    out << "  \"elapsed_seconds\": " << elapsed_seconds << "\n";
    out << "}\n";
}

static void build_blt_vocab(
    const std::string & input_path,
    const std::string & output_dir,
    int max_vocab_size,
    int max_patch_bytes) {
    const auto started = std::chrono::steady_clock::now();
    if (max_vocab_size <= 257) {
        throw std::runtime_error("BLT vocab size must be greater than 257");
    }
    std::ifstream in(input_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open BLT vocab input: " + input_path);
    }
    std::unordered_map<std::string, uint64_t> counts;
    std::string line;
    uint64_t bytes_read = 0;
    uint64_t special_count = 0;
    uint64_t last_report = 0;
    std::vector<std::vector<uint8_t>> patches;
    while (std::getline(in, line)) {
        line.push_back('\n');
        bytes_read += line.size();
        size_t start = 0;
        while (start < line.size()) {
            const size_t pos = line.find(SPECIAL, start);
            const size_t end = pos == std::string::npos ? line.size() : pos;
            patches.clear();
            dynamic_byte_patches(reinterpret_cast<const uint8_t *>(line.data() + start), end - start, max_patch_bytes, patches);
            for (const auto & patch : patches) {
                if (patch.size() > 1) {
                    std::string key(reinterpret_cast<const char *>(patch.data()), patch.size());
                    ++counts[key];
                }
            }
            if (pos == std::string::npos) {
                break;
            }
            ++special_count;
            start = pos + std::strlen(SPECIAL);
        }
        if (bytes_read - last_report >= 256ull * 1024ull * 1024ull) {
            last_report = bytes_read;
            std::cerr << "BLT vocab read " << (double(bytes_read) / (1024.0 * 1024.0 * 1024.0))
                      << " GiB; patch types=" << counts.size() << "\n";
        }
    }

    std::vector<std::pair<std::string, uint64_t>> ranked(counts.begin(), counts.end());
    std::sort(ranked.begin(), ranked.end(), [](const auto & a, const auto & b) {
        if (a.second != b.second) {
            return a.second > b.second;
        }
        return a.first > b.first;
    });

    std::vector<std::vector<uint8_t>> vocab;
    vocab.reserve(size_t(max_vocab_size));
    for (int i = 0; i < 256; ++i) {
        vocab.push_back({uint8_t(i)});
    }
    vocab.emplace_back(SPECIAL, SPECIAL + std::strlen(SPECIAL));
    for (const auto & item : ranked) {
        if ((int)vocab.size() >= max_vocab_size) {
            break;
        }
        vocab.emplace_back(item.first.begin(), item.first.end());
    }

    ensure_directory(output_dir);
    write_tokenizer_vocab_json(output_dir + "/vocab.json", vocab);
    std::ofstream merges(output_dir + "/merges.tsv");
    merges << "# BLT patch tokenizer has no BPE merges\n";
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    write_blt_metadata_json(output_dir + "/metadata.json", input_path, bytes_read, special_count, max_vocab_size, max_patch_bytes, int(vocab.size()), elapsed);
    std::cerr << "done: wrote BLT vocab=" << vocab.size() << " to " << output_dir
              << " in " << elapsed << "s\n";
}

struct TokenFile {
    int fd = -1;
    size_t bytes = 0;
    size_t n_tokens = 0;
    const uint16_t * data = nullptr;

    TokenFile() = default;
    TokenFile(const TokenFile &) = delete;
    TokenFile & operator=(const TokenFile &) = delete;

    explicit TokenFile(const std::string & path) {
        open(path);
    }

    ~TokenFile() {
        close();
    }

    void open(const std::string & path) {
        close();
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("failed to open token file: " + path);
        }
        struct stat st {};
        if (::fstat(fd, &st) != 0) {
            throw std::runtime_error("failed to stat token file: " + path);
        }
        if (st.st_size <= 0 || (st.st_size % 2) != 0) {
            throw std::runtime_error("token file has invalid byte count: " + path);
        }
        bytes = size_t(st.st_size);
        n_tokens = bytes / sizeof(uint16_t);
        void * p = ::mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) {
            throw std::runtime_error("failed to mmap token file: " + path);
        }
        data = reinterpret_cast<const uint16_t *>(p);
    }

    void close() {
        if (data != nullptr) {
            ::munmap((void *)data, bytes);
            data = nullptr;
        }
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
        bytes = 0;
        n_tokens = 0;
    }

    uint16_t operator[](size_t i) const {
        return data[i];
    }
};

struct LlmConfig {
    int vocab_size = 10000;
    int context = 128;
    int batch = 32;
    int d_model = 256;
    int n_heads = 8;
    int n_layers = 4;
    int d_ff = 1024;
    float rope_theta = 10000.0f;
    float rms_eps = 1e-5f;
    float beta1 = 0.9f;
    float beta2 = 0.95f;
    float adam_eps = 1e-4f;
};

struct TrainConfig {
    uint64_t steps = 20000;
    uint64_t lr_steps = 0;
    uint64_t seed = 1337;
    int warmup_steps = 200;
    int log_every = 10;
    int valid_every = 250;
    int save_every = 1000;
    int valid_batches = 64;
    bool final_validate = true;
    bool final_save = true;
    float learning_rate = 3e-4f;
    float min_learning_rate = 3e-5f;
    float weight_decay = 0.1f;
    float grad_clip = 1.0f;
    float muon_learning_rate = 0.02f;
    float muon_min_learning_rate = 0.002f;
    float muon_weight_decay = 0.025f;
    float muon_momentum_start = 0.85f;
    float muon_momentum = 0.95f;
    int muon_momentum_warmup = 300;
    int muon_momentum_cooldown = 50;
};

static int64_t product(const std::vector<int64_t> & dims) {
    int64_t n = 1;
    for (int64_t d : dims) {
        n *= d;
    }
    return n;
}

static NSString * ns(const std::string & s) {
    return [NSString stringWithUTF8String:s.c_str()];
}

static MPSShape * mps_shape(const std::vector<int64_t> & dims) {
    NSMutableArray<NSNumber *> * a = [NSMutableArray arrayWithCapacity:dims.size()];
    for (int64_t d : dims) {
        [a addObject:@(d)];
    }
    return a;
}

static NSArray<NSNumber *> * nums(std::initializer_list<int64_t> values) {
    NSMutableArray<NSNumber *> * a = [NSMutableArray arrayWithCapacity:values.size()];
    for (int64_t v : values) {
        [a addObject:@(v)];
    }
    return a;
}

static std::vector<float> normal_vec(size_t n, float stddev, std::mt19937 & rng) {
    std::normal_distribution<float> dist(0.0f, stddev);
    std::vector<float> v(n);
    for (float & x : v) {
        x = dist(rng);
    }
    return v;
}

static std::vector<float> filled_vec(size_t n, float value) {
    return std::vector<float>(n, value);
}

struct AttentionKernelParams {
    uint32_t batch = 0;
    uint32_t heads = 0;
    uint32_t seq = 0;
    uint32_t dim = 0;
    float scale = 1.0f;
};

static const char * ATTENTION_METAL_SOURCE = R"metal(
#include <metal_stdlib>
using namespace metal;

struct AttentionKernelParams {
    uint batch;
    uint heads;
    uint seq;
    uint dim;
    float scale;
};

static inline uint attn_idx(uint b, uint h, uint t, uint d, constant AttentionKernelParams & p) {
    return (((b * p.heads + h) * p.seq + t) * p.dim + d);
}

kernel void zero_floats(
    device float * x [[buffer(0)]],
    constant uint & n [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (i < n) {
        x[i] = 0.0f;
    }
}

constant uint ATTENTION_TG_SIZE = 256;
constant uint ATTENTION_MAX_SEQ = 1024;

kernel void causal_attention_forward(
    device const float * q [[buffer(0)]],
    device const float * k [[buffer(1)]],
    device const float * v [[buffer(2)]],
    device float * y [[buffer(3)]],
    constant AttentionKernelParams & p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float scores[ATTENTION_MAX_SEQ];
    threadgroup float scratch[ATTENTION_TG_SIZE];
    const uint rows = p.batch * p.heads * p.seq;
    if (row >= rows || p.seq > ATTENTION_MAX_SEQ) {
        return;
    }
    const uint b = row / (p.heads * p.seq);
    const uint rem = row - b * p.heads * p.seq;
    const uint h = rem / p.seq;
    const uint i = rem - h * p.seq;

    float local_max = -INFINITY;
    for (uint j = tid; j <= i; j += ATTENTION_TG_SIZE) {
        float dot = 0.0f;
        for (uint d = 0; d < p.dim; ++d) {
            dot += q[attn_idx(b, h, i, d, p)] * k[attn_idx(b, h, j, d, p)];
        }
        const float scaled = dot * p.scale;
        scores[j] = scaled;
        local_max = max(local_max, scaled);
    }
    scratch[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ATTENTION_TG_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float row_max = scratch[0];

    float local_denom = 0.0f;
    for (uint j = tid; j <= i; j += ATTENTION_TG_SIZE) {
        local_denom += exp(scores[j] - row_max);
    }
    scratch[tid] = local_denom;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ATTENTION_TG_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denom = scratch[0];

    for (uint d = tid; d < p.dim; d += ATTENTION_TG_SIZE) {
        float acc = 0.0f;
        for (uint j = 0; j <= i; ++j) {
            const float prob = exp(scores[j] - row_max) / denom;
            acc += prob * v[attn_idx(b, h, j, d, p)];
        }
        y[attn_idx(b, h, i, d, p)] = acc;
    }
}

kernel void causal_attention_backward(
    device const float * q [[buffer(0)]],
    device const float * k [[buffer(1)]],
    device const float * v [[buffer(2)]],
    device const float * dy [[buffer(3)]],
    device const float * y [[buffer(4)]],
    device float * dq [[buffer(5)]],
    device atomic_float * dk [[buffer(6)]],
    device atomic_float * dv [[buffer(7)]],
    constant AttentionKernelParams & p [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float scores[ATTENTION_MAX_SEQ];
    threadgroup float probs[ATTENTION_MAX_SEQ];
    threadgroup float dps[ATTENTION_MAX_SEQ];
    threadgroup float scratch[ATTENTION_TG_SIZE];
    const uint rows = p.batch * p.heads * p.seq;
    if (row >= rows || p.seq > ATTENTION_MAX_SEQ) {
        return;
    }
    const uint b = row / (p.heads * p.seq);
    const uint rem = row - b * p.heads * p.seq;
    const uint h = rem / p.seq;
    const uint i = rem - h * p.seq;

    for (uint d = tid; d < p.dim; d += ATTENTION_TG_SIZE) {
        dq[attn_idx(b, h, i, d, p)] = 0.0f;
    }

    float local_max = -INFINITY;
    for (uint j = tid; j <= i; j += ATTENTION_TG_SIZE) {
        float dot = 0.0f;
        for (uint d = 0; d < p.dim; ++d) {
            dot += q[attn_idx(b, h, i, d, p)] * k[attn_idx(b, h, j, d, p)];
        }
        const float scaled = dot * p.scale;
        scores[j] = scaled;
        local_max = max(local_max, scaled);
    }
    scratch[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ATTENTION_TG_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float row_max = scratch[0];

    float local_denom = 0.0f;
    for (uint j = tid; j <= i; j += ATTENTION_TG_SIZE) {
        local_denom += exp(scores[j] - row_max);
    }
    scratch[tid] = local_denom;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ATTENTION_TG_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denom = scratch[0];

    float local_dpsum = 0.0f;
    for (uint j = tid; j <= i; j += ATTENTION_TG_SIZE) {
        const float prob = exp(scores[j] - row_max) / denom;
        float dp = 0.0f;
        for (uint d = 0; d < p.dim; ++d) {
            dp += dy[attn_idx(b, h, i, d, p)] * v[attn_idx(b, h, j, d, p)];
        }
        probs[j] = prob;
        dps[j] = dp;
        local_dpsum += prob * dp;
    }
    scratch[tid] = local_dpsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ATTENTION_TG_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float dpsum = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid; d < p.dim; d += ATTENTION_TG_SIZE) {
        float q_acc = 0.0f;
        for (uint j = 0; j <= i; ++j) {
            const float ds = probs[j] * (dps[j] - dpsum);
            q_acc += p.scale * ds * k[attn_idx(b, h, j, d, p)];
        }
        dq[attn_idx(b, h, i, d, p)] = q_acc;
    }

    for (uint j = tid; j <= i; j += ATTENTION_TG_SIZE) {
        const float prob = probs[j];
        const float ds = prob * (dps[j] - dpsum);
        for (uint d = 0; d < p.dim; ++d) {
            const uint qidx = attn_idx(b, h, i, d, p);
            const uint kvidx = attn_idx(b, h, j, d, p);
            atomic_fetch_add_explicit(&dk[kvidx], p.scale * ds * q[qidx], memory_order_relaxed);
            atomic_fetch_add_explicit(&dv[kvidx], prob * dy[qidx], memory_order_relaxed);
        }
    }
}
)metal";

static void cpu_attention_forward_backward(
    const std::vector<float> & q,
    const std::vector<float> & k,
    const std::vector<float> & v,
    const std::vector<float> & dy,
    const AttentionKernelParams & p,
    std::vector<float> & y,
    std::vector<float> & dq,
    std::vector<float> & dk,
    std::vector<float> & dv) {
    auto idx = [&](uint32_t b, uint32_t h, uint32_t t, uint32_t d) -> size_t {
        return size_t((((b * p.heads + h) * p.seq + t) * p.dim + d));
    };
    const size_t n = size_t(p.batch) * p.heads * p.seq * p.dim;
    y.assign(n, 0.0f);
    dq.assign(n, 0.0f);
    dk.assign(n, 0.0f);
    dv.assign(n, 0.0f);
    std::vector<float> prob(p.seq, 0.0f);
    std::vector<float> dp(p.seq, 0.0f);

    for (uint32_t b = 0; b < p.batch; ++b) {
        for (uint32_t h = 0; h < p.heads; ++h) {
            for (uint32_t i = 0; i < p.seq; ++i) {
                float row_max = -std::numeric_limits<float>::infinity();
                for (uint32_t j = 0; j <= i; ++j) {
                    float dot = 0.0f;
                    for (uint32_t d = 0; d < p.dim; ++d) {
                        dot += q[idx(b, h, i, d)] * k[idx(b, h, j, d)];
                    }
                    row_max = std::max(row_max, dot * p.scale);
                }
                float denom = 0.0f;
                for (uint32_t j = 0; j <= i; ++j) {
                    float dot = 0.0f;
                    for (uint32_t d = 0; d < p.dim; ++d) {
                        dot += q[idx(b, h, i, d)] * k[idx(b, h, j, d)];
                    }
                    prob[j] = std::exp(dot * p.scale - row_max);
                    denom += prob[j];
                }
                for (uint32_t j = 0; j <= i; ++j) {
                    prob[j] /= denom;
                }
                for (uint32_t d = 0; d < p.dim; ++d) {
                    float acc = 0.0f;
                    for (uint32_t j = 0; j <= i; ++j) {
                        acc += prob[j] * v[idx(b, h, j, d)];
                    }
                    y[idx(b, h, i, d)] = acc;
                }

                float dpsum = 0.0f;
                for (uint32_t j = 0; j <= i; ++j) {
                    dp[j] = 0.0f;
                    for (uint32_t d = 0; d < p.dim; ++d) {
                        dp[j] += dy[idx(b, h, i, d)] * v[idx(b, h, j, d)];
                    }
                    dpsum += prob[j] * dp[j];
                }
                for (uint32_t j = 0; j <= i; ++j) {
                    const float ds = prob[j] * (dp[j] - dpsum);
                    for (uint32_t d = 0; d < p.dim; ++d) {
                        dq[idx(b, h, i, d)] += p.scale * ds * k[idx(b, h, j, d)];
                        dk[idx(b, h, j, d)] += p.scale * ds * q[idx(b, h, i, d)];
                        dv[idx(b, h, j, d)] += prob[j] * dy[idx(b, h, i, d)];
                    }
                }
            }
        }
    }
}

struct MetalAttentionKernels {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> zero_pipeline = nil;
    id<MTLComputePipelineState> forward_pipeline = nil;
    id<MTLComputePipelineState> backward_pipeline = nil;

    explicit MetalAttentionKernels(id<MTLDevice> metal_device) {
        device = metal_device;
        queue = [device newCommandQueue];
        if (!queue) {
            throw std::runtime_error("failed to create attention command queue");
        }

        NSError * error = nil;
        MTLCompileOptions * options = [MTLCompileOptions new];
        id<MTLLibrary> library = [device newLibraryWithSource:ns(ATTENTION_METAL_SOURCE)
                                                      options:options
                                                        error:&error];
        if (!library) {
            std::string msg = "failed to compile attention Metal library";
            if (error) {
                msg += ": ";
                msg += [[error localizedDescription] UTF8String];
            }
            throw std::runtime_error(msg);
        }

        id<MTLFunction> zero_fn = [library newFunctionWithName:@"zero_floats"];
        id<MTLFunction> forward_fn = [library newFunctionWithName:@"causal_attention_forward"];
        id<MTLFunction> backward_fn = [library newFunctionWithName:@"causal_attention_backward"];
        if (!zero_fn || !forward_fn || !backward_fn) {
            throw std::runtime_error("failed to find attention Metal functions");
        }
        zero_pipeline = [device newComputePipelineStateWithFunction:zero_fn error:&error];
        if (!zero_pipeline) {
            std::string msg = "failed to build attention zero pipeline";
            if (error) {
                msg += ": ";
                msg += [[error localizedDescription] UTF8String];
            }
            throw std::runtime_error(msg);
        }
        forward_pipeline = [device newComputePipelineStateWithFunction:forward_fn error:&error];
        if (!forward_pipeline) {
            std::string msg = "failed to build attention forward pipeline";
            if (error) {
                msg += ": ";
                msg += [[error localizedDescription] UTF8String];
            }
            throw std::runtime_error(msg);
        }
        backward_pipeline = [device newComputePipelineStateWithFunction:backward_fn error:&error];
        if (!backward_pipeline) {
            std::string msg = "failed to build attention backward pipeline";
            if (error) {
                msg += ": ";
                msg += [[error localizedDescription] UTF8String];
            }
            throw std::runtime_error(msg);
        }
    }

    id<MTLBuffer> make_buffer(size_t bytes) {
        id<MTLBuffer> b = [device newBufferWithLength:std::max<size_t>(bytes, 4)
                                              options:MTLResourceStorageModeShared];
        if (!b) {
            throw std::runtime_error("failed to allocate attention buffer");
        }
        return b;
    }

    id<MTLBuffer> buffer_from(const void * data, size_t bytes) {
        id<MTLBuffer> b = [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
        if (!b) {
            throw std::runtime_error("failed to allocate attention buffer");
        }
        return b;
    }

    id<MTLBuffer> zero_buffer(size_t bytes) {
        id<MTLBuffer> b = [device newBufferWithLength:std::max<size_t>(bytes, 4) options:MTLResourceStorageModeShared];
        if (!b) {
            throw std::runtime_error("failed to allocate attention zero buffer");
        }
        std::memset([b contents], 0, bytes);
        return b;
    }

    void encode_threads(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline, uint32_t count) const {
        const NSUInteger w = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 128);
        MTLSize grid = MTLSizeMake(count, 1, 1);
        MTLSize tg = MTLSizeMake(w, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    }

    void encode_attention_rows(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline, uint32_t rows) const {
        constexpr NSUInteger kAttentionThreads = 256;
        if (pipeline.maxTotalThreadsPerThreadgroup < kAttentionThreads) {
            throw std::runtime_error("attention pipeline does not support 256-thread groups");
        }
        MTLSize groups = MTLSizeMake(rows, 1, 1);
        MTLSize tg = MTLSizeMake(kAttentionThreads, 1, 1);
        [enc dispatchThreadgroups:groups threadsPerThreadgroup:tg];
    }

    void encode_zero(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> b, id<MTLBuffer> count_b, uint32_t count) const {
        [enc setComputePipelineState:zero_pipeline];
        [enc setBuffer:b offset:0 atIndex:0];
        [enc setBuffer:count_b offset:0 atIndex:1];
        encode_threads(enc, zero_pipeline, count);
    }

    void encode_forward_backward(
        id<MTLComputeCommandEncoder> enc,
        id<MTLBuffer> qb,
        id<MTLBuffer> kb,
        id<MTLBuffer> vb,
        id<MTLBuffer> dyb,
        id<MTLBuffer> yb,
        id<MTLBuffer> dqb,
        id<MTLBuffer> dkb,
        id<MTLBuffer> dvb,
        id<MTLBuffer> pb,
        id<MTLBuffer> count_b,
        const AttentionKernelParams & p) const {
        const uint32_t rows = p.batch * p.heads * p.seq;
        const uint32_t count = rows * p.dim;
        if (p.seq > 1024) {
            throw std::runtime_error("custom attention kernel currently supports --context <= 1024");
        }
        encode_zero(enc, dkb, count_b, count);
        encode_zero(enc, dvb, count_b, count);

        [enc setComputePipelineState:forward_pipeline];
        [enc setBuffer:qb offset:0 atIndex:0];
        [enc setBuffer:kb offset:0 atIndex:1];
        [enc setBuffer:vb offset:0 atIndex:2];
        [enc setBuffer:yb offset:0 atIndex:3];
        [enc setBuffer:pb offset:0 atIndex:4];
        encode_attention_rows(enc, forward_pipeline, rows);

        [enc setComputePipelineState:backward_pipeline];
        [enc setBuffer:qb offset:0 atIndex:0];
        [enc setBuffer:kb offset:0 atIndex:1];
        [enc setBuffer:vb offset:0 atIndex:2];
        [enc setBuffer:dyb offset:0 atIndex:3];
        [enc setBuffer:yb offset:0 atIndex:4];
        [enc setBuffer:dqb offset:0 atIndex:5];
        [enc setBuffer:dkb offset:0 atIndex:6];
        [enc setBuffer:dvb offset:0 atIndex:7];
        [enc setBuffer:pb offset:0 atIndex:8];
        encode_attention_rows(enc, backward_pipeline, rows);
    }

    void run_preallocated(
        id<MTLBuffer> qb,
        id<MTLBuffer> kb,
        id<MTLBuffer> vb,
        id<MTLBuffer> dyb,
        id<MTLBuffer> yb,
        id<MTLBuffer> dqb,
        id<MTLBuffer> dkb,
        id<MTLBuffer> dvb,
        id<MTLBuffer> pb,
        id<MTLBuffer> count_b,
        const AttentionKernelParams & p) const {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        encode_forward_backward(enc, qb, kb, vb, dyb, yb, dqb, dkb, dvb, pb, count_b, p);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (NSError * error = cb.error) {
            throw std::runtime_error(std::string("attention command buffer failed: ") +
                                     [[error localizedDescription] UTF8String]);
        }
    }

    double benchmark(
        const std::vector<float> & q,
        const std::vector<float> & k,
        const std::vector<float> & v,
        const std::vector<float> & dy,
        const AttentionKernelParams & p,
        int warmup,
        int iters) {
        const size_t n = size_t(p.batch) * p.heads * p.seq * p.dim;
        const size_t bytes = n * sizeof(float);
        id<MTLBuffer> qb = buffer_from(q.data(), bytes);
        id<MTLBuffer> kb = buffer_from(k.data(), bytes);
        id<MTLBuffer> vb = buffer_from(v.data(), bytes);
        id<MTLBuffer> dyb = buffer_from(dy.data(), bytes);
        id<MTLBuffer> yb = make_buffer(bytes);
        id<MTLBuffer> dqb = make_buffer(bytes);
        id<MTLBuffer> dkb = make_buffer(bytes);
        id<MTLBuffer> dvb = make_buffer(bytes);
        id<MTLBuffer> pb = buffer_from(&p, sizeof(p));
        const uint32_t count = uint32_t(n);
        id<MTLBuffer> count_b = buffer_from(&count, sizeof(count));

        for (int i = 0; i < warmup; ++i) {
            run_preallocated(qb, kb, vb, dyb, yb, dqb, dkb, dvb, pb, count_b, p);
        }

        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            run_preallocated(qb, kb, vb, dyb, yb, dqb, dkb, dvb, pb, count_b, p);
        }
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(end - start).count();
    }

    void run(
        const std::vector<float> & q,
        const std::vector<float> & k,
        const std::vector<float> & v,
        const std::vector<float> & dy,
        const AttentionKernelParams & p,
        std::vector<float> & y,
        std::vector<float> & dq,
        std::vector<float> & dk,
        std::vector<float> & dv) {
        const size_t n = size_t(p.batch) * p.heads * p.seq * p.dim;
        const size_t bytes = n * sizeof(float);
        id<MTLBuffer> qb = buffer_from(q.data(), bytes);
        id<MTLBuffer> kb = buffer_from(k.data(), bytes);
        id<MTLBuffer> vb = buffer_from(v.data(), bytes);
        id<MTLBuffer> dyb = buffer_from(dy.data(), bytes);
        id<MTLBuffer> yb = zero_buffer(bytes);
        id<MTLBuffer> dqb = zero_buffer(bytes);
        id<MTLBuffer> dkb = zero_buffer(bytes);
        id<MTLBuffer> dvb = zero_buffer(bytes);
        id<MTLBuffer> pb = buffer_from(&p, sizeof(p));
        const uint32_t count = uint32_t(n);
        id<MTLBuffer> count_b = buffer_from(&count, sizeof(count));

        run_preallocated(qb, kb, vb, dyb, yb, dqb, dkb, dvb, pb, count_b, p);

        y.resize(n);
        dq.resize(n);
        dk.resize(n);
        dv.resize(n);
        std::memcpy(y.data(), [yb contents], bytes);
        std::memcpy(dq.data(), [dqb contents], bytes);
        std::memcpy(dk.data(), [dkb contents], bytes);
        std::memcpy(dv.data(), [dvb contents], bytes);
    }
};

static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("max_abs_diff size mismatch");
    }
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, double(std::abs(a[i] - b[i])));
    }
    return m;
}

struct CheckpointData {
    bool found = false;
    LlmConfig cfg;
    std::unordered_map<std::string, std::vector<float>> params;
};

static void write_u32(std::ofstream & out, uint32_t v) {
    out.write(reinterpret_cast<const char *>(&v), sizeof(v));
}

static void write_u64(std::ofstream & out, uint64_t v) {
    out.write(reinterpret_cast<const char *>(&v), sizeof(v));
}

static void write_i32(std::ofstream & out, int32_t v) {
    out.write(reinterpret_cast<const char *>(&v), sizeof(v));
}

static void write_f32(std::ofstream & out, float v) {
    out.write(reinterpret_cast<const char *>(&v), sizeof(v));
}

static uint32_t read_u32(std::ifstream & in) {
    uint32_t v = 0;
    in.read(reinterpret_cast<char *>(&v), sizeof(v));
    return v;
}

static uint64_t read_u64(std::ifstream & in) {
    uint64_t v = 0;
    in.read(reinterpret_cast<char *>(&v), sizeof(v));
    return v;
}

static int32_t read_i32(std::ifstream & in) {
    int32_t v = 0;
    in.read(reinterpret_cast<char *>(&v), sizeof(v));
    return v;
}

static float read_f32(std::ifstream & in) {
    float v = 0.0f;
    in.read(reinterpret_cast<char *>(&v), sizeof(v));
    return v;
}

static CheckpointData load_checkpoint(const std::string & path) {
    CheckpointData ckpt;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return ckpt;
    }
    char magic[8] {};
    in.read(magic, sizeof(magic));
    if (std::memcmp(magic, "TSMPSG2", 7) != 0) {
        return ckpt;
    }

    const uint32_t version = read_u32(in);
    if (version != 2) {
        throw std::runtime_error("unsupported checkpoint version: " + std::to_string(version));
    }

    LlmConfig cfg;
    cfg.vocab_size = read_i32(in);
    cfg.context = read_i32(in);
    cfg.batch = read_i32(in);
    cfg.d_model = read_i32(in);
    cfg.n_heads = read_i32(in);
    cfg.n_layers = read_i32(in);
    cfg.d_ff = read_i32(in);
    cfg.rope_theta = read_f32(in);
    cfg.rms_eps = read_f32(in);
    cfg.beta1 = read_f32(in);
    cfg.beta2 = read_f32(in);
    cfg.adam_eps = read_f32(in);

    const uint32_t n_params = read_u32(in);
    for (uint32_t p = 0; p < n_params; ++p) {
        const uint32_t name_len = read_u32(in);
        std::string name(name_len, '\0');
        in.read(&name[0], name_len);
        const uint32_t ndims = read_u32(in);
        std::vector<int64_t> dims(ndims);
        for (uint32_t i = 0; i < ndims; ++i) {
            dims[i] = int64_t(read_u64(in));
        }
        const int64_t count = product(dims);
        std::vector<float> data((size_t)count);
        in.read(reinterpret_cast<char *>(data.data()), count * sizeof(float));
        if (!in) {
            throw std::runtime_error("truncated checkpoint: " + path);
        }
        ckpt.params.emplace(std::move(name), std::move(data));
    }

    ckpt.found = true;
    ckpt.cfg = cfg;
    return ckpt;
}

struct ParamRef {
    std::string name;
    std::vector<int64_t> dims;
    int64_t count = 0;
    MPSGraphTensor * var = nil;
    MPSGraphTensor * read = nil;
    MPSGraphTensor * m = nil;
    MPSGraphTensor * m_read = nil;
    MPSGraphTensor * v = nil;
    MPSGraphTensor * v_read = nil;
};

struct MetalTransformer {
    LlmConfig cfg;
    MPSGraph * graph = nil;
    id<MTLDevice> metal_device = nil;
    id<MTLCommandQueue> queue = nil;
    MPSGraphDevice * graph_device = nil;
    MPSGraphExecutionDescriptor * exec_desc = nil;
    MPSGraphTensor * input_ids = nil;
    MPSGraphTensor * target_ids = nil;
    MPSGraphTensor * learning_rate = nil;
    MPSGraphTensor * weight_decay = nil;
    MPSGraphTensor * muon_learning_rate = nil;
    MPSGraphTensor * muon_weight_decay = nil;
    MPSGraphTensor * muon_momentum = nil;
    MPSGraphTensor * beta1_power = nil;
    MPSGraphTensor * beta2_power = nil;
    MPSGraphTensor * max_grad_norm = nil;
    MPSGraphTensor * loss = nil;
    MPSGraphTensor * logits = nil;
    MPSGraphTensor * batch0_logits = nil;
    MPSGraphTensor * rope_cos = nil;
    MPSGraphTensor * rope_sin = nil;
    MPSGraphTensor * causal_mask = nil;
    NSMutableArray<MPSGraphOperation *> * train_ops = nil;
    id<MTLBuffer> input_buffer = nil;
    id<MTLBuffer> target_buffer = nil;
    id<MTLBuffer> learning_rate_buffer = nil;
    id<MTLBuffer> weight_decay_buffer = nil;
    id<MTLBuffer> muon_learning_rate_buffer = nil;
    id<MTLBuffer> muon_weight_decay_buffer = nil;
    id<MTLBuffer> muon_momentum_buffer = nil;
    id<MTLBuffer> beta1_power_buffer = nil;
    id<MTLBuffer> beta2_power_buffer = nil;
    id<MTLBuffer> max_grad_norm_buffer = nil;
    MPSGraphTensorData * input_data = nil;
    MPSGraphTensorData * target_data = nil;
    MPSGraphTensorData * learning_rate_data = nil;
    MPSGraphTensorData * weight_decay_data = nil;
    MPSGraphTensorData * muon_learning_rate_data = nil;
    MPSGraphTensorData * muon_weight_decay_data = nil;
    MPSGraphTensorData * muon_momentum_data = nil;
    MPSGraphTensorData * beta1_power_data = nil;
    MPSGraphTensorData * beta2_power_data = nil;
    MPSGraphTensorData * max_grad_norm_data = nil;
    NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> * train_feed_cache = nil;
    NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> * input_feed_cache = nil;
    std::vector<ParamRef> params;
    std::mt19937 rng;

    explicit MetalTransformer(
        LlmConfig c,
        uint32_t seed,
        const std::unordered_map<std::string, std::vector<float>> & initial = {})
        : cfg(c), rng(seed) {
        if (cfg.d_model % cfg.n_heads != 0) {
            throw std::runtime_error("d_model must be divisible by n_heads");
        }
        if ((cfg.d_model / cfg.n_heads) % 2 != 0) {
            throw std::runtime_error("head dimension must be even for RoPE");
        }

        metal_device = MTLCreateSystemDefaultDevice();
        if (!metal_device) {
            throw std::runtime_error("Metal device unavailable");
        }
        queue = [metal_device newCommandQueue];
        if (!queue) {
            throw std::runtime_error("failed to create Metal command queue");
        }
        graph_device = [MPSGraphDevice deviceWithMTLDevice:metal_device];
        graph = [MPSGraph new];
        graph.options = MPSGraphOptionsNone;
        MPSGraphCompilationDescriptor * compile_desc = [MPSGraphCompilationDescriptor new];
        compile_desc.optimizationLevel = MPSGraphOptimizationLevel0;
        exec_desc = [MPSGraphExecutionDescriptor new];
        exec_desc.compilationDescriptor = compile_desc;
        exec_desc.waitUntilCompleted = YES;
        train_ops = [NSMutableArray array];
        build(initial);
    }

    MPSGraphTensor * scalar(float value) {
        return [graph constantWithScalar:value dataType:MPSDataTypeFloat32];
    }

    MPSGraphTensor * add(MPSGraphTensor * a, MPSGraphTensor * b, const std::string & name) {
        return [graph additionWithPrimaryTensor:a secondaryTensor:b name:ns(name)];
    }

    MPSGraphTensor * sub(MPSGraphTensor * a, MPSGraphTensor * b, const std::string & name) {
        return [graph subtractionWithPrimaryTensor:a secondaryTensor:b name:ns(name)];
    }

    MPSGraphTensor * mul(MPSGraphTensor * a, MPSGraphTensor * b, const std::string & name) {
        return [graph multiplicationWithPrimaryTensor:a secondaryTensor:b name:ns(name)];
    }

    MPSGraphTensor * div(MPSGraphTensor * a, MPSGraphTensor * b, const std::string & name) {
        return [graph divisionWithPrimaryTensor:a secondaryTensor:b name:ns(name)];
    }

    MPSGraphTensor * scale(MPSGraphTensor * a, float value, const std::string & name) {
        return mul(a, scalar(value), name);
    }

    MPSGraphTensor * matmul(MPSGraphTensor * a, MPSGraphTensor * b, const std::string & name) {
        return [graph matrixMultiplicationWithPrimaryTensor:a secondaryTensor:b name:ns(name)];
    }

    MPSGraphTensor * cast(MPSGraphTensor * x, MPSDataType type, const std::string & name) {
        return [graph castTensor:x toType:type name:ns(name)];
    }

    MPSGraphTensor * fast_matmul(MPSGraphTensor * a, MPSGraphTensor * b, const std::string & name) {
        MPSGraphTensor * ah = cast(a, MPSDataTypeFloat16, name + ".a.f16");
        MPSGraphTensor * bh = cast(b, MPSDataTypeFloat16, name + ".b.f16");
        MPSGraphTensor * yh = matmul(ah, bh, name + ".matmul.f16");
        return cast(yh, MPSDataTypeFloat32, name + ".f32");
    }

    id<MTLBuffer> make_buffer(size_t bytes, const std::string & label) {
        id<MTLBuffer> buffer = [metal_device newBufferWithLength:std::max<size_t>(bytes, 4)
                                                         options:MTLResourceStorageModeShared];
        if (!buffer) {
            throw std::runtime_error("failed to allocate Metal buffer: " + label);
        }
        buffer.label = ns(label);
        return buffer;
    }

    void write_buffer(id<MTLBuffer> buffer, const void * data, size_t bytes) {
        std::memcpy([buffer contents], data, bytes);
        if ([buffer storageMode] == MTLStorageModeManaged) {
            [buffer didModifyRange:NSMakeRange(0, bytes)];
        }
    }

    void write_scalar(id<MTLBuffer> buffer, float value) {
        write_buffer(buffer, &value, sizeof(value));
    }

    ParamRef & add_param(
        const std::string & name,
        const std::vector<int64_t> & dims,
        float init_std,
        bool ones,
        const std::unordered_map<std::string, std::vector<float>> & initial) {
        const int64_t count = product(dims);
        std::vector<float> data;
        auto it = initial.find(name);
        if (it != initial.end()) {
            if ((int64_t)it->second.size() != count) {
                throw std::runtime_error("checkpoint shape mismatch for " + name);
            }
            data = it->second;
        } else if (ones) {
            data = filled_vec(size_t(count), 1.0f);
        } else {
            data = normal_vec(size_t(count), init_std, rng);
        }

        NSData * ns_data = [NSData dataWithBytes:data.data() length:data.size() * sizeof(float)];
        MPSShape * shape = mps_shape(dims);
        MPSGraphTensor * var = [graph variableWithData:ns_data
                                                 shape:shape
                                              dataType:MPSDataTypeFloat32
                                                  name:ns(name)];
        MPSGraphTensor * read = [graph readVariable:var name:ns(name + ".read")];

        std::vector<float> zeros(size_t(count), 0.0f);
        NSData * zero_data = [NSData dataWithBytes:zeros.data() length:zeros.size() * sizeof(float)];
        MPSGraphTensor * m = [graph variableWithData:zero_data
                                               shape:shape
                                            dataType:MPSDataTypeFloat32
                                                name:ns(name + ".adam_m")];
        MPSGraphTensor * v = [graph variableWithData:zero_data
                                               shape:shape
                                            dataType:MPSDataTypeFloat32
                                                name:ns(name + ".adam_v")];

        ParamRef p;
        p.name = name;
        p.dims = dims;
        p.count = count;
        p.var = var;
        p.read = read;
        p.m = m;
        p.m_read = [graph readVariable:m name:ns(name + ".adam_m.read")];
        p.v = v;
        p.v_read = [graph readVariable:v name:ns(name + ".adam_v.read")];
        params.push_back(p);
        return params.back();
    }

    void create_rope_constants() {
        const int hd = cfg.d_model / cfg.n_heads;
        const int half = hd / 2;
        std::vector<float> c(size_t(cfg.context * half));
        std::vector<float> s(size_t(cfg.context * half));
        for (int t = 0; t < cfg.context; ++t) {
            for (int i = 0; i < half; ++i) {
                const double freq = 1.0 / std::pow(double(cfg.rope_theta), double(2 * i) / double(hd));
                const double angle = double(t) * freq;
                c[size_t(t * half + i)] = float(std::cos(angle));
                s[size_t(t * half + i)] = float(std::sin(angle));
            }
        }
        NSData * cd = [NSData dataWithBytes:c.data() length:c.size() * sizeof(float)];
        NSData * sd = [NSData dataWithBytes:s.data() length:s.size() * sizeof(float)];
        rope_cos = [graph constantWithData:cd
                                     shape:mps_shape({1, 1, cfg.context, half})
                                  dataType:MPSDataTypeFloat32];
        rope_sin = [graph constantWithData:sd
                                     shape:mps_shape({1, 1, cfg.context, half})
                                  dataType:MPSDataTypeFloat32];
    }

    void create_causal_mask() {
        std::vector<float> m(size_t(cfg.context * cfg.context), 0.0f);
        for (int i = 0; i < cfg.context; ++i) {
            for (int j = 0; j < cfg.context; ++j) {
                if (j > i) {
                    m[size_t(i * cfg.context + j)] = -1.0e9f;
                }
            }
        }
        NSData * data = [NSData dataWithBytes:m.data() length:m.size() * sizeof(float)];
        causal_mask = [graph constantWithData:data
                                        shape:mps_shape({1, 1, cfg.context, cfg.context})
                                     dataType:MPSDataTypeFloat32];
    }

    MPSGraphTensor * linear(MPSGraphTensor * x, MPSGraphTensor * w, const std::string & name) {
        return fast_matmul(x, w, name);
    }

    MPSGraphTensor * rms_norm(MPSGraphTensor * x, MPSGraphTensor * gain, const std::string & name) {
        MPSGraphTensor * x2 = [graph squareWithTensor:x name:ns(name + ".square")];
        MPSGraphTensor * sum = [graph reductionSumWithTensor:x2 axis:2 name:ns(name + ".sum")];
        MPSGraphTensor * mean = scale(sum, 1.0f / float(cfg.d_model), name + ".mean");
        MPSGraphTensor * eps = add(mean, scalar(cfg.rms_eps), name + ".eps");
        MPSGraphTensor * rsqrt = [graph reciprocalSquareRootWithTensor:eps name:ns(name + ".rsqrt")];
        return mul(mul(x, rsqrt, name + ".norm"), gain, name + ".gain");
    }

    MPSGraphTensor * apply_rope(MPSGraphTensor * x, const std::string & name) {
        const int hd = cfg.d_model / cfg.n_heads;
        MPSGraphTensor * even = [graph sliceTensor:x
                                           starts:nums({0, 0, 0, 0})
                                             ends:nums({cfg.batch, cfg.n_heads, cfg.context, hd})
                                          strides:nums({1, 1, 1, 2})
                                             name:ns(name + ".even")];
        MPSGraphTensor * odd = [graph sliceTensor:x
                                          starts:nums({0, 0, 0, 1})
                                            ends:nums({cfg.batch, cfg.n_heads, cfg.context, hd})
                                         strides:nums({1, 1, 1, 2})
                                            name:ns(name + ".odd")];
        MPSGraphTensor * even_cos = mul(even, rope_cos, name + ".even_cos");
        MPSGraphTensor * odd_sin = mul(odd, rope_sin, name + ".odd_sin");
        MPSGraphTensor * even_new = sub(even_cos, odd_sin, name + ".even_new");
        MPSGraphTensor * even_sin = mul(even, rope_sin, name + ".even_sin");
        MPSGraphTensor * odd_cos = mul(odd, rope_cos, name + ".odd_cos");
        MPSGraphTensor * odd_new = add(even_sin, odd_cos, name + ".odd_new");
        return [graph concatTensors:@[ even_new, odd_new ]
                          dimension:3
                         interleave:YES
                               name:ns(name + ".interleave")];
    }

    MPSGraphTensor * attention(
        MPSGraphTensor * x,
        MPSGraphTensor * wq,
        MPSGraphTensor * wk,
        MPSGraphTensor * wv,
        MPSGraphTensor * wo,
        const std::string & name) {
        const int hd = cfg.d_model / cfg.n_heads;
        MPSGraphTensor * q = linear(x, wq, name + ".q");
        MPSGraphTensor * k = linear(x, wk, name + ".k");
        MPSGraphTensor * v = linear(x, wv, name + ".v");

        q = [graph reshapeTensor:q withShape:mps_shape({cfg.batch, cfg.context, cfg.n_heads, hd}) name:ns(name + ".q.reshape")];
        k = [graph reshapeTensor:k withShape:mps_shape({cfg.batch, cfg.context, cfg.n_heads, hd}) name:ns(name + ".k.reshape")];
        v = [graph reshapeTensor:v withShape:mps_shape({cfg.batch, cfg.context, cfg.n_heads, hd}) name:ns(name + ".v.reshape")];
        q = [graph transposeTensor:q permutation:nums({0, 2, 1, 3}) name:ns(name + ".q.transpose")];
        k = [graph transposeTensor:k permutation:nums({0, 2, 1, 3}) name:ns(name + ".k.transpose")];
        v = [graph transposeTensor:v permutation:nums({0, 2, 1, 3}) name:ns(name + ".v.transpose")];
        q = apply_rope(q, name + ".q.rope");
        k = apply_rope(k, name + ".k.rope");

        const float att_scale = 1.0f / std::sqrt(float(hd));
        MPSGraphTensor * kt = [graph transposeTensor:k dimension:2 withDimension:3 name:ns(name + ".kt")];
        MPSGraphTensor * scores = fast_matmul(q, kt, name + ".scores");
        scores = scale(scores, att_scale, name + ".scores.scale");
        scores = add(scores, causal_mask, name + ".scores.mask");
        scores = [graph softMaxWithTensor:scores axis:3 name:ns(name + ".softmax")];
        MPSGraphTensor * y = fast_matmul(scores, v, name + ".attv");

        y = [graph transposeTensor:y permutation:nums({0, 2, 1, 3}) name:ns(name + ".out.transpose")];
        y = [graph reshapeTensor:y withShape:mps_shape({cfg.batch, cfg.context, cfg.d_model}) name:ns(name + ".out.reshape")];
        return linear(y, wo, name + ".wo");
    }

    MPSGraphTensor * ffn(
        MPSGraphTensor * x,
        MPSGraphTensor * w1,
        MPSGraphTensor * w2,
        MPSGraphTensor * w3,
        const std::string & name) {
        MPSGraphTensor * gate = linear(x, w1, name + ".gate");
        MPSGraphTensor * sig = [graph sigmoidWithTensor:gate name:ns(name + ".sigmoid")];
        gate = mul(gate, sig, name + ".silu");
        MPSGraphTensor * up = linear(x, w3, name + ".up");
        MPSGraphTensor * hidden = mul(gate, up, name + ".hidden");
        return linear(hidden, w2, name + ".down");
    }

    void create_feed_buffers() {
        const size_t batch_bytes = size_t(cfg.batch) * size_t(cfg.context) * sizeof(int32_t);
        input_buffer = make_buffer(batch_bytes, "feed.input_ids");
        target_buffer = make_buffer(batch_bytes, "feed.target_ids");
        learning_rate_buffer = make_buffer(sizeof(float), "feed.learning_rate");
        weight_decay_buffer = make_buffer(sizeof(float), "feed.weight_decay");
        muon_learning_rate_buffer = make_buffer(sizeof(float), "feed.muon_learning_rate");
        muon_weight_decay_buffer = make_buffer(sizeof(float), "feed.muon_weight_decay");
        muon_momentum_buffer = make_buffer(sizeof(float), "feed.muon_momentum");
        beta1_power_buffer = make_buffer(sizeof(float), "feed.beta1_power");
        beta2_power_buffer = make_buffer(sizeof(float), "feed.beta2_power");
        max_grad_norm_buffer = make_buffer(sizeof(float), "feed.max_grad_norm");

        input_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:input_buffer
                                                             shape:mps_shape({cfg.batch, cfg.context})
                                                          dataType:MPSDataTypeInt32];
        target_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:target_buffer
                                                              shape:mps_shape({cfg.batch, cfg.context})
                                                           dataType:MPSDataTypeInt32];
        learning_rate_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:learning_rate_buffer
                                                                     shape:mps_shape({})
                                                                  dataType:MPSDataTypeFloat32];
        weight_decay_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:weight_decay_buffer
                                                                    shape:mps_shape({})
                                                                 dataType:MPSDataTypeFloat32];
        muon_learning_rate_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:muon_learning_rate_buffer
                                                                          shape:mps_shape({})
                                                                       dataType:MPSDataTypeFloat32];
        muon_weight_decay_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:muon_weight_decay_buffer
                                                                         shape:mps_shape({})
                                                                      dataType:MPSDataTypeFloat32];
        muon_momentum_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:muon_momentum_buffer
                                                                     shape:mps_shape({})
                                                                  dataType:MPSDataTypeFloat32];
        beta1_power_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:beta1_power_buffer
                                                                   shape:mps_shape({})
                                                                dataType:MPSDataTypeFloat32];
        beta2_power_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:beta2_power_buffer
                                                                   shape:mps_shape({})
                                                                dataType:MPSDataTypeFloat32];
        max_grad_norm_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:max_grad_norm_buffer
                                                                     shape:mps_shape({})
                                                                  dataType:MPSDataTypeFloat32];

        train_feed_cache = [NSMutableDictionary dictionaryWithCapacity:10];
        [train_feed_cache setObject:input_data forKey:input_ids];
        [train_feed_cache setObject:target_data forKey:target_ids];
        [train_feed_cache setObject:learning_rate_data forKey:learning_rate];
        [train_feed_cache setObject:weight_decay_data forKey:weight_decay];
        [train_feed_cache setObject:muon_learning_rate_data forKey:muon_learning_rate];
        [train_feed_cache setObject:muon_weight_decay_data forKey:muon_weight_decay];
        [train_feed_cache setObject:muon_momentum_data forKey:muon_momentum];
        [train_feed_cache setObject:beta1_power_data forKey:beta1_power];
        [train_feed_cache setObject:beta2_power_data forKey:beta2_power];
        [train_feed_cache setObject:max_grad_norm_data forKey:max_grad_norm];

        input_feed_cache = [NSMutableDictionary dictionaryWithCapacity:1];
        [input_feed_cache setObject:input_data forKey:input_ids];
    }

    void build(const std::unordered_map<std::string, std::vector<float>> & initial) {
        params.reserve(size_t(3 + 9 * cfg.n_layers));

        input_ids = [graph placeholderWithShape:mps_shape({cfg.batch, cfg.context})
                                       dataType:MPSDataTypeInt32
                                           name:@"input_ids"];
        target_ids = [graph placeholderWithShape:mps_shape({cfg.batch, cfg.context})
                                        dataType:MPSDataTypeInt32
                                            name:@"target_ids"];
        learning_rate = [graph placeholderWithShape:mps_shape({})
                                           dataType:MPSDataTypeFloat32
                                               name:@"learning_rate"];
        weight_decay = [graph placeholderWithShape:mps_shape({})
                                          dataType:MPSDataTypeFloat32
                                              name:@"weight_decay"];
        muon_learning_rate = [graph placeholderWithShape:mps_shape({})
                                                dataType:MPSDataTypeFloat32
                                                    name:@"muon_learning_rate"];
        muon_weight_decay = [graph placeholderWithShape:mps_shape({})
                                               dataType:MPSDataTypeFloat32
                                                   name:@"muon_weight_decay"];
        muon_momentum = [graph placeholderWithShape:mps_shape({})
                                           dataType:MPSDataTypeFloat32
                                               name:@"muon_momentum"];
        beta1_power = [graph placeholderWithShape:mps_shape({})
                                         dataType:MPSDataTypeFloat32
                                             name:@"beta1_power"];
        beta2_power = [graph placeholderWithShape:mps_shape({})
                                         dataType:MPSDataTypeFloat32
                                             name:@"beta2_power"];
        max_grad_norm = [graph placeholderWithShape:mps_shape({})
                                           dataType:MPSDataTypeFloat32
                                               name:@"max_grad_norm"];
        create_feed_buffers();

        create_rope_constants();
        create_causal_mask();

        const float emb_std = 0.02f;
        const float proj_std = 1.0f / std::sqrt(float(cfg.d_model));
        const float ff_std = 1.0f / std::sqrt(float(cfg.d_model));

        ParamRef & tok_emb = add_param("tok_emb", {cfg.vocab_size, cfg.d_model}, emb_std, false, initial);
        std::vector<ParamRef *> rms_att;
        std::vector<ParamRef *> rms_ffn;
        std::vector<ParamRef *> wq;
        std::vector<ParamRef *> wk;
        std::vector<ParamRef *> wv;
        std::vector<ParamRef *> wo;
        std::vector<ParamRef *> w1;
        std::vector<ParamRef *> w2;
        std::vector<ParamRef *> w3;
        rms_att.reserve(cfg.n_layers);
        rms_ffn.reserve(cfg.n_layers);
        wq.reserve(cfg.n_layers);
        wk.reserve(cfg.n_layers);
        wv.reserve(cfg.n_layers);
        wo.reserve(cfg.n_layers);
        w1.reserve(cfg.n_layers);
        w2.reserve(cfg.n_layers);
        w3.reserve(cfg.n_layers);

        for (int i = 0; i < cfg.n_layers; ++i) {
            const std::string p = "layers." + std::to_string(i) + ".";
            rms_att.push_back(&add_param(p + "rms_att", {cfg.d_model}, 0.0f, true, initial));
            rms_ffn.push_back(&add_param(p + "rms_ffn", {cfg.d_model}, 0.0f, true, initial));
            wq.push_back(&add_param(p + "wq", {cfg.d_model, cfg.d_model}, proj_std, false, initial));
            wk.push_back(&add_param(p + "wk", {cfg.d_model, cfg.d_model}, proj_std, false, initial));
            wv.push_back(&add_param(p + "wv", {cfg.d_model, cfg.d_model}, proj_std, false, initial));
            wo.push_back(&add_param(p + "wo", {cfg.d_model, cfg.d_model}, proj_std, false, initial));
            w1.push_back(&add_param(p + "w1_gate", {cfg.d_model, cfg.d_ff}, ff_std, false, initial));
            w2.push_back(&add_param(p + "w2_down", {cfg.d_ff, cfg.d_model}, 1.0f / std::sqrt(float(cfg.d_ff)), false, initial));
            w3.push_back(&add_param(p + "w3_up", {cfg.d_model, cfg.d_ff}, ff_std, false, initial));
        }
        ParamRef & final_norm = add_param("final_norm", {cfg.d_model}, 0.0f, true, initial);
        ParamRef & lm_head = add_param("lm_head", {cfg.d_model, cfg.vocab_size}, emb_std, false, initial);

        MPSGraphTensor * x = [graph gatherWithUpdatesTensor:tok_emb.read
                                              indicesTensor:input_ids
                                                       axis:0
                                            batchDimensions:0
                                                       name:@"tok_emb.gather"];
        for (int i = 0; i < cfg.n_layers; ++i) {
            const std::string p = "layers." + std::to_string(i);
            MPSGraphTensor * ax = rms_norm(x, rms_att[i]->read, p + ".attn_norm");
            MPSGraphTensor * ay = attention(ax, wq[i]->read, wk[i]->read, wv[i]->read, wo[i]->read, p + ".attn");
            x = add(x, ay, p + ".attn_resid");
            MPSGraphTensor * fx = rms_norm(x, rms_ffn[i]->read, p + ".ffn_norm");
            MPSGraphTensor * fy = ffn(fx, w1[i]->read, w2[i]->read, w3[i]->read, p + ".ffn");
            x = add(x, fy, p + ".ffn_resid");
        }
        x = rms_norm(x, final_norm.read, "final_norm");
        logits = linear(x, lm_head.read, "lm_head");
        MPSGraphTensor * first = [graph sliceTensor:logits
                                          dimension:0
                                              start:0
                                             length:1
                                               name:@"generate.batch0"];
        batch0_logits = [graph reshapeTensor:first
                                   withShape:mps_shape({cfg.context, cfg.vocab_size})
                                        name:@"generate.batch0.flat"];
        loss = cross_entropy(logits);
        build_train_ops();
    }

    MPSGraphTensor * cross_entropy(MPSGraphTensor * source_logits) {
        MPSGraphTensor * labels = [graph oneHotWithIndicesTensor:target_ids
                                                           depth:cfg.vocab_size
                                                            axis:2
                                                        dataType:MPSDataTypeFloat32
                                                            name:@"loss.labels"];
        return [graph softMaxCrossEntropyWithSourceTensor:source_logits
                                             labelsTensor:labels
                                                     axis:2
                                            reductionType:MPSGraphLossReductionTypeMean
                                                     name:@"loss.mean"];
    }

    bool is_muon_param(const ParamRef & p) const {
        if (p.dims.size() != 2) {
            return false;
        }
        if (p.name == "tok_emb" || p.name == "lm_head") {
            return false;
        }
        if (p.name.find("w1_gate") != std::string::npos) {
            return false;
        }
        return p.name.rfind("layers.", 0) == 0;
    }

    float adam_weight_decay_scale(const ParamRef & p) const {
        return p.dims.size() == 1 ? 0.0f : 1.0f;
    }

    float muon_lr_multiplier(const ParamRef & p) const {
        const float rows = float(p.dims[0]);
        const float cols = float(p.dims[1]);
        return std::sqrt(std::max(1.0f, rows / cols));
    }

    MPSGraphTensor * transpose2d(MPSGraphTensor * x, const std::string & name) {
        return [graph transposeTensor:x dimension:0 withDimension:1 name:ns(name)];
    }

    MPSGraphTensor * frobenius_norm_2d(MPSGraphTensor * x, const std::string & name) {
        MPSGraphTensor * sq = [graph squareWithTensor:x name:ns(name + ".square")];
        MPSGraphTensor * sum = [graph reductionSumWithTensor:sq axes:nums({0, 1}) name:ns(name + ".sum")];
        sum = [graph reshapeTensor:sum withShape:mps_shape({}) name:ns(name + ".scalar")];
        return [graph squareRootWithTensor:sum name:ns(name + ".sqrt")];
    }

    MPSGraphTensor * polar_express_zeropower(MPSGraphTensor * g, const ParamRef & p, const std::string & name) {
        MPSGraphTensor * norm = frobenius_norm_2d(g, name + ".norm");
        MPSGraphTensor * denom = add(scale(norm, 1.02f, name + ".norm.safety"),
                                     scalar(1.0e-6f),
                                     name + ".norm.denom");
        MPSGraphTensor * x = div(g, denom, name + ".normalized");
        const bool tall = p.dims[0] > p.dims[1];
        const float coeffs[5][3] = {
            {8.156554524902461f, -22.48329292557795f, 15.878769915207462f},
            {4.042929935166739f, -2.808917465908714f, 0.5000178451051316f},
            {3.8916678022926607f, -2.772484153217685f, 0.5060648178503393f},
            {3.285753657755655f, -2.3681294933425376f, 0.46449024233003106f},
            {2.3465413258596377f, -1.7097828382687081f, 0.42323551169305323f},
        };

        for (int i = 0; i < 5; ++i) {
            const std::string it = name + ".pe" + std::to_string(i);
            const float a = coeffs[i][0];
            const float b = coeffs[i][1];
            const float c = coeffs[i][2];
            if (tall) {
                MPSGraphTensor * xt = transpose2d(x, it + ".xt");
                MPSGraphTensor * gram = fast_matmul(xt, x, it + ".xtx");
                MPSGraphTensor * gram2 = fast_matmul(gram, gram, it + ".xtx2");
                MPSGraphTensor * bgram = add(scale(gram, b, it + ".bA"),
                                             scale(gram2, c, it + ".cAA"),
                                             it + ".B");
                MPSGraphTensor * xb = fast_matmul(x, bgram, it + ".XB");
                x = add(scale(x, a, it + ".aX"), xb, it + ".next");
            } else {
                MPSGraphTensor * xt = transpose2d(x, it + ".xt");
                MPSGraphTensor * gram = fast_matmul(x, xt, it + ".xxt");
                MPSGraphTensor * gram2 = fast_matmul(gram, gram, it + ".xxt2");
                MPSGraphTensor * bgram = add(scale(gram, b, it + ".bA"),
                                             scale(gram2, c, it + ".cAA"),
                                             it + ".B");
                MPSGraphTensor * bx = fast_matmul(bgram, x, it + ".BX");
                x = add(scale(x, a, it + ".aX"), bx, it + ".next");
            }
        }
        return x;
    }

    MPSGraphTensor * cautious_weight_decay(
        MPSGraphTensor * step,
        MPSGraphTensor * lr,
        MPSGraphTensor * wd,
        const ParamRef & p,
        const std::string & name) {
        MPSGraphTensor * decay = mul(mul(lr, wd, name + ".lr_wd"), p.read, name + ".raw");
        MPSGraphTensor * same_direction = mul(step, p.read, name + ".step_times_param");
        MPSGraphTensor * pred = [graph greaterThanOrEqualToWithPrimaryTensor:same_direction
                                                             secondaryTensor:scalar(0.0f)
                                                                        name:ns(name + ".mask")];
        MPSGraphTensor * zero = scale(p.read, 0.0f, name + ".zero");
        return [graph selectWithPredicateTensor:pred
                            truePredicateTensor:decay
                           falsePredicateTensor:zero
                                           name:ns(name + ".select")];
    }

    void build_train_ops() {
        NSMutableArray<MPSGraphTensor *> * wrt = [NSMutableArray arrayWithCapacity:params.size()];
        for (const ParamRef & p : params) {
            [wrt addObject:p.read];
        }
        NSDictionary<MPSGraphTensor *, MPSGraphTensor *> * grads =
            [graph gradientForPrimaryTensor:loss withTensors:wrt name:@"grad"];

        MPSGraphTensor * b1 = scalar(cfg.beta1);
        MPSGraphTensor * b2 = scalar(cfg.beta2);
        MPSGraphTensor * one = scalar(1.0f);
        MPSGraphTensor * one_minus_b1 = scalar(1.0f - cfg.beta1);
        MPSGraphTensor * one_minus_b2 = scalar(1.0f - cfg.beta2);
        MPSGraphTensor * one_minus_muon_momentum = sub(one, muon_momentum, "muon.one_minus_momentum");
        MPSGraphTensor * eps = scalar(cfg.adam_eps);
        MPSGraphTensor * one_minus_b1p = sub(one, beta1_power, "adam.one_minus_b1p");
        MPSGraphTensor * one_minus_b2p = sub(one, beta2_power, "adam.one_minus_b2p");
        MPSGraphTensor * grad_sq_sum = scalar(0.0f);
        std::vector<std::pair<const ParamRef *, MPSGraphTensor *>> grad_pairs;
        grad_pairs.reserve(params.size());

        for (const ParamRef & p : params) {
            MPSGraphTensor * g = [grads objectForKey:p.read];
            if (!g) {
                throw std::runtime_error("missing gradient for " + p.name);
            }
            grad_pairs.emplace_back(&p, g);
            NSMutableArray<NSNumber *> * axes = [NSMutableArray arrayWithCapacity:p.dims.size()];
            for (NSUInteger i = 0; i < p.dims.size(); ++i) {
                [axes addObject:@(i)];
            }
            MPSGraphTensor * sq = [graph squareWithTensor:g name:ns(p.name + ".clip.g2")];
            MPSGraphTensor * part = [graph reductionSumWithTensor:sq axes:axes name:ns(p.name + ".clip.sum")];
            part = [graph reshapeTensor:part withShape:mps_shape({}) name:ns(p.name + ".clip.scalar")];
            grad_sq_sum = add(grad_sq_sum, part, p.name + ".clip.accum");
        }

        MPSGraphTensor * grad_norm = [graph squareRootWithTensor:grad_sq_sum name:@"clip.global_norm"];
        MPSGraphTensor * clip_denom = add(grad_norm, scalar(1e-6f), "clip.denom");
        MPSGraphTensor * clip_scale = [graph minimumWithPrimaryTensor:one
                                                       secondaryTensor:div(max_grad_norm, clip_denom, "clip.raw_scale")
                                                                  name:@"clip.scale"];

        for (const auto & item : grad_pairs) {
            const ParamRef & p = *item.first;
            MPSGraphTensor * g = mul(item.second, clip_scale, p.name + ".clip.grad");
            if (is_muon_param(p)) {
                MPSGraphTensor * new_m = add(mul(muon_momentum, p.m_read, p.name + ".muon.m_decay"),
                                             mul(one_minus_muon_momentum, g, p.name + ".muon.m_grad"),
                                             p.name + ".muon.m");
                MPSGraphTensor * nesterov = add(mul(one_minus_muon_momentum, g, p.name + ".muon.nesterov_grad"),
                                                mul(muon_momentum, new_m, p.name + ".muon.nesterov_m"),
                                                p.name + ".muon.nesterov");
                MPSGraphTensor * update = polar_express_zeropower(nesterov, p, p.name + ".muon");
                MPSGraphTensor * eff_lr = scale(muon_learning_rate, muon_lr_multiplier(p), p.name + ".muon.eff_lr");
                MPSGraphTensor * step = mul(eff_lr, update, p.name + ".muon.step");
                MPSGraphTensor * decay = cautious_weight_decay(step, eff_lr, muon_weight_decay, p, p.name + ".muon.decay");
                MPSGraphTensor * new_p = sub(sub(p.read, decay, p.name + ".muon.decayed"),
                                             step,
                                             p.name + ".muon.new_p");
                [train_ops addObject:[graph assignVariable:p.var withValueOfTensor:new_p name:ns(p.name + ".assign")]];
                [train_ops addObject:[graph assignVariable:p.m withValueOfTensor:new_m name:ns(p.name + ".muon.m.assign")]];
            } else {
                MPSGraphTensor * new_m = add(mul(b1, p.m_read, p.name + ".adam.m_decay"),
                                             mul(one_minus_b1, g, p.name + ".adam.m_grad"),
                                             p.name + ".adam.m");
                MPSGraphTensor * gg = [graph squareWithTensor:g name:ns(p.name + ".adam.g2")];
                MPSGraphTensor * new_v = add(mul(b2, p.v_read, p.name + ".adam.v_decay"),
                                             mul(one_minus_b2, gg, p.name + ".adam.v_grad"),
                                             p.name + ".adam.v");
                MPSGraphTensor * mhat = div(new_m, one_minus_b1p, p.name + ".adam.mhat");
                MPSGraphTensor * vhat = div(new_v, one_minus_b2p, p.name + ".adam.vhat");
                MPSGraphTensor * denom = add([graph squareRootWithTensor:vhat name:ns(p.name + ".adam.sqrtv")],
                                             eps,
                                             p.name + ".adam.denom");
                MPSGraphTensor * step = div(mul(learning_rate, mhat, p.name + ".adam.lr_mhat"),
                                            denom,
                                            p.name + ".adam.step");
                MPSGraphTensor * adam_wd = scale(weight_decay, adam_weight_decay_scale(p), p.name + ".adam.wd_scaled");
                MPSGraphTensor * decay = cautious_weight_decay(step, learning_rate, adam_wd, p, p.name + ".adam.decay");
                MPSGraphTensor * new_p = sub(sub(p.read, decay, p.name + ".adam.decayed"),
                                             step,
                                             p.name + ".adam.new_p");
                [train_ops addObject:[graph assignVariable:p.var withValueOfTensor:new_p name:ns(p.name + ".assign")]];
                [train_ops addObject:[graph assignVariable:p.m withValueOfTensor:new_m name:ns(p.name + ".adam.m.assign")]];
                [train_ops addObject:[graph assignVariable:p.v withValueOfTensor:new_v name:ns(p.name + ".adam.v.assign")]];
            }
        }
    }

    NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> * feeds(
        const std::vector<int32_t> & input,
        const std::vector<int32_t> & target,
        float lr,
        float wd,
        float muon_lr,
        float muon_wd,
        float muon_mom,
        float b1p,
        float b2p,
        float clip_norm) {
        const size_t batch_count = size_t(cfg.batch) * size_t(cfg.context);
        if (input.size() != batch_count || target.size() != batch_count) {
            throw std::runtime_error("bad batch size for feeds");
        }
        write_buffer(input_buffer, input.data(), batch_count * sizeof(int32_t));
        write_buffer(target_buffer, target.data(), batch_count * sizeof(int32_t));
        write_scalar(learning_rate_buffer, lr);
        write_scalar(weight_decay_buffer, wd);
        write_scalar(muon_learning_rate_buffer, muon_lr);
        write_scalar(muon_weight_decay_buffer, muon_wd);
        write_scalar(muon_momentum_buffer, muon_mom);
        write_scalar(beta1_power_buffer, b1p);
        write_scalar(beta2_power_buffer, b2p);
        write_scalar(max_grad_norm_buffer, clip_norm <= 0.0f ? 1.0e30f : clip_norm);
        return train_feed_cache;
    }

    NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> * input_feeds(const std::vector<int32_t> & input) {
        const size_t batch_count = size_t(cfg.batch) * size_t(cfg.context);
        if (input.size() != batch_count) {
            throw std::runtime_error("bad input size for feeds");
        }
        write_buffer(input_buffer, input.data(), batch_count * sizeof(int32_t));
        return input_feed_cache;
    }

    float tensor_to_float(MPSGraphTensorData * data) {
        float value = 0.0f;
        MPSNDArray * arr = [data mpsndarray];
        [arr readBytes:&value strideBytes:nil];
        return value;
    }

    std::vector<float> tensor_to_vec(MPSGraphTensorData * data, size_t count) {
        std::vector<float> out(count);
        MPSNDArray * arr = [data mpsndarray];
        [arr readBytes:out.data() strideBytes:nil];
        return out;
    }

    float train_step(
        const std::vector<int32_t> & input,
        const std::vector<int32_t> & target,
        float lr,
        float wd,
        float muon_lr,
        float muon_wd,
        float muon_mom,
        float clip_norm,
        uint64_t step_index,
        bool fetch_loss) {
        @autoreleasepool {
            const float b1p = std::pow(cfg.beta1, float(step_index));
            const float b2p = std::pow(cfg.beta2, float(step_index));
            NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> * f =
                feeds(input, target, lr, wd, muon_lr, muon_wd, muon_mom, b1p, b2p, clip_norm);
            if (fetch_loss) {
                MPSGraphTensorDataDictionary * result =
                    [graph runAsyncWithMTLCommandQueue:queue
                                                 feeds:f
                                         targetTensors:@[ loss ]
                                      targetOperations:train_ops
                                   executionDescriptor:exec_desc];
                return tensor_to_float([result objectForKey:loss]);
            }
            NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> * empty = [NSMutableDictionary dictionary];
            [graph runAsyncWithMTLCommandQueue:queue
                                         feeds:f
                              targetOperations:train_ops
                             resultsDictionary:empty
                           executionDescriptor:exec_desc];
            return std::numeric_limits<float>::quiet_NaN();
        }
    }

    float eval_batch(const std::vector<int32_t> & input, const std::vector<int32_t> & target) {
        @autoreleasepool {
            NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> * f =
                feeds(input, target, 0.0f, 0.0f, 0.0f, 0.0f, 0.95f, 1.0f, 1.0f, 1.0e30f);
            MPSGraphTensorDataDictionary * result =
                [graph runAsyncWithMTLCommandQueue:queue
                                             feeds:f
                                     targetTensors:@[ loss ]
                                  targetOperations:nil
                               executionDescriptor:exec_desc];
            return tensor_to_float([result objectForKey:loss]);
        }
    }

    std::vector<float> logits_for_batch0(const std::vector<int32_t> & input) {
        @autoreleasepool {
            if ((int)input.size() != cfg.batch * cfg.context) {
                throw std::runtime_error("bad input size for logits");
            }
            NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> * f = input_feeds(input);
            MPSGraphTensorDataDictionary * result =
                [graph runAsyncWithMTLCommandQueue:queue
                                             feeds:f
                                     targetTensors:@[ batch0_logits ]
                                  targetOperations:nil
                               executionDescriptor:exec_desc];
            return tensor_to_vec([result objectForKey:batch0_logits], size_t(cfg.context * cfg.vocab_size));
        }
    }

    void save_checkpoint(const std::string & path) {
        @autoreleasepool {
            NSMutableArray<MPSGraphTensor *> * reads = [NSMutableArray arrayWithCapacity:params.size()];
            for (const ParamRef & p : params) {
                [reads addObject:p.read];
            }
            MPSGraphTensorDataDictionary * result =
                [graph runAsyncWithMTLCommandQueue:queue
                                             feeds:@{}
                                     targetTensors:reads
                                  targetOperations:nil
                               executionDescriptor:exec_desc];

            std::ofstream out(path, std::ios::binary);
            if (!out) {
                throw std::runtime_error("failed to write checkpoint: " + path);
            }
            char magic[8] {};
            std::memcpy(magic, "TSMPSG2", 7);
            out.write(magic, sizeof(magic));
            write_u32(out, 2);
            write_i32(out, cfg.vocab_size);
            write_i32(out, cfg.context);
            write_i32(out, cfg.batch);
            write_i32(out, cfg.d_model);
            write_i32(out, cfg.n_heads);
            write_i32(out, cfg.n_layers);
            write_i32(out, cfg.d_ff);
            write_f32(out, cfg.rope_theta);
            write_f32(out, cfg.rms_eps);
            write_f32(out, cfg.beta1);
            write_f32(out, cfg.beta2);
            write_f32(out, cfg.adam_eps);
            write_u32(out, uint32_t(params.size()));
            for (const ParamRef & p : params) {
                write_u32(out, uint32_t(p.name.size()));
                out.write(p.name.data(), p.name.size());
                write_u32(out, uint32_t(p.dims.size()));
                for (int64_t d : p.dims) {
                    write_u64(out, uint64_t(d));
                }
                MPSGraphTensorData * td = [result objectForKey:p.read];
                std::vector<float> values = tensor_to_vec(td, size_t(p.count));
                out.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
            }
            std::cerr << "wrote checkpoint " << path << "\n";
        }
    }
};

static LlmConfig config_from_args(int argc, char ** argv, LlmConfig base = LlmConfig()) {
    LlmConfig cfg = base;
    cfg.vocab_size = get_arg_i(argc, argv, "--vocab-size", cfg.vocab_size);
    cfg.context = get_arg_i(argc, argv, "--context", cfg.context);
    cfg.batch = get_arg_i(argc, argv, "--batch", cfg.batch);
    cfg.d_model = get_arg_i(argc, argv, "--d-model", cfg.d_model);
    cfg.n_heads = get_arg_i(argc, argv, "--heads", cfg.n_heads);
    cfg.n_layers = get_arg_i(argc, argv, "--layers", cfg.n_layers);
    cfg.d_ff = get_arg_i(argc, argv, "--d-ff", cfg.d_ff);
    cfg.rope_theta = get_arg_f(argc, argv, "--rope-theta", cfg.rope_theta);
    cfg.rms_eps = get_arg_f(argc, argv, "--rms-eps", cfg.rms_eps);
    cfg.beta1 = get_arg_f(argc, argv, "--beta1", cfg.beta1);
    cfg.beta2 = get_arg_f(argc, argv, "--beta2", cfg.beta2);
    cfg.adam_eps = get_arg_f(argc, argv, "--adam-eps", cfg.adam_eps);
    return cfg;
}

static TrainConfig train_config_from_args(int argc, char ** argv) {
    TrainConfig cfg;
    cfg.steps = get_arg_u64(argc, argv, "--steps", cfg.steps);
    cfg.lr_steps = get_arg_u64(argc, argv, "--lr-steps", cfg.lr_steps);
    cfg.seed = get_arg_u64(argc, argv, "--seed", cfg.seed);
    cfg.warmup_steps = get_arg_i(argc, argv, "--warmup", cfg.warmup_steps);
    cfg.log_every = get_arg_i(argc, argv, "--log-every", cfg.log_every);
    cfg.valid_every = get_arg_i(argc, argv, "--valid-every", cfg.valid_every);
    cfg.save_every = get_arg_i(argc, argv, "--save-every", cfg.save_every);
    cfg.valid_batches = get_arg_i(argc, argv, "--valid-batches", cfg.valid_batches);
    cfg.final_validate = !has_arg(argc, argv, "--no-final-validate");
    cfg.final_save = !has_arg(argc, argv, "--no-final-save");
    cfg.learning_rate = get_arg_f(argc, argv, "--lr", cfg.learning_rate);
    cfg.min_learning_rate = get_arg_f(argc, argv, "--min-lr", cfg.min_learning_rate);
    cfg.weight_decay = get_arg_f(argc, argv, "--weight-decay", cfg.weight_decay);
    cfg.grad_clip = get_arg_f(argc, argv, "--grad-clip", cfg.grad_clip);
    cfg.muon_learning_rate = get_arg_f(argc, argv, "--muon-lr", cfg.muon_learning_rate);
    cfg.muon_min_learning_rate = get_arg_f(argc, argv, "--muon-min-lr", cfg.muon_min_learning_rate);
    cfg.muon_weight_decay = get_arg_f(argc, argv, "--muon-weight-decay", cfg.muon_weight_decay);
    cfg.muon_momentum_start = get_arg_f(argc, argv, "--muon-momentum-start", cfg.muon_momentum_start);
    cfg.muon_momentum = get_arg_f(argc, argv, "--muon-momentum", cfg.muon_momentum);
    cfg.muon_momentum_warmup = get_arg_i(argc, argv, "--muon-momentum-warmup", cfg.muon_momentum_warmup);
    cfg.muon_momentum_cooldown = get_arg_i(argc, argv, "--muon-momentum-cooldown", cfg.muon_momentum_cooldown);
    return cfg;
}

static float scheduled_lr_for_step(uint64_t step, uint64_t total_steps, int warmup_steps, float peak_lr, float min_lr) {
    if (warmup_steps > 0 && step <= uint64_t(warmup_steps)) {
        return peak_lr * float(step) / float(warmup_steps);
    }
    if (total_steps <= uint64_t(std::max(1, warmup_steps))) {
        return peak_lr;
    }
    const double denom = double(total_steps - uint64_t(warmup_steps));
    const double ratio = std::min(1.0, std::max(0.0, double(step - uint64_t(warmup_steps)) / denom));
    const double coeff = 0.5 * (1.0 + std::cos(3.14159265358979323846 * ratio));
    return float(double(min_lr) + coeff * double(peak_lr - min_lr));
}

static float lr_for_step(uint64_t step, const TrainConfig & cfg) {
    return scheduled_lr_for_step(step, cfg.lr_steps > 0 ? cfg.lr_steps : cfg.steps,
                                 cfg.warmup_steps, cfg.learning_rate, cfg.min_learning_rate);
}

static float muon_lr_for_step(uint64_t step, const TrainConfig & cfg) {
    return scheduled_lr_for_step(step, cfg.lr_steps > 0 ? cfg.lr_steps : cfg.steps,
                                 cfg.warmup_steps, cfg.muon_learning_rate, cfg.muon_min_learning_rate);
}

static float muon_momentum_for_step(uint64_t step, const TrainConfig & cfg) {
    const float lo = cfg.muon_momentum_start;
    const float hi = cfg.muon_momentum;
    const uint64_t schedule_steps = cfg.lr_steps > 0 ? cfg.lr_steps : cfg.steps;
    if (cfg.muon_momentum_warmup > 0 && step < uint64_t(cfg.muon_momentum_warmup)) {
        const float frac = float(step) / float(cfg.muon_momentum_warmup);
        return lo + frac * (hi - lo);
    }
    if (cfg.muon_momentum_cooldown > 0 && schedule_steps > uint64_t(cfg.muon_momentum_cooldown)) {
        const uint64_t start = schedule_steps - uint64_t(cfg.muon_momentum_cooldown);
        if (step > start) {
            const float frac = float(step - start) / float(cfg.muon_momentum_cooldown);
            return hi - std::min(1.0f, frac) * (hi - lo);
        }
    }
    return hi;
}

static void fill_random_batch(
    const TokenFile & tokens,
    int batch,
    int context,
    std::mt19937_64 & rng,
    std::vector<int32_t> & input,
    std::vector<int32_t> & target) {
    if (tokens.n_tokens <= size_t(context + 1)) {
        throw std::runtime_error("token file too small for requested context");
    }
    std::uniform_int_distribution<size_t> dist(0, tokens.n_tokens - size_t(context + 2));
    input.resize(size_t(batch * context));
    target.resize(size_t(batch * context));
    for (int b = 0; b < batch; ++b) {
        const size_t pos = dist(rng);
        for (int t = 0; t < context; ++t) {
            input[size_t(b * context + t)] = int32_t(tokens[pos + size_t(t)]);
            target[size_t(b * context + t)] = int32_t(tokens[pos + size_t(t + 1)]);
        }
    }
}

static bool fill_sequential_batch(
    const TokenFile & tokens,
    int batch,
    int context,
    size_t batch_index,
    std::vector<int32_t> & input,
    std::vector<int32_t> & target) {
    input.resize(size_t(batch * context));
    target.resize(size_t(batch * context));
    const size_t stride = size_t(context);
    const size_t first = batch_index * size_t(batch) * stride;
    if (first + size_t(context + 1) >= tokens.n_tokens) {
        return false;
    }
    for (int b = 0; b < batch; ++b) {
        size_t pos = first + size_t(b) * stride;
        if (pos + size_t(context + 1) >= tokens.n_tokens) {
            pos = tokens.n_tokens - size_t(context + 2);
        }
        for (int t = 0; t < context; ++t) {
            input[size_t(b * context + t)] = int32_t(tokens[pos + size_t(t)]);
            target[size_t(b * context + t)] = int32_t(tokens[pos + size_t(t + 1)]);
        }
    }
    return true;
}

static double validate(MetalTransformer & model, const TokenFile & valid, int max_batches) {
    std::vector<int32_t> input;
    std::vector<int32_t> target;
    double sum = 0.0;
    int batches = 0;
    for (size_t i = 0;; ++i) {
        if (max_batches > 0 && batches >= max_batches) {
            break;
        }
        if (!fill_sequential_batch(valid, model.cfg.batch, model.cfg.context, i, input, target)) {
            break;
        }
        sum += model.eval_batch(input, target);
        ++batches;
    }
    if (batches == 0) {
        throw std::runtime_error("no validation batches available");
    }
    return sum / double(batches);
}

struct TokenizerArgs {
    std::string dataset = "tinystories";
    TokenizerKind kind = TokenizerKind::RegexBpe;
    std::string dir;
    std::string train_tokens;
    std::string valid_tokens;
    int vocab_size = 10000;
    int max_patch_bytes = 8;
    int tokenizer_threads = 0;
    uint64_t tokenizer_train_bytes = 0;
    double superbpe_transition_frac = 0.9;
    int superbpe_transition_vocab = 0;
    bool force_tokenizer = false;
    bool force_encode = false;
};

static TokenizerArgs tokenizer_args_from_args(int argc, char ** argv) {
    TokenizerArgs args;
    args.dataset = get_arg(argc, argv, "--dataset", "tinystories");
    args.kind = parse_tokenizer_kind(get_arg(argc, argv, "--tokenizer", "regex-bpe"));
    args.vocab_size = get_arg_i(argc, argv, "--tokenizer-vocab-size", default_tokenizer_vocab_size(args.dataset));
    args.dir = get_arg(argc, argv, "--tokenizer-dir", default_tokenizer_dir(args.kind, args.dataset, args.vocab_size));
    args.max_patch_bytes = get_arg_i(argc, argv, "--max-patch-bytes", 8);
    args.tokenizer_threads = get_arg_i(argc, argv, "--tokenizer-threads", 0);
    args.tokenizer_train_bytes = get_arg_bytes(argc, argv, "--tokenizer-train-bytes", 0);
    args.superbpe_transition_frac = get_arg_f(argc, argv, "--superbpe-transition-frac", 0.9f);
    args.superbpe_transition_vocab = get_arg_i(argc, argv, "--superbpe-transition", 0);
    args.force_tokenizer = has_arg(argc, argv, "--force-tokenizer");
    args.force_encode = has_arg(argc, argv, "--force-encode") || args.force_tokenizer;
    args.train_tokens = default_train_tokens(args.kind, args.dataset);
    args.valid_tokens = default_valid_tokens(args.kind, args.dataset);
    get_arg_if_present(argc, argv, "--train-tokens", args.train_tokens);
    get_arg_if_present(argc, argv, "--valid-tokens", args.valid_tokens);
    return args;
}

static void ensure_tokens(
    TokenizerKind tokenizer_kind,
    const std::string & tokenizer_dir,
    const std::string & train_text,
    const std::string & valid_text,
    const std::string & train_tokens,
    const std::string & valid_tokens,
    int max_vocab_size,
    int max_patch_bytes,
    int tokenizer_threads,
    uint64_t tokenizer_train_bytes,
    double superbpe_transition_frac,
    int superbpe_transition_vocab,
    bool force_tokenizer,
    bool force_encode) {
    const bool tokenizer_exists = file_exists(tokenizer_dir + "/vocab.json");
    if (force_tokenizer || !tokenizer_exists) {
        if (tokenizer_kind == TokenizerKind::BltPatch) {
            build_blt_vocab(train_text, tokenizer_dir, max_vocab_size, max_patch_bytes);
        } else {
            build_bpe_vocab(train_text, tokenizer_dir, tokenizer_kind, max_vocab_size,
                            superbpe_transition_frac, superbpe_transition_vocab,
                            tokenizer_threads, tokenizer_train_bytes);
        }
    }
    if (!force_encode && file_exists(train_tokens) && file_exists(valid_tokens)) {
        return;
    }
    Tokenizer tok;
    tok.max_patch_bytes = max_patch_bytes;
    tok.load_dir(tokenizer_dir, tokenizer_kind);
    if (force_encode || !file_exists(train_tokens)) {
        encode_file(tok, train_text, train_tokens, tokenizer_threads);
    }
    if (force_encode || !file_exists(valid_tokens)) {
        encode_file(tok, valid_text, valid_tokens, tokenizer_threads);
    }
}

static void train_cmd(int argc, char ** argv) {
    const TokenizerArgs tokenizer_args = tokenizer_args_from_args(argc, argv);
    const std::string train_text = get_arg(argc, argv, "--train-text", default_train_text(tokenizer_args.dataset));
    const std::string valid_text = get_arg(argc, argv, "--valid-text", default_valid_text(tokenizer_args.dataset));
    const std::string checkpoint_path = get_arg(argc, argv, "--checkpoint", default_checkpoint_path(tokenizer_args.dataset, tokenizer_args.kind));
    const std::string log_path = get_arg(argc, argv, "--log", default_log_path(tokenizer_args.dataset, tokenizer_args.kind));
    const bool resume = get_arg_bool(argc, argv, "--resume", false);

    ensure_tokens(tokenizer_args.kind, tokenizer_args.dir, train_text, valid_text,
                  tokenizer_args.train_tokens, tokenizer_args.valid_tokens,
                  tokenizer_args.vocab_size, tokenizer_args.max_patch_bytes,
                  tokenizer_args.tokenizer_threads, tokenizer_args.tokenizer_train_bytes,
                  tokenizer_args.superbpe_transition_frac, tokenizer_args.superbpe_transition_vocab,
                  tokenizer_args.force_tokenizer, tokenizer_args.force_encode);
    std::cerr << "tokenization artifacts ready; starting Metal training\n";
    Tokenizer train_tok;
    train_tok.max_patch_bytes = tokenizer_args.max_patch_bytes;
    train_tok.load_dir(tokenizer_args.dir, tokenizer_args.kind);
    TokenFile train_tokens_file(tokenizer_args.train_tokens);
    TokenFile valid_tokens_file(tokenizer_args.valid_tokens);

    CheckpointData ckpt;
    LlmConfig model_cfg;
    if (resume) {
        ckpt = load_checkpoint(checkpoint_path);
        if (!ckpt.found) {
            throw std::runtime_error("--resume was set but no compatible checkpoint was found: " + checkpoint_path);
        }
        model_cfg = ckpt.cfg;
    }
    model_cfg = config_from_args(argc, argv, model_cfg);
    if (!has_arg(argc, argv, "--vocab-size")) {
        model_cfg.vocab_size = int(train_tok.vocab.size());
    }
    TrainConfig train_cfg = train_config_from_args(argc, argv);

    std::cerr << "Metal device training config: "
              << "tokenizer=" << tokenizer_kind_name(tokenizer_args.kind)
              << " tokenizer_dir=" << tokenizer_args.dir
              << " "
              << "B=" << model_cfg.batch
              << " T=" << model_cfg.context
              << " D=" << model_cfg.d_model
              << " H=" << model_cfg.n_heads
              << " L=" << model_cfg.n_layers
              << " FF=" << model_cfg.d_ff
              << " vocab=" << model_cfg.vocab_size
              << " attention=explicit_causal"
              << " matmul=f16"
              << " optimizer=muon+adam"
              << "\n";

    MetalTransformer model(model_cfg, uint32_t(train_cfg.seed), ckpt.params);
    std::mt19937_64 batch_rng(train_cfg.seed + 1);
    std::vector<int32_t> input;
    std::vector<int32_t> target;

    const bool new_log = !file_exists(log_path) || !resume;
    std::ofstream log(log_path, resume ? std::ios::app : std::ios::trunc);
    if (!log) {
        throw std::runtime_error("failed to open log: " + log_path);
    }
    if (new_log) {
        log << "step\ttrain_loss\tvalid_loss\ttok_per_s\tlr\tmuon_lr\tmuon_momentum\n";
    }

    auto last = std::chrono::steady_clock::now();
    double loss_sum = 0.0;
    uint64_t tok_count = 0;
    uint64_t loss_count = 0;
    for (uint64_t step = 1; step <= train_cfg.steps; ++step) {
        fill_random_batch(train_tokens_file, model_cfg.batch, model_cfg.context, batch_rng, input, target);
        const float lr = lr_for_step(step, train_cfg);
        const float muon_lr = muon_lr_for_step(step, train_cfg);
        const float muon_momentum = muon_momentum_for_step(step, train_cfg);
        const bool fetch_loss = train_cfg.log_every > 0 && (step % uint64_t(train_cfg.log_every) == 0 || step == 1);
        const float loss = model.train_step(input, target, lr, train_cfg.weight_decay,
                                            muon_lr, train_cfg.muon_weight_decay, muon_momentum,
                                            train_cfg.grad_clip, step, fetch_loss);
        if (fetch_loss && std::isfinite(loss)) {
            loss_sum += loss;
            loss_count += 1;
        }
        tok_count += uint64_t(model_cfg.batch) * uint64_t(model_cfg.context);

        if (fetch_loss) {
            const auto now = std::chrono::steady_clock::now();
            const double seconds = std::chrono::duration<double>(now - last).count();
            const double tps = seconds > 0.0 ? double(tok_count) / seconds : 0.0;
            const double avg_loss = loss_sum / double(std::max<uint64_t>(1, loss_count));
            std::cerr << "step " << step
                      << " train_loss " << avg_loss
                      << " tok/s " << tps
                      << " lr " << lr
                      << " muon_lr " << muon_lr
                      << " muon_momentum " << muon_momentum
                      << "\n";
            log << step << '\t' << avg_loss << "\t\t" << tps << '\t' << lr << '\t'
                << muon_lr << '\t' << muon_momentum << '\n';
            log.flush();
            last = now;
            loss_sum = 0.0;
            loss_count = 0;
            tok_count = 0;
        }

        if (train_cfg.valid_every > 0 && step % uint64_t(train_cfg.valid_every) == 0) {
            const double valid_loss = validate(model, valid_tokens_file, train_cfg.valid_batches);
            std::cerr << "step " << step << " valid_loss " << valid_loss << "\n";
            log << step << "\t\t" << valid_loss << "\t\t" << lr << '\t'
                << muon_lr << '\t' << muon_momentum << '\n';
            log.flush();
        }

        if (train_cfg.save_every > 0 && step % uint64_t(train_cfg.save_every) == 0) {
            model.save_checkpoint(checkpoint_path);
        }
    }

    if (train_cfg.final_validate) {
        const double final_valid = validate(model, valid_tokens_file, train_cfg.valid_batches);
        std::cerr << "final valid_loss " << final_valid << "\n";
    }
    if (train_cfg.final_save) {
        model.save_checkpoint(checkpoint_path);
    }
}

static void validate_cmd(int argc, char ** argv) {
    const TokenizerArgs tokenizer_args = tokenizer_args_from_args(argc, argv);
    const std::string checkpoint_path = get_arg(argc, argv, "--checkpoint", default_checkpoint_path(tokenizer_args.dataset, tokenizer_args.kind));
    const int batches = get_arg_i(argc, argv, "--batches", 0);
    CheckpointData ckpt = load_checkpoint(checkpoint_path);
    if (!ckpt.found) {
        throw std::runtime_error("no compatible checkpoint found: " + checkpoint_path);
    }
    LlmConfig cfg = config_from_args(argc, argv, ckpt.cfg);
    TokenFile valid_tokens_file(tokenizer_args.valid_tokens);
    MetalTransformer model(cfg, 1, ckpt.params);
    const double loss = validate(model, valid_tokens_file, batches);
    std::cout << "valid_loss " << loss << "\n";
}

static int sample_from_logits(
    const float * logits,
    int vocab_size,
    float temperature,
    float top_p,
    std::mt19937 & rng) {
    if (temperature <= 0.0f) {
        return int(std::max_element(logits, logits + vocab_size) - logits);
    }
    std::vector<std::pair<float, int>> probs;
    probs.reserve(size_t(vocab_size));
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < vocab_size; ++i) {
        max_logit = std::max(max_logit, logits[i] / temperature);
    }
    double sum = 0.0;
    for (int i = 0; i < vocab_size; ++i) {
        const float p = std::exp(logits[i] / temperature - max_logit);
        probs.emplace_back(p, i);
        sum += p;
    }
    for (auto & p : probs) {
        p.first = float(double(p.first) / sum);
    }
    std::sort(probs.begin(), probs.end(), [](const auto & a, const auto & b) {
        return a.first > b.first;
    });
    double cdf = 0.0;
    size_t keep = 0;
    for (; keep < probs.size(); ++keep) {
        cdf += probs[keep].first;
        if (cdf >= double(top_p)) {
            ++keep;
            break;
        }
    }
    keep = std::max<size_t>(1, std::min(keep, probs.size()));
    std::uniform_real_distribution<float> dist(0.0f, float(cdf));
    const float r = dist(rng);
    float running = 0.0f;
    for (size_t i = 0; i < keep; ++i) {
        running += probs[i].first;
        if (r <= running) {
            return probs[i].second;
        }
    }
    return probs[keep - 1].second;
}

static void generate_cmd(int argc, char ** argv) {
    const TokenizerArgs tokenizer_args = tokenizer_args_from_args(argc, argv);
    const std::string checkpoint_path = get_arg(argc, argv, "--checkpoint", default_checkpoint_path(tokenizer_args.dataset, tokenizer_args.kind));
    const std::string prompt = get_arg(argc, argv, "--prompt", "Once upon a time");
    const int max_new = get_arg_i(argc, argv, "--max-new", 128);
    const float temperature = get_arg_f(argc, argv, "--temperature", 0.8f);
    const float top_p = get_arg_f(argc, argv, "--top-p", 0.9f);
    const uint64_t seed = get_arg_u64(argc, argv, "--seed", 42);

    CheckpointData ckpt = load_checkpoint(checkpoint_path);
    if (!ckpt.found) {
        throw std::runtime_error("no compatible checkpoint found: " + checkpoint_path);
    }
    LlmConfig cfg = config_from_args(argc, argv, ckpt.cfg);
    cfg.batch = 1;

    Tokenizer tok;
    tok.max_patch_bytes = tokenizer_args.max_patch_bytes;
    tok.load_dir(tokenizer_args.dir, tokenizer_args.kind);
    if (int(tok.vocab.size()) != cfg.vocab_size) {
        throw std::runtime_error("checkpoint vocab_size=" + std::to_string(cfg.vocab_size) +
                                 " but tokenizer vocab_size=" + std::to_string(tok.vocab.size()));
    }
    std::vector<uint32_t> ids = tok.encode(prompt);
    if (ids.empty()) {
        ids.push_back(tok.special_id);
    }

    MetalTransformer model(cfg, uint32_t(seed), ckpt.params);
    std::mt19937 rng{uint32_t(seed)};
    std::vector<int32_t> input(size_t(cfg.context), int32_t(tok.special_id));

    for (int step = 0; step < max_new; ++step) {
        std::fill(input.begin(), input.end(), int32_t(tok.special_id));
        const int window = std::min<int>(cfg.context, int(ids.size()));
        const int start = int(ids.size()) - window;
        for (int i = 0; i < window; ++i) {
            input[size_t(i)] = int32_t(ids[size_t(start + i)]);
        }
        std::vector<float> all_logits = model.logits_for_batch0(input);
        const float * next_logits = all_logits.data() + size_t(window - 1) * size_t(cfg.vocab_size);
        const int next = sample_from_logits(next_logits, cfg.vocab_size, temperature, top_p, rng);
        ids.push_back(uint32_t(next));
        if (uint32_t(next) == tok.special_id) {
            break;
        }
    }

    std::cout << tok.decode(ids) << "\n";
}

static void encode_cmd(int argc, char ** argv) {
    const TokenizerArgs tokenizer_args = tokenizer_args_from_args(argc, argv);
    const std::string train_text = get_arg(argc, argv, "--train-text", default_train_text(tokenizer_args.dataset));
    const std::string valid_text = get_arg(argc, argv, "--valid-text", default_valid_text(tokenizer_args.dataset));
    ensure_tokens(tokenizer_args.kind, tokenizer_args.dir, train_text, valid_text,
                  tokenizer_args.train_tokens, tokenizer_args.valid_tokens,
                  tokenizer_args.vocab_size, tokenizer_args.max_patch_bytes,
                  tokenizer_args.tokenizer_threads, tokenizer_args.tokenizer_train_bytes,
                  tokenizer_args.superbpe_transition_frac, tokenizer_args.superbpe_transition_vocab,
                  tokenizer_args.force_tokenizer, true);
}

static void attention_smoke_cmd(int argc, char ** argv) {
    AttentionKernelParams p;
    p.batch = uint32_t(get_arg_i(argc, argv, "--batch", 2));
    p.heads = uint32_t(get_arg_i(argc, argv, "--heads", 3));
    p.seq = uint32_t(get_arg_i(argc, argv, "--context", 7));
    p.dim = uint32_t(get_arg_i(argc, argv, "--head-dim", 16));
    p.scale = 1.0f / std::sqrt(float(p.dim));
    const uint64_t seed = get_arg_u64(argc, argv, "--seed", 123);
    const size_t n = size_t(p.batch) * p.heads * p.seq * p.dim;

    std::mt19937 rng{uint32_t(seed)};
    std::normal_distribution<float> dist(0.0f, 0.25f);
    std::vector<float> q(n), k(n), v(n), dy(n);
    for (size_t i = 0; i < n; ++i) {
        q[i] = dist(rng);
        k[i] = dist(rng);
        v[i] = dist(rng);
        dy[i] = dist(rng);
    }

    std::vector<float> cpu_y, cpu_dq, cpu_dk, cpu_dv;
    cpu_attention_forward_backward(q, k, v, dy, p, cpu_y, cpu_dq, cpu_dk, cpu_dv);

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        throw std::runtime_error("Metal device unavailable");
    }
    MetalAttentionKernels kernels(device);
    std::vector<float> metal_y, metal_dq, metal_dk, metal_dv;
    kernels.run(q, k, v, dy, p, metal_y, metal_dq, metal_dk, metal_dv);

    const double y_err = max_abs_diff(cpu_y, metal_y);
    const double dq_err = max_abs_diff(cpu_dq, metal_dq);
    const double dk_err = max_abs_diff(cpu_dk, metal_dk);
    const double dv_err = max_abs_diff(cpu_dv, metal_dv);
    std::cout << "attention_smoke"
              << " B=" << p.batch
              << " H=" << p.heads
              << " T=" << p.seq
              << " D=" << p.dim
              << " max_abs_y=" << y_err
              << " max_abs_dq=" << dq_err
              << " max_abs_dk=" << dk_err
              << " max_abs_dv=" << dv_err
              << "\n";
    const double tol = get_arg_f(argc, argv, "--tol", 2e-5f);
    if (y_err > tol || dq_err > tol || dk_err > tol || dv_err > tol) {
        throw std::runtime_error("attention smoke failed tolerance " + std::to_string(tol));
    }
}

static void attention_bench_cmd(int argc, char ** argv) {
    AttentionKernelParams p;
    p.batch = uint32_t(get_arg_i(argc, argv, "--batch", 16));
    p.heads = uint32_t(get_arg_i(argc, argv, "--heads", 8));
    p.seq = uint32_t(get_arg_i(argc, argv, "--context", 256));
    p.dim = uint32_t(get_arg_i(argc, argv, "--head-dim", 64));
    p.scale = 1.0f / std::sqrt(float(p.dim));
    const int warmup = get_arg_i(argc, argv, "--warmup-iters", 5);
    const int iters = get_arg_i(argc, argv, "--iters", 25);
    if (p.batch == 0 || p.heads == 0 || p.seq == 0 || p.dim == 0 || iters <= 0) {
        throw std::runtime_error("invalid attention bench shape or iteration count");
    }
    const uint64_t seed = get_arg_u64(argc, argv, "--seed", 123);
    const size_t n = size_t(p.batch) * p.heads * p.seq * p.dim;

    std::mt19937 rng{uint32_t(seed)};
    std::normal_distribution<float> dist(0.0f, 0.25f);
    std::vector<float> q(n), k(n), v(n), dy(n);
    for (size_t i = 0; i < n; ++i) {
        q[i] = dist(rng);
        k[i] = dist(rng);
        v[i] = dist(rng);
        dy[i] = dist(rng);
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        throw std::runtime_error("Metal device unavailable");
    }
    MetalAttentionKernels kernels(device);
    const double seconds = kernels.benchmark(q, k, v, dy, p, warmup, iters);
    const double ms = 1000.0 * seconds / double(iters);
    const double attn_tok_per_s = double(p.batch) * double(p.seq) * double(iters) / seconds;
    const double row_per_s = double(p.batch) * double(p.heads) * double(p.seq) * double(iters) / seconds;
    const double pair_count = double(p.batch) * double(p.heads) * double(p.seq) * double(p.seq + 1) * 0.5;
    const double ideal_train_tflops = pair_count * double(p.dim) * 12.0 * double(iters) / seconds / 1.0e12;
    std::cout << "attention_bench"
              << " device=\"" << [[device name] UTF8String] << "\""
              << " B=" << p.batch
              << " H=" << p.heads
              << " T=" << p.seq
              << " D=" << p.dim
              << " warmup=" << warmup
              << " iters=" << iters
              << " ms_per_iter=" << ms
              << " attn_tok/s=" << attn_tok_per_s
              << " rows/s=" << row_per_s
              << " ideal_train_tflops=" << ideal_train_tflops
              << "\n";
}

static void usage(const char * argv0) {
    std::cerr
        << "usage:\n"
        << "  " << argv0 << " encode [--dataset tinystories|openwebtext] [--tokenizer regex-bpe|superbpe|blt] [--force-tokenizer]\n"
        << "  " << argv0 << " train [--dataset tinystories|openwebtext] [--tokenizer regex-bpe|superbpe|blt] [--force-tokenizer] [--steps N] [--batch B] [--context T]\n"
        << "  " << argv0 << " validate [--dataset tinystories|openwebtext] [--checkpoint path] [--batches N] [--valid-tokens path]\n"
        << "  " << argv0 << " generate [--dataset tinystories|openwebtext] [--tokenizer regex-bpe|superbpe|blt] [--prompt text] [--max-new N]\n\n"
        << "  " << argv0 << " attention-smoke [--batch B] [--heads H] [--context T] [--head-dim D]\n\n"
        << "  " << argv0 << " attention-bench [--batch B] [--heads H] [--context T] [--head-dim D] [--iters N]\n\n"
        << "tokenizer options: --tokenizer-dir DIR --tokenizer-vocab-size N --superbpe-transition N\n"
        << "                   --superbpe-transition-frac F --tokenizer-threads N --tokenizer-train-bytes N|4G --max-patch-bytes N\n"
        << "                   --force-tokenizer --force-encode\n"
        << "OpenWebText defaults use OpenWebText-train.txt/OpenWebText-valid.txt, 32k vocab, and train continues after tokenization in one process.\n"
        << "training uses explicit causal attention, f16 matmul, Muon for hidden projection matrices, and Adam for embeddings/norms/gates/lm_head.\n"
        << "optimizer options: --lr F --min-lr F --weight-decay F --muon-lr F --muon-min-lr F --muon-weight-decay F\n"
        << "                   --muon-momentum-start F --muon-momentum F --muon-momentum-warmup N --muon-momentum-cooldown N --grad-clip F --lr-steps N\n"
        << "benchmark options: --no-final-validate --no-final-save\n"
        << "training defaults write " << DEFAULT_CHECKPOINT << " and " << DEFAULT_LOG << "\n";
}

int main(int argc, char ** argv) {
    @autoreleasepool {
        try {
            if (argc < 2) {
                usage(argv[0]);
                return 2;
            }
            const std::string cmd = argv[1];
            if (cmd == "encode") {
                encode_cmd(argc, argv);
            } else if (cmd == "train") {
                train_cmd(argc, argv);
            } else if (cmd == "validate") {
                validate_cmd(argc, argv);
            } else if (cmd == "generate") {
                generate_cmd(argc, argv);
            } else if (cmd == "attention-smoke") {
                attention_smoke_cmd(argc, argv);
            } else if (cmd == "attention-bench") {
                attention_bench_cmd(argc, argv);
            } else {
                usage(argv[0]);
                return 2;
            }
            return 0;
        } catch (const std::exception & e) {
            std::cerr << "error: " << e.what() << "\n";
            return 1;
        }
    }
}
