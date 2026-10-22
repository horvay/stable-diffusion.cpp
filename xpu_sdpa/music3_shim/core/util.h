// Stand-in for sd.cpp's core/util.h, so xpu_music3 can build sd.cpp's HF tokenizer without the rest of sd.cpp.
#pragma once
#include <cstdio>
#include <string>
#define LOG_ERROR(...) (std::fprintf(stderr, "[xpu_music3] " __VA_ARGS__), std::fputc('\n', stderr))
#define LOG_WARN(...) (std::fprintf(stderr, "[xpu_music3] " __VA_ARGS__), std::fputc('\n', stderr))
#define LOG_INFO(...) ((void)0)
#define LOG_DEBUG(...) ((void)0)
#define LOG_VERBOSE(...) ((void)0)
inline bool ends_with(const std::string& s, const std::string& e) { return s.size() >= e.size() && s.compare(s.size() - e.size(), e.size(), e) == 0; }
inline std::string trim(const std::string& s) {
    const size_t b = s.find_first_not_of(" \t\n\r\f\v");
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(" \t\n\r\f\v") - b + 1);
}
