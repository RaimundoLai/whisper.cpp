#pragma once

#include <string>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>

#ifndef O_RDONLY
#define O_RDONLY _O_RDONLY
#endif
#ifndef O_BINARY
#define O_BINARY _O_BINARY
#endif

inline int64_t qwen_get_file_size(const std::string& path) {
    struct _stat64 st;
    if (_stat64(path.c_str(), &st) == 0) {
        return st.st_size;
    }
    return -1;
}

inline void* qwen_mmap(int fd, size_t size) {
    HANDLE hFile = (HANDLE) _get_osfhandle(fd);
    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (hMapping == NULL) {
        return nullptr;
    }
    void* addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMapping);
    return addr;
}

inline void qwen_munmap(void* addr, size_t size) {
    if (addr) {
        UnmapViewOfFile(addr);
    }
}

#ifndef close
#define close _close
#endif
#ifndef open
#define open _open
#endif

#else
// POSIX
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#ifndef O_BINARY
#define O_BINARY 0
#endif

inline int64_t qwen_get_file_size(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        return st.st_size;
    }
    return -1;
}

inline void* qwen_mmap(int fd, size_t size) {
    void* addr = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        return nullptr;
    }
    return addr;
}

inline void qwen_munmap(void* addr, size_t size) {
    if (addr && addr != MAP_FAILED) {
        // On macOS, munmap alone does not immediately release physical pages from
        // MAP_PRIVATE file-backed mappings — the kernel keeps them in the resident
        // set (page cache). This causes GPU OOM on Apple Silicon when starting a
        // new stream before old model pages are evicted.
        //
        // Fix: remap the region with MAP_FIXED | MAP_ANONYMOUS to atomically replace
        // the file-backed pages with zero-fill anonymous pages (which have no physical
        // backing), then munmap the anonymous region to release the virtual address space.
        void* anon = mmap(addr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
        if (anon != MAP_FAILED) {
            munmap(anon, size);
        } else {
            munmap(addr, size);
        }
    }
}

#endif

