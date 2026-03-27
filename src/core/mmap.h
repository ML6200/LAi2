#ifndef LAI_CORE_MMAP_H
#define LAI_CORE_MMAP_H

#include "types.h"
#include <string>

#ifdef _WIN32
    #define LAI_WINDOWS 1
    #include <windows.h>
#else
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
#endif

namespace lai {

// Platform-abstracted memory-mapped file (read-only)
class MappedFile {
public:
    MappedFile() = default;

    ~MappedFile() {
        close();
    }

    // Non-copyable, movable
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    MappedFile(MappedFile&& other) noexcept
        : data_(other.data_), size_(other.size_)
#ifdef LAI_WINDOWS
        , file_handle_(other.file_handle_), mapping_handle_(other.mapping_handle_)
#else
        , fd_(other.fd_)
#endif
    {
        other.data_ = nullptr;
        other.size_ = 0;
#ifdef LAI_WINDOWS
        other.file_handle_ = INVALID_HANDLE_VALUE;
        other.mapping_handle_ = nullptr;
#else
        other.fd_ = -1;
#endif
    }

    MappedFile& operator=(MappedFile&& other) noexcept {
        if (this != &other) {
            close();
            data_ = other.data_;
            size_ = other.size_;
#ifdef LAI_WINDOWS
            file_handle_ = other.file_handle_;
            mapping_handle_ = other.mapping_handle_;
            other.file_handle_ = INVALID_HANDLE_VALUE;
            other.mapping_handle_ = nullptr;
#else
            fd_ = other.fd_;
            other.fd_ = -1;
#endif
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    bool open(const std::string& path) {
        close();

#ifdef LAI_WINDOWS
        file_handle_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                   nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_handle_ == INVALID_HANDLE_VALUE) return false;

        LARGE_INTEGER file_size;
        if (!GetFileSizeEx(file_handle_, &file_size)) {
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
            return false;
        }
        size_ = static_cast<size_t>(file_size.QuadPart);

        mapping_handle_ = CreateFileMapping(file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping_handle_) {
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
            return false;
        }

        data_ = MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0);
        if (!data_) {
            CloseHandle(mapping_handle_);
            CloseHandle(file_handle_);
            mapping_handle_ = nullptr;
            file_handle_ = INVALID_HANDLE_VALUE;
            return false;
        }
#else
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) return false;

        struct stat st;
        if (fstat(fd_, &st) != 0) {
            ::close(fd_);
            fd_ = -1;
            return false;
        }
        size_ = static_cast<size_t>(st.st_size);

        data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            ::close(fd_);
            fd_ = -1;
            return false;
        }

        // Hint: random access pattern (weights are accessed by row)
        madvise(data_, size_, MADV_RANDOM);
#endif

        return true;
    }

    void close() {
#ifdef LAI_WINDOWS
        if (data_) { UnmapViewOfFile(data_); data_ = nullptr; }
        if (mapping_handle_) { CloseHandle(mapping_handle_); mapping_handle_ = nullptr; }
        if (file_handle_ != INVALID_HANDLE_VALUE) { CloseHandle(file_handle_); file_handle_ = INVALID_HANDLE_VALUE; }
#else
        if (data_) { munmap(data_, size_); data_ = nullptr; }
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
#endif
        size_ = 0;
    }

    bool is_open() const { return data_ != nullptr; }
    const void* data() const { return data_; }
    void* data() { return data_; }
    size_t size() const { return size_; }

    // Access at byte offset
    const void* at(size_t offset) const {
        return static_cast<const u8*>(data_) + offset;
    }

private:
    void* data_ = nullptr;
    size_t size_ = 0;

#ifdef LAI_WINDOWS
    HANDLE file_handle_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};

} // namespace lai

#endif // LAI_CORE_MMAP_H
