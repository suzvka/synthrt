#include "Archive.h"

#include <bit7z/bitfilecompressor.hpp>
#include <bit7z/bitfileextractor.hpp>
#include <bit7z/bitmemcompressor.hpp>
#include <bit7z/bitmemextractor.hpp>

#include <stdcorelib/system.h>
#include <stdcorelib/console.h>
#include <stdcorelib/path.h>

#define STRINGIFY_DETAIL(x) #x
#define STRINGIFY(x)        STRINGIFY_DETAIL(x)
const bit7z::Bit7zLibrary Archive::lib{STRINGIFY(BIT7Z_SHARED_LIB_NAME)};

const std::unordered_map<Archive::ErrorCode, std::string> Archive::errorText = {
    {Archive::ErrorCode::InvalidArchive,    "Invalid archive format"                  },
    {Archive::ErrorCode::PasswordRequired,  "Password required for encrypted archive" },
    {Archive::ErrorCode::PasswordIncorrect, "Incorrect password for encrypted archive"},
    {Archive::ErrorCode::FileNotFound,      "File not found in archive"               },
    {Archive::ErrorCode::DirectoryNotFound, "Directory not found in archive"          },
    {Archive::ErrorCode::PackageNotFound,   "Package not found"                       },
    {Archive::ErrorCode::UnsupportedFormat, "Unsupported archive format"              },
    {Archive::ErrorCode::ExtractionFailed,  "Failed to extract file from archive"     },
    {Archive::ErrorCode::UnknownError,      "Unknown error occurred"                  }
};

const std::unordered_map<Archive::ErrorCode, srt::Error::Type> Archive::errorCode = {
    {Archive::ErrorCode::InvalidArchive,    srt::Error::InvalidFormat      },
    {Archive::ErrorCode::PasswordRequired,  srt::Error::FeatureNotSupported},
    {Archive::ErrorCode::PasswordIncorrect, srt::Error::InvalidArgument    },
    {Archive::ErrorCode::FileNotFound,      srt::Error::FileNotFound       },
    {Archive::ErrorCode::DirectoryNotFound, srt::Error::FileNotFound       },
    {Archive::ErrorCode::PackageNotFound,   srt::Error::FileNotFound       },
    {Archive::ErrorCode::UnsupportedFormat, srt::Error::FeatureNotSupported},
    {Archive::ErrorCode::ExtractionFailed,  srt::Error::SessionError       },
    {Archive::ErrorCode::UnknownError,      srt::Error::SessionError       }
};

namespace fs = std::filesystem;

Archive::Archive(const fs::path &path, const std::string &password)
    : Archive(path, [password](const std::string &) { return password; }) {
}

Archive::Archive(const std::vector<std::byte> &data, const std::string &password)
    : Archive(data, [password](const std::string &) { return password; }) {
}

Archive::Archive(const std::filesystem::path &loadPath,
                 const EnterPassword &enterPasswordCallback) {
    if (!load(loadPath)) {
        return;
    }

    _packagePath = loadPath;
    _packageName = loadPath.filename();
    _size = fs::file_size(loadPath);
    _isEncrypted = _archive->hasEncryptedItems();

    if (_isEncrypted) {
        auto packageName = stdc::path::to_utf8(name());
        auto password = enterPasswordCallback(packageName);
        if (!setPassword(password)) {
            return;
        }
    }

    _extractedSize = _archive->size();
    _isValid = true;
}

Archive::Archive(const std::vector<std::byte> &data, const EnterPassword &enterPasswordCallback) {
    if (!load(data)) {
        return;
    }

    _isEncrypted = _archive->hasEncryptedItems();

    if (_isEncrypted) {
        auto packageName = stdc::path::to_utf8(name());
        auto password = enterPasswordCallback(packageName);
        if (!setPassword(password)) {
            return;
        }
    }

    _extractedSize = _archive->size();

    _isValid = true;
}

Archive::ExpectedVoid Archive::setPassword(const std::string &password) {
    if (!_archive || !_archive->hasEncryptedItems()) {
        return {};
    }
    if (!_password.empty() && _password == password) {
        return {};
    }

    try {
        _archive->setPassword(password);
        _password = password;
        return {};
    } catch (...) {
        return ExpectedVoid(ErrorCode::PasswordIncorrect);
    }
}

const Archive::PreviewView &Archive::previewDir(const fs::path &path) const {
    if (_lastPreview.first == path) {
        return _lastPreview.second;
    }

    const auto items = _archive->items();
    PreviewView directory;
    for (const auto &item : items) {
        if (item.path().find(path.generic_u8string()) != 0) {
            continue;
        }
        ArchiveEntry entry;
        entry._basePath = stdc::path::from_utf8(item.path());
        entry._isDir = item.isDir();
        entry._size = item.size();
        entry._index = item.index();
        std::wstring name = fs::path(entry._basePath).filename();
        directory[name] = entry;
    }

    _lastPreview.first = path;
    _lastPreview.second = directory;
    return _lastPreview.second;
}

Archive::ExpectedVoid Archive::allExtractTo(const fs::path &outputPath) const {
    try {
        auto fullOutputPath = outputPath / fs::path(_packageName).stem();

        _archive->extract(fullOutputPath.generic_u8string());
        return {};
    } catch (const bit7z::BitException &) {
        return ErrorCode::ExtractionFailed;
    } catch (...) {
        return ErrorCode::UnknownError;
    }
}

Archive::ExpectedVoid Archive::extractTo(const fs::path &path, const FileName &name,
                                         const fs::path &outputPath) const {
    if (!hasFile(path, name)) {
        return ErrorCode::FileNotFound;
    }
    try {
        auto item = previewDir(path);
        if (item.empty()) {
            return ErrorCode::FileNotFound;
        }
        auto &entry = item.find(name)->second;

        if (!fs::exists(outputPath.parent_path())) {
            fs::create_directories(outputPath.parent_path());
        }

        _archive->extract(outputPath.generic_u8string(), std::vector<uint32_t>{entry._index});

        return {};
    } catch (const bit7z::BitException) {
        return ErrorCode::ExtractionFailed;
    } catch (...) {
        return ErrorCode::UnknownError;
    }
}

Archive::ExpectedVoid Archive::extractTo(const fs::path &fullPath,
                                         const fs::path &outputPath) const {
    auto parentPath = fullPath.has_parent_path() ? fullPath.parent_path() : "";
    auto name = fullPath.filename();
    if (!hasFile(parentPath, name)) {
        return ErrorCode::FileNotFound;
    }

    return extractTo(parentPath.generic_string(), name, outputPath);
}

Archive::ExpectedVoid Archive::hasFile(const fs::path &path, const FileName &name) const {
    auto items = previewDir(path);
    if (items.empty()) {
        return ErrorCode::DirectoryNotFound;
    } else if (items.find(name) != items.end() && !items.at(name)._isDir) {
        return {};
    } else {
        return ErrorCode::FileNotFound;
    }
}

Archive::ExpectedVoid Archive::hasFile(const fs::path &fullPath) const {
    auto parentPath = fullPath.has_parent_path() ? fullPath.parent_path() : "";
    auto name = fullPath.filename();
    return hasFile(parentPath, name);
}

Archive::ExpectedData Archive::getFile(const fs::path &path, const FileName &name) const {
    try {
        auto item = previewDir(path);
        if (item.empty()) {
            return ErrorCode::DirectoryNotFound;
        }
        if (!hasFile(path, name)) {
            return ErrorCode::FileNotFound;
        }

        auto &entry = item.find(name)->second;

        std::vector<std::byte> data(entry._size);

        MemoryBuffer buffer(data);
        std::ostream out_stream(&buffer);

        _archive->extractTo(out_stream, entry._index);

        return data;
    } catch (const bit7z::BitException &e) {
        return ErrorCode::ExtractionFailed;
    } catch (const std::exception &e) {
        return {};
    } catch (...) {
        return {};
    }
}

Archive::ExpectedData Archive::getFile(const fs::path &fullPath) const {
    auto parentPath = fullPath.has_parent_path() ? fullPath.parent_path() : "";
    auto name = fullPath.filename();
    return getFile(parentPath, name);
}

bool Archive::load(const fs::path &path) {
    try {
        _archive = std::make_unique<bit7z::BitArchiveReader>(lib, path.generic_u8string(),
                                                             bit7z::BitFormat::Auto);
        return true;
    } catch (...) {
        return false;
    }
}

bool Archive::load(const std::vector<std::byte> &data) {
    if (data.empty()) {
        return false;
    }

    try {
        MemoryBuffer buffer(data.data(), data.size());
        std::istream inArchive(&buffer);

        _archive =
            std::make_unique<bit7z::BitArchiveReader>(lib, inArchive, bit7z::BitFormat::Auto);
        return true;
    } catch (...) {
        return false;
    }
}

ArchiveRule::ArchiveRule(Archive &archive) {
    _archive = &archive;
    _basePath = archive.path();
}

ArchiveRule::ArchiveRule(const fs::path &path) {
    _basePath = path.generic_string();
    _archive = nullptr;
}

ArchiveRule &ArchiveRule::hasFile(const fs::path &name) {
    _fileChecks.push_back(name);
    return *this;
}

ArchiveRule &ArchiveRule::hasDir(const fs::path &name) {
    _fileChecks.push_back(name / "");
    return *this;
}

ArchiveRule &ArchiveRule::addRule(const fs::path &path, const Archive::ContentCheck &rule) {
    _contentRules.push_back(std::pair(path, rule));
    return *this;
}

std::vector<std::byte> ArchiveRule::getData(const fs::path &path, const std::wstring &name) const {
    try {
        if (_archive) {
            return getArchiveData(path, name);
        } else {
            return getFileData(path, name);
        }
    } catch (...) {
        return std::vector<std::byte>();
    }
}

std::vector<std::byte> ArchiveRule::getData(const fs::path &fullPath) const {
    auto parentPath = fullPath.has_parent_path() ? fullPath.parent_path() : "";
    auto fileName = fullPath.filename();

    return getData(parentPath, fileName);
}

Archive::ExpectedVoid ArchiveRule::check() const {
    if (_archive) {
        if (!checkArchive()) {
            return Archive::ErrorCode::FileNotFound;
        }
    } else {
        if (!checkFileSystem()) {
            return Archive::ErrorCode::FileNotFound;
        }
    }
    if (!checkRules()) {
        return Archive::ErrorCode::InvalidArchive;
    }
    return {};
}

bool ArchiveRule::checkArchive() const {
    for (const auto &file : _fileChecks) {
        if (!_archive->hasFile("", file)) {
            return false;
        }
    }
    return true;
}

bool ArchiveRule::checkFileSystem() const {
    for (const auto &file : _fileChecks) {
        if (!fs::exists(fs::path(file))) {
            return false;
        }
    }
    return true;
}

bool ArchiveRule::checkRules() const {
    if (_contentRules.empty()) {
        return true;
    }
    for (const auto &rule : _contentRules) {
        const auto &path = rule.first;
        const auto &check = rule.second;
        std::vector<std::byte> data = getData(path);

        if (!check(data)) {
            return false;
        }
    }
    return true;
}

std::vector<std::byte> ArchiveRule::getArchiveData(const fs::path &path,
                                                   const std::wstring &name) const {
    return _archive->getFile(path, name).get();
}

std::vector<std::byte> ArchiveRule::getFileData(const fs::path &path,
                                                const std::wstring &name) const {
    auto fullpath = _basePath / path / name;
    std::ifstream file(fullpath, std::ios::binary | std::ios::ate);
    if (!file) {
        return {};
    }

    auto size = file.tellg();
    if (size == -1) {
        return {};
    }

    std::vector<std::byte> buffer(static_cast<size_t>(size));

    file.seekg(0);

    if (!file.read(reinterpret_cast<char *>(buffer.data()), size)) {
        return {};
    }

    return buffer;
}

std::string Archive::composeMessage(ErrorCode errorCode, const std::string &message) {
    auto it = errorText.find(errorCode);
    it != errorText.end() ? it : it = errorText.find(Archive::ErrorCode::UnknownError);

    const std::string &baseMsg = it->second;
    if (!message.empty()) {
        return baseMsg + ": " + message;
    }

    return baseMsg;
}

Archive::ExpectedVoid::ExpectedVoid(ErrorCode errorCode, const std::string &message)
    : srt::Expected<void>(
          srt::Error(Archive::errorCode.at(errorCode), composeMessage(errorCode, message))) {
}

Archive::ExpectedData::ExpectedData(ErrorCode errorCode, const std::string &message)
    : srt::Expected<std::vector<std::byte>>(
          srt::Error(Archive::errorCode.at(errorCode), composeMessage(errorCode, message))) {
}

Archive::ExpectedData::ExpectedData(const std::vector<std::byte> &data)
    : srt::Expected<std::vector<std::byte>>(data) {
}

Archive::MemoryBuffer::MemoryBuffer(std::vector<std::byte> &target_vector)
    : m_write_vec(&target_vector) {
}

Archive::MemoryBuffer::MemoryBuffer(const std::byte *data, size_t size) : m_write_vec(nullptr) {
    char *begin = const_cast<char *>(reinterpret_cast<const char *>(data));
    char *end = begin + size;
    this->setg(begin, begin, end);
}

std::streamsize Archive::MemoryBuffer::xsputn(const char *s, std::streamsize n) {
    if (m_write_vec) {
        const auto *byte_s = reinterpret_cast<const std::byte *>(s);
        m_write_vec->insert(m_write_vec->end(), byte_s, byte_s + n);
        return n;
    }
    return std::streambuf::xsputn(s, n);
}

Archive::MemoryBuffer::int_type Archive::MemoryBuffer::overflow(int_type ch) {
    if (m_write_vec) {
        if (ch != traits_type::eof()) {
            m_write_vec->push_back(static_cast<std::byte>(ch));
            return ch;
        }
        return traits_type::eof();
    }
    return std::streambuf::overflow(ch);
}
