#ifndef COMPRESSED_PACKAGE_H
#define COMPRESSED_PACKAGE_H
#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>

#include <bit7z/bit7z.hpp>

#include "synthrt/Support/Expected.h"

class Archive {
public:
	struct ArchiveEntry;
    enum class ErrorCode;
	class ExpectedVoid;
    class ExpectedData;
    class MemoryBuffer;

	using FileName		= std::wstring;
	using Reader		= std::unique_ptr<bit7z::BitArchiveReader>;
    using EnterPassword = std::function<std::string(const std::string &)>;
	using ContentCheck	= std::function<bool(const std::vector<std::byte> &)>;
	using ContentRule	= std::pair<std::filesystem::path, ContentCheck>;
	using PreviewView	= std::unordered_map<FileName, ArchiveEntry>;
	using LastPreview	= std::pair<std::filesystem::path, PreviewView>;
	
	Archive(const std::filesystem::path &loadPath	, const std::string &password = {});
	Archive(const std::vector<std::byte> &data		, const std::string &password = {});
    Archive(const std::filesystem::path &loadPath	, const EnterPassword &enterPasswordCallback);
    Archive(const std::vector<std::byte> &data		, const EnterPassword &enterPasswordCallback);

	std::filesystem::path path() const { return _packagePath; }
	std::wstring name() const { return _packageName; }
	size_t size() const { return _size;}
	size_t extractedSize() const { return _extractedSize; }
	bool isEncrypted() const { return _isEncrypted; }
	bool isValid() const { return _isValid; }

	//====================================================================================
	// Enter password
    //------------------------------------------------------------------------------
	// @param password
	//
    //------------------------------------------------------------------------------
    ExpectedVoid setPassword(
		const std::string &password
	);

	//====================================================================================
	// Preview the directory structure of the specified path layer
    //------------------------------------------------------------------------------
	// @param path
	//
    //------------------------------------------------------------------------------
    const PreviewView& previewDir(
		const std::filesystem::path &path
	) const;

	//====================================================================================
	// Extract all to the specified path
    //------------------------------------------------------------------------------
	// @param outputPath
	//
    //------------------------------------------------------------------------------
    ExpectedVoid allExtractTo(
		const std::filesystem::path& outputPath
	) const;

	//====================================================================================
	// Extract individual files to the specified path
    //------------------------------------------------------------------------------
	// @param path
	// @param name
	// @param outputPath
	//
    //------------------------------------------------------------------------------
    ExpectedVoid extractTo(
		const std::filesystem::path& path, 
		const FileName &name, 
		const std::filesystem::path& outputPath
	) const;

	//====================================================================================
	// Extract individual files to the specified path
    //------------------------------------------------------------------------------
    // @param fullPath
    // @param outputPath
	//
	//------------------------------------------------------------------------------
    ExpectedVoid extractTo(
		const std::filesystem::path& fullPath, 
		const std::filesystem::path& outputPath
	) const;

	//====================================================================================
	// Specifies whether a specified file exists in the path layer
    //------------------------------------------------------------------------------
	// @param path
	// @param name
	//
    //------------------------------------------------------------------------------
    ExpectedVoid hasFile(
		const std::filesystem::path& path, 
		const FileName &name
	) const;

	//====================================================================================
	// Specifies whether a file exists for the path
    //------------------------------------------------------------------------------
    // @param fullPath
	//
    //------------------------------------------------------------------------------
    ExpectedVoid hasFile(
		const std::filesystem::path& fullPath
	) const;

	//====================================================================================
	// Extracts the specified file for the specified path layer
    //------------------------------------------------------------------------------
	// @param path
	// @param name
	//
    //------------------------------------------------------------------------------
    ExpectedData getFile(
		const std::filesystem::path& path, 
		const FileName &name
	) const;

	//====================================================================================
	// Extract files from the specified path
    //------------------------------------------------------------------------------
    // @param fullPath
	//
    //------------------------------------------------------------------------------
    ExpectedData getFile(
		const std::filesystem::path& fullPath
	) const;

private:
    std::filesystem::path _packagePath = std::string();
    FileName _packageName		= std::wstring();
	std::string _password		= std::string();
	size_t _size				= 0;
	size_t _extractedSize		= 0;
	bool _isEncrypted			= false;
	bool _isValid				= false;

	Reader _archive;

	mutable LastPreview _lastPreview = {};

	bool load(const std::filesystem::path &path);
    bool load(const std::vector<std::byte> &data);

	static std::string composeMessage(ErrorCode errorCode, const std::string &message);

	static const bit7z::Bit7zLibrary lib;
    static const std::unordered_map<ErrorCode, std::string> errorText;

	
};

class ArchiveRule {
public:
	ArchiveRule(Archive &archive);
    ArchiveRule(const std::filesystem::path &path);

	ArchiveRule &hasFile(const std::filesystem::path &name);
    ArchiveRule &hasDir(const std::filesystem::path &name);
    ArchiveRule &addRule(const std::filesystem::path &path, const Archive::ContentCheck&);

	Archive::ExpectedVoid check() const;

private:
	Archive *_archive = nullptr;
    std::filesystem::path _basePath;
    std::vector<std::filesystem::path> _fileChecks;
    std::vector<Archive::ContentRule> _contentRules;

	bool checkArchive() const;
	bool checkFileSystem() const;
	bool checkRules() const;

	std::vector<std::byte> getData(const std::filesystem::path &path, const std::wstring &name) const;
    std::vector<std::byte> getData(const std::filesystem::path &fullPath) const;
    std::vector<std::byte> getArchiveData(const std::filesystem::path &path, const std::wstring &name) const;
    std::vector<std::byte> getFileData(const std::filesystem::path &path, const std::wstring &name) const;
};

enum class Archive::ErrorCode {
    FileNotFound,
    DirectoryNotFound,
    PackageNotFound,
    ExtractionFailed,
    PasswordRequired,
    PasswordIncorrect,
    InvalidArchive,
    UnsupportedFormat,
    UnknownError
};

struct Archive::ArchiveEntry {
    std::filesystem::path _basePath = std::string();
    size_t _size = 0;
    bool _isDir = false;
    uint32_t _index = 0;
};

class Archive::ExpectedVoid : public srt::Expected<void> {
public:
    ExpectedVoid() = default;
    ExpectedVoid(ErrorCode errorCode, const std::string &message = "");
};

class Archive::ExpectedData : public srt::Expected<std::vector<std::byte>> {
public:
    ExpectedData() = default;
    ExpectedData(ErrorCode errorCode, const std::string &message = "");
    ExpectedData(const std::vector<std::byte>& data);
};

class Archive::MemoryBuffer : public std::streambuf {
public:
    explicit MemoryBuffer(std::vector<std::byte> &target_vector);

    MemoryBuffer(const std::byte *data, size_t size);

protected:
    std::streamsize xsputn(const char *s, std::streamsize n) override;
    int_type overflow(int_type ch = traits_type::eof()) override;

private:
    std::vector<std::byte> *m_write_vec;
};

#endif