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
#include <locale>
#include <codecvt>

#include <bit7z/bit7z.hpp>
#include <bit7z/bitfilecompressor.hpp>
#include <bit7z/bitfileextractor.hpp>
#include <bit7z/bitmemcompressor.hpp>
#include <bit7z/bitmemextractor.hpp>


namespace fs = std::filesystem;
class Archive;
using FileName = std::wstring;
using Reader = std::unique_ptr<bit7z::BitArchiveReader>;
using ContentCheck = std::function<bool(const std::vector<char> &)>;
using ContentRule = std::pair<fs::path, ContentCheck>;

class Archive {
	static bit7z::Bit7zLibrary lib;
	
public:
	struct ArchiveEntry;
	enum class ErrorCode;
	class Error;
	using PreviewView = std::unordered_map<FileName, ArchiveEntry>;

	Archive(const fs::path &loadPath				, const std::string &password = {});
	Archive(const std::vector<char> &data			, const std::string &password = {});
	Archive(const std::vector<unsigned char> &data	, const std::string &password = {});

	std::wstring path() const { return _packagePath; }
	std::wstring name() const { return _packageName; }
	size_t size() const { return _size;}
	size_t extractedSize() const { return _extractedSize; }

	//====================================================================================
	// Enter password
    //------------------------------------------------------------------------------
	// @param password
	//
    //------------------------------------------------------------------------------
	bool setPassword(
		const std::string &password
	);

	//====================================================================================
	// Preview the directory structure of the specified path layer
    //------------------------------------------------------------------------------
	// @param path
	//
    //------------------------------------------------------------------------------
	PreviewView previewDir(
		const fs::path &path
	) const;

	//====================================================================================
	// Extract all to the specified path
    //------------------------------------------------------------------------------
	// @param outputPath
	//
    //------------------------------------------------------------------------------
	ErrorCode allExtractTo(
		const fs::path& outputPath
	) const;

	//====================================================================================
	// Extract individual files to the specified path
    //------------------------------------------------------------------------------
	// @param path
	// @param name
	// @param outputPath
	//
    //------------------------------------------------------------------------------
	ErrorCode extractTo(
		const fs::path& path, 
		const std::wstring& name, 
		const fs::path& outputPath
	) const;

	//====================================================================================
	// Extract individual files to the specified path
    //------------------------------------------------------------------------------
    // @param fullPath
    // @param outputPath
	//
	//------------------------------------------------------------------------------
	ErrorCode extractTo(
		const fs::path& fullPath, 
		const fs::path& outputPath
	) const;

	//====================================================================================
	// Specifies whether a specified file exists in the path layer
    //------------------------------------------------------------------------------
	// @param path
	// @param name
	//
    //------------------------------------------------------------------------------
	bool hasFile(
		const fs::path& path, 
		const std::wstring& name
	) const;

	//====================================================================================
	// Specifies whether a file exists for the path
    //------------------------------------------------------------------------------
    // @param fullPath
	//
    //------------------------------------------------------------------------------
	bool hasFile(
		const fs::path& fullPath
	) const;

	//====================================================================================
	// Extracts the specified file for the specified path layer
    //------------------------------------------------------------------------------
	// @param path
	// @param name
	//
    //------------------------------------------------------------------------------
	std::vector<char> getFile(
		const fs::path& path, 
		const std::wstring& name
	) const;

	//====================================================================================
	// Extract files from the specified path
    //------------------------------------------------------------------------------
    // @param fullPath
	//
    //------------------------------------------------------------------------------
	std::vector<char> getFile(
		const fs::path& fullPath
	) const;

private:
    fs::path _packagePath		= std::string();
    std::wstring _packageName	= std::wstring();
	std::string _password		= std::string();
	size_t _size				= 0;
	size_t _extractedSize		= 0;
	bool _isEncrypted			= false;
	bool _isValid				= false;

	Reader _archive;

	bool load(const fs::path &path);
	bool load(const std::vector<unsigned char> &data);

	static std::vector<unsigned char> convert_vector(const std::vector<char> &input);
	static std::vector<char> convert_vector(const std::vector<unsigned char> &input);
};

enum class Archive::ErrorCode {
    None = 0,
    FileNotFound,
    DirectoryNotFound,
    ExtractionFailed,
    PasswordRequired,
    PasswordIncorrect,
    InvalidArchive,
    UnsupportedFormat,
    UnknownError 
};

struct Archive::ArchiveEntry {
    fs::path _basePath = std::string();
    size_t _size = 0;
    bool _isDir = false;
    uint32_t _index = 0;
};

class Archive::Error : public std::runtime_error {
public:
    explicit Error(const std::string &message, ErrorCode errorCode = ErrorCode::UnknownError)
        : std::runtime_error(message) {
    }
};

class ArchiveRule {
public:
	ArchiveRule(Archive &archive);
	ArchiveRule(const fs::path &path);

	ArchiveRule &hasFile(const std::wstring &name);
    ArchiveRule &hasDir(const std::wstring &name);
    ArchiveRule &addRule(const fs::path &path, ContentCheck rule);

	bool check() const;

private:
	Archive *_archive = nullptr;
    fs::path _basePath;
    std::vector<fs::path> _fileChecks;
	std::vector<ContentRule> _contentRules;

	bool checkArchive() const;
	bool checkFileSystem() const;
	bool checkRules() const;

	std::vector<char> getData(const fs::path &path, const std::wstring &name) const;
    std::vector<char> getData(const fs::path &fullPath) const;
    std::vector<char> getArchiveData(const fs::path &path, const std::wstring &name) const;
    std::vector<char> getFileData(const fs::path &path, const std::wstring &name) const;
};

#endif