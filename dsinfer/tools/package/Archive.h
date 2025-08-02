#ifndef COMPRESSED_PACKAGE_H
#define COMPRESSED_PACKAGE_H

#pragma once

#include <filesystem>
#include <iostream>
#include <string>
#include <memory>
#include <unordered_map>

#include <bit7z/bit7z.hpp>
#include <bit7z/bitfilecompressor.hpp>
#include <bit7z/bitfileextractor.hpp>
#include <bit7z/bitmemcompressor.hpp>
#include <bit7z/bitmemextractor.hpp>


namespace fs = std::filesystem;
class Archive;
using FileName = std::string;
using Reader = std::unique_ptr<bit7z::BitArchiveReader>;


class Archive {
	static bit7z::Bit7zLibrary lib;
	
public:
    struct ArchiveEntry;
    enum class ErrorCode;
    class Error;
	using PreviewView = std::unordered_map<FileName, ArchiveEntry>;

	Archive(const std::string &loadPath				, const std::string &password = "");
    Archive(const std::vector<char> &data			, const std::string &password = "");
    Archive(const std::vector<unsigned char> &data	, const std::string &password = "");

    std::string path() const { return _packagePath; }
	std::string name() const { return _packageName; }
	size_t size() const { return _size;}
	size_t extractedSize() const { return _extractedSize; }

	// 输入密码
    bool setPassword(
		const std::string &password
	);

	// 指定路径层的目录结构
    PreviewView previewDir(
		const std::string &path
	) const;

	// 全部解压到指定路径
    ErrorCode allExtractTo(
		const fs::path& outputPath
	) const;

	// 单个文件解压到指定路径
    ErrorCode extractTo(
		const std::string& path, 
		const std::string& name, 
		const fs::path& outputPath
	) const;

	// 指定路径层中是否存在指定文件
    bool hasFile(
		const std::string& path, 
		const std::string& name
	) const;

	// 解压指定路径层的指定文件
	// 返回数据到内存
    std::vector<char> getFile(
		const std::string& path, 
		const std::string& name
	) const;

private:
    std::string _packagePath	= std::string();
    std::string _packageName	= std::string();
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

struct Archive::ArchiveEntry {
    std::string _path;
    size_t _size;
    bool _isDir;
    uint32_t _index;
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

class Archive::Error : public std::runtime_error {
	public:
    explicit Error(const std::string &message, ErrorCode errorCode = ErrorCode::UnknownError)
		: std::runtime_error(message) {}
};


#endif