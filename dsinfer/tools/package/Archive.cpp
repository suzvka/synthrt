#include "Archive.h"

#include <stdcorelib/system.h>
#include <stdcorelib/console.h>
#include <stdcorelib/path.h>
#include <synthrt/Support/Logging.h>

bit7z::Bit7zLibrary Archive::lib{"7zip.dll"};

Archive::Archive(const std::string &loadPath, const std::string &password) {
    fs::path path(loadPath);

	if (!load(path)){					throw Error(
											stdc::formatN(R"(Failed to extract archive from path: )" + path.generic_u8string()),
											ErrorCode::InvalidArchive
										);
	}
	if (_archive->hasEncryptedItems() && password.empty()) {
										throw Error(stdc::formatN(R"(Archive requires a password)"), ErrorCode::PasswordRequired);
    }

	if (!setPassword(password)){		throw Error(stdc::formatN(R"(Password incorrect)"), ErrorCode::PasswordIncorrect);
	}

	_packagePath = path.generic_u8string();
	_packageName = fs::path(path).filename().string();
	_size = fs::file_size(path);
	_extractedSize = _archive->size();
}

Archive::Archive(const std::vector<char> &data, const std::string &password) {
    if (!load(convert_vector(data)))	throw Error(stdc::formatN(R"(Failed to extract archive)"));
    if (!setPassword(password))			throw Error(stdc::formatN(R"(Password incorrect)"));
}

Archive::Archive(const std::vector<unsigned char> &data, const std::string &password) {
	if (!load(data))					throw Error(stdc::formatN(R"(Failed to extract archive)"));
    if (!setPassword(password))			throw Error(stdc::formatN(R"(Password incorrect)"));
}

bool Archive::setPassword(const std::string &password) {
	if (!_archive || !_archive->hasEncryptedItems())	return true;
    if (!_password.empty() && _password == password)	return true;

	try {
		_archive->setPassword(password);
		_password = password;
        return true;
	} 
	catch (...)											{ return false; }
}

Archive::PreviewView Archive::previewDir(const std::string &path) const {
	const auto items = _archive->items();
    PreviewView directory;
	for (const auto &item : items) {
		if (item.path().find(path) == 0) { // 检查路径前缀
			ArchiveEntry entry;
			entry._path = item.path();
			entry._isDir = item.isDir();
			entry._size = item.size();
            entry._index = item.index();
			std::string name = fs::path(entry._path).filename().string();
			directory[name] = entry;
		}
	}

	return directory;
}

Archive::ErrorCode Archive::allExtractTo(const fs::path &outputPath) const {
    try {
        _archive->extract(outputPath.generic_string());
    } 
	catch (const bit7z::BitException)	{ return ErrorCode::ExtractionFailed; } 
	catch (...)							{ return ErrorCode::UnknownError;}
    return ErrorCode::None;
}

Archive::ErrorCode Archive::extractTo(const std::string &path, const std::string &name, const fs::path &outputPath) const {
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

		_archive->extract(outputPath.generic_string(), std::vector<uint32_t>{entry._index});

		return ErrorCode::None;
	} 
	catch (const bit7z::BitException)	{ return ErrorCode::ExtractionFailed; } 
	catch (...)							{ return ErrorCode::UnknownError;}
    return ErrorCode();
}

bool Archive::hasFile(const std::string &path, const std::string &name) const {
    auto items = previewDir(path);
    return items.find(name) != items.end() && !items.at(name)._isDir;
}

std::vector<char> Archive::getFile(const std::string &path, const std::string &name) const {
    try {
        auto item = previewDir(path);
		if (item.empty())				{ throw Error(stdc::formatN(R"(No such file in archive: )" + path), ErrorCode::DirectoryNotFound); }
        if (!hasFile(path, name))		{ throw Error(stdc::formatN(R"(File not found in archive: )" + name), ErrorCode::FileNotFound); }

        auto &entry = item.find(name)->second;
        std::vector<unsigned char> data;
        _archive->extract(data, entry._index);

		return convert_vector(data);
        
	} 
	catch (const bit7z::BitException &e){ throw Error(stdc::formatN(R"(Failed to extract file: )" + name + R"(, error: )" + e.what()), ErrorCode::ExtractionFailed);} 
	catch (const std::exception &e)		{ throw Error(stdc::formatN(R"(Failed to extract file: )" + name + R"(, error: )" + e.what()));} 
	catch (...)							{ throw Error(stdc::formatN(R"(Failed to extract file: )" + name + R"(, unknown error)"));}
    return std::vector<char>();
}

bool Archive::load(const fs::path &path) {
	try {
		_archive = std::make_unique<bit7z::BitArchiveReader>(
			lib, path.generic_u8string(),
			bit7z::BitFormat::Auto
		);
		return true;
	} 
	catch (...) {
		return false;
	}
}

bool Archive::load(const std::vector<unsigned char> &data) {
	try {
		_archive = std::make_unique<bit7z::BitArchiveReader>(
			lib, data, 
			bit7z::BitFormat::Auto
		);
		return true;
	} 
	catch (...) {
		return false;
	}
}

std::vector<unsigned char> Archive::convert_vector(const std::vector<char> &input) {
	return std::vector<unsigned char>(input.begin(), input.end());
}

std::vector<char> Archive::convert_vector(const std::vector<unsigned char> &input) {
	return std::vector<char>(input.begin(), input.end());
}
