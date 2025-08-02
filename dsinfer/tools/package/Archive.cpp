#include "Archive.h"



#include <stdcorelib/system.h>
#include <stdcorelib/console.h>
#include <stdcorelib/path.h>
#include <synthrt/Support/Logging.h>

bit7z::Bit7zLibrary Archive::lib{"7zip.dll"};
static std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;
static std::wstring strToWstr(const std::string &str) {
    return conv.from_bytes(str);
}
static std::string wstrTostr(const std::wstring &wstr) {
    return std::string(wstr.cbegin(), wstr.cend());
}

Archive::Archive(const fs::path &path, const std::string &password) {
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

	_packagePath = path;
	_packageName = fs::path(path).filename();
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

Archive::PreviewView Archive::previewDir(const fs::path &path) const {
	const auto items = _archive->items();
    PreviewView directory;
	for (const auto &item : items) {
        if (item.path().find(path.generic_u8string()) != 0) continue;
        ArchiveEntry entry;
        entry._basePath = strToWstr(item.path());
        entry._isDir = item.isDir();
        entry._size = item.size();
        entry._index = item.index();
        std::wstring name = fs::path(entry._basePath).filename();
        directory[name] = entry;
	}

	return directory;
}

Archive::ErrorCode Archive::allExtractTo(const fs::path &outputPath) const {
    try {
        auto fullOutputPath = outputPath / fs::path(_packageName).stem();

        _archive->extract(fullOutputPath.generic_u8string());
    } 
    catch (const bit7z::BitException&)  { return ErrorCode::ExtractionFailed; } 
    catch (...)                         { return ErrorCode::UnknownError; }
    return ErrorCode::None;
}

Archive::ErrorCode Archive::extractTo(const fs::path &path, const std::wstring &name, const fs::path &outputPath) const {
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

		return ErrorCode::None;
	} 
	catch (const bit7z::BitException)	{ return ErrorCode::ExtractionFailed; } 
	catch (...)							{ return ErrorCode::UnknownError;}
    return ErrorCode();
}

Archive::ErrorCode Archive::extractTo(const fs::path &fullPath, const fs::path &outputPath) const {
    auto parentPath = fullPath.has_parent_path() ? fullPath.parent_path() : "";
    auto name = fullPath.filename();
	if (!hasFile(parentPath, name))     return ErrorCode::FileNotFound;
	
	return extractTo(parentPath.generic_string(), name, outputPath);
}

bool Archive::hasFile(const fs::path &path, const std::wstring &name) const {
    auto items = previewDir(path);
    return items.find(name) != items.end() && !items.at(name)._isDir;
}

bool Archive::hasFile(const fs::path &fullPath) const {
    auto parentPath = fullPath.has_parent_path() ? fullPath.parent_path() : "";
    auto name = fullPath.filename();
    return hasFile(parentPath, name);
}

std::vector<char> Archive::getFile(const fs::path &path, const std::wstring &name) const {
    try {
        auto item = previewDir(path);
		if (item.empty())				{ throw Error(stdc::formatN(R"(No such file in archive: )" + path.generic_u8string()), ErrorCode::DirectoryNotFound); }
        if (!hasFile(path, name))		{ throw Error(stdc::formatN(R"(File not found in archive: )" + wstrTostr(name)), ErrorCode::FileNotFound); }

        auto &entry = item.find(name)->second;
        std::vector<unsigned char> data;
        _archive->extract(data, entry._index);

		return convert_vector(data);
        
	} 
	catch (const bit7z::BitException &e){ throw Error(stdc::formatN(R"(Failed to extract file: )" + wstrTostr(name) + R"(, error: )" + e.what()), ErrorCode::ExtractionFailed);} 
	catch (const std::exception &e)		{ throw Error(stdc::formatN(R"(Failed to extract file: )" + wstrTostr(name) + R"(, error: )" + e.what()));} 
	catch (...)							{ throw Error(stdc::formatN(R"(Failed to extract file: )" + wstrTostr(name) + R"(, unknown error)"));}
}

std::vector<char> Archive::getFile(const fs::path &fullPath) const {
    auto parentPath = fullPath.has_parent_path() ? fullPath.parent_path() : "";
    auto name = fullPath.filename();
    return getFile(parentPath, name);
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

ArchiveRule::ArchiveRule(Archive &archive) {
    _archive = &archive;
    _basePath = archive.path();
}

ArchiveRule::ArchiveRule(const fs::path &path) {
    _basePath = path.generic_string();
    _archive = nullptr;
}

ArchiveRule &ArchiveRule::hasFile(const std::wstring &name) {
    _fileChecks.push_back(name);
    return *this;
}

ArchiveRule &ArchiveRule::hasDir(const std::wstring &name) {
    _fileChecks.push_back(name + L"/");
    return *this;
}

ArchiveRule &ArchiveRule::addRule(const fs::path &path, ContentCheck rule) {
    _contentRules.push_back(std::pair(path, rule));
    return *this;
}

std::vector<char> ArchiveRule::getData(const fs::path &path, const std::wstring &name) const {
    try {
		if (_archive)	return getArchiveData(path, name);
		else			return getFileData(path, name);
    } 
	catch (...) {
        return std::vector<char>();
    }
}

std::vector<char> ArchiveRule::getData(const fs::path &fullPath) const {
    auto parentPath = fullPath.has_parent_path() ? fullPath.parent_path() : "";
    auto fileName = fullPath.filename();

    return getData(parentPath, fileName);
}

bool ArchiveRule::check() const {
    if (_archive) {
        if (!checkArchive())
            return false;
    } else {
        if (!checkFileSystem())
            return false;
    }
    return checkRules();
}

bool ArchiveRule::checkArchive() const {
    for (const auto &file : _fileChecks) {
        if (!_archive->hasFile(strToWstr(""), file))
            return false;
    }
    return true;
}

bool ArchiveRule::checkFileSystem() const {
    for (const auto &file : _fileChecks) {
        if (!fs::exists(fs::path(file)))
            return false;
    }
    return true;
}

bool ArchiveRule::checkRules() const {
    for (const auto &rule : _contentRules) {
		const auto &path = rule.first;
		const auto &check = rule.second;
        std::vector<char> data = getData(path);
		if (!check(data)) {
			return false;
        }
    }
    return true;
}

std::vector<char> ArchiveRule::getArchiveData(const fs::path &path, const std::wstring &name) const{
    return _archive->getFile(path, name);
}

std::vector<char> ArchiveRule::getFileData(const fs::path &path, const std::wstring &name) const {
    auto fullpath = _basePath / path / name;
    std::ifstream file(fullpath, std::ios::binary | std::ios::ate);
    if (!file)
        return {};

    auto size = file.tellg();
    if (size == -1)
        return {};

    std::vector<char> buffer(static_cast<size_t>(size));

    file.seekg(0);
    if (!file.read(buffer.data(), size))
        return {};

    return buffer;
}