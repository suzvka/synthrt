#include <filesystem>
#include <iostream>
#include <string>

#include <stdcorelib/system.h>
#include <stdcorelib/console.h>
#include <stdcorelib/path.h>
#include <synthrt/Support/Logging.h>

#include "Archive.h"

namespace fs = std::filesystem;

using ContentCheck = std::function<bool(const std::vector<char> &)>;
using UninstallCallBack = std::function<bool()>;

static srt::LogCategory cliLog("unpacker");
static void log_report_callback(int level, const srt::LogContext &ctx, const std::string_view &msg) {
	using namespace srt;
	using namespace stdc;

	if (level < Logger::Success) {
		return;
	}

	auto t = std::time(nullptr);
	auto tm = std::localtime(&t);

	std::stringstream ss;
	ss << std::put_time(tm, "%Y-%m-%d %H:%M:%S");
	auto dts = ss.str();

	int foreground, background;
	switch (level) {
		case Logger::Success:
			foreground = console::lightgreen;
			background = foreground;
			break;
		case Logger::Warning:
			foreground = console::yellow;
			background = foreground;
			break;
		case Logger::Critical:
		case Logger::Fatal:
			foreground = console::red;
			background = foreground;
			break;
		default:
			foreground = console::nocolor;
			background = console::white;
			break;
	}

	const char *sig;
	switch (level) {
		case Logger::Trace:
			sig = "T";
			break;
		case Logger::Debug:
			sig = "D";
			break;
		case Logger::Success:
			sig = "S";
			break;
		case Logger::Warning:
			sig = "W";
			break;
		case Logger::Critical:
			sig = "C";
			break;
		case Logger::Fatal:
			sig = "F";
			break;
		default:
			sig = "I";
			break;
	}
	console::printf(console::nostyle, foreground, console::nocolor, "[%s] %-15s", dts.c_str(),
					ctx.category);
	console::printf(console::nostyle, console::nocolor, background, " %s ", sig);
	console::printf(console::nostyle, console::nocolor, console::nocolor, "  ");
	console::println(console::nostyle, foreground, console::nocolor, msg);
}
static inline std::string exception_message(const std::exception &e) {
	std::string msg = e.what();
#ifdef _WIN32
	if (typeid(e) == typeid(fs::filesystem_error)) {
		auto &err = static_cast<const fs::filesystem_error &>(e);
		msg = stdc::wstring_conv::to_utf8(stdc::wstring_conv::from_ansi(err.what()));
	}
#endif
	return msg;
}

static bool testCheck(const std::vector<char> &fileData) {
    return true;
}

//====================================================================================
// Install the package to the specified location (must be a package that complies with the rules)
//------------------------------------------------------------------------------
// @param packerPath Package path
// @param outputDir Output directory
// @param checkFilePath Check file path for custom file check rules
// @param checkFunction: Custom file check rule: the file content will be passed to this function,
// and an empty data block will be returned if the target file is abnormal
// 
//------------------------------------------------------------------------------
static int installPackage(
	const fs::path &packerPath, 
	const fs::path &outputDir,
	const fs::path &checkFilePath = "",
    ContentCheck	checkFunction = [](const std::vector<char> &) { return true; }) {
	Archive package(packerPath);

    ArchiveRule check(package);
    if (!checkFilePath.empty()) {
        check.addRule(checkFilePath, checkFunction);
    }
	
	if (!check.check()) {
        throw std::runtime_error(stdc::formatN(R"(Unrecognized package: "%1")", packerPath));
	}

	if (package.allExtractTo(outputDir) != Archive::ErrorCode::None) {
        throw std::runtime_error("Failed to extract package to: " + outputDir.string());
    }

	return 0;
}

//====================================================================================
// Uninstall the specified installed package (must be a package that complies with the rules)
//------------------------------------------------------------------------------
// @param installedDir: Installed directory
// @param checkFilePath Check: file path for custom file check rules
// @param checkFunction: Custom file check rule: the file content will be passed to this function, 
// and an empty data block will be returned if the target file is abnormal
// @param uninstallCallback: Callback triggered before uninstallation, return true to continue uninstallation
// 
//------------------------------------------------------------------------------
static int uninstallPackage(
    const fs::path &installedDir, 
	const fs::path &checkFilePath		= "",
	ContentCheck	checkFunction		= [](const std::vector<char> &) { return true; },
	UninstallCallBack uninstallCallback = []() { return true; }
    ) {
    if (!uninstallCallback()) return -1;

    ArchiveRule check(installedDir);
    if (!checkFilePath.empty()) {
        check.addRule(checkFilePath, checkFunction);
    }

    if (!check.check()) {
        throw std::runtime_error("Unrecognized installation at: " + installedDir.string());
    }

    std::error_code ec;
    fs::remove_all(installedDir, ec);
    if (ec) {
        throw std::runtime_error("Failed to uninstall from: " + installedDir.string() + " - " + ec.message());
    }

    return 0;
}

//====================================================================================
// Test method
// Only all install
//------------------------------------------------------------------------------
// Command line
// dsinfer-package.exe "C:\path\to\package.zip" "C:\path\to\output"
//------------------------------------------------------------------------------
// Keyboard input
// - packerPath
// - outputDir
//------------------------------------------------------------------------------
int main(int /*argc*/, char * /*argv*/[]) {
	auto cmdline = stdc::system::command_line_arguments();
	std::string zipPath;
	std::string outputDir;

	if (cmdline.size() < 3) {
		stdc::u8println("Enter the path to the zip package:");
		std::getline(std::cin, zipPath);
		stdc::u8println("Enter the output directory:");
		std::getline(std::cin, outputDir);
	} 
	else {
		zipPath = cmdline[1];
		outputDir = cmdline[2];
	}

	srt::Logger::setLogCallback(log_report_callback);

	int ret;
	try {
        ret = installPackage(zipPath, outputDir);
	} 
	catch (const std::exception &e) {
		std::string msg = exception_message(e);
		stdc::console::critical("Error: %1", msg);
		ret = -1;
	}
	return ret;
}