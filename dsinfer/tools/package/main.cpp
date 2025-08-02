#include <filesystem>
#include <iostream>
#include <string>

#include <stdcorelib/system.h>
#include <stdcorelib/console.h>
#include <stdcorelib/path.h>
#include <synthrt/Support/Logging.h>

#include "Archive.h"



namespace fs = std::filesystem;

using CheckFunction = std::function<Archive::ErrorCode(const Archive &)>;

static srt::LogCategory cliLog("unpacker");

static void log_report_callback(int level, const srt::LogContext &ctx,
								const std::string_view &msg) {
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

// 检查压缩包目录结构
// 注入检查规则
// 传入参数为 CompactionPackage 的函数
static Archive::ErrorCode check(
	const Archive &packer, CheckFunction checkFunction) {
	return checkFunction(packer);
}

static int installPackage(
	const fs::path &packerPath, 
	const fs::path &outputDir,
    CheckFunction checkFunction = [](const Archive &) { return Archive::ErrorCode::None; }) {
	Archive package(packerPath.generic_string());
    
	// 检查压缩包
    Archive::ErrorCode code = check(package, checkFunction);
    if (code != Archive::ErrorCode::None) {
        throw std::runtime_error(stdc::formatN(R"(Unrecognized package: "%1")", packerPath));
    }

	// 解压缩包
	cliLog.srtInfo("Extracting package: " + stdc::path::to_utf8(packerPath));
    package.allExtractTo(outputDir);
	cliLog.srtSuccess("Package successfully installed at: " + stdc::path::to_utf8(outputDir));

	return 0;
}

// 主函数 Main
// ===============================
int main(int /*argc*/, char * /*argv*/[]) {
	// 解析命令行参数
	//auto cmdline = stdc::system::command_line_arguments();
	//if (cmdline.size() < 3) {
	//	stdc::u8println("Usage: %1 <zip_package> <output_directory>",
	//					stdc::system::application_name());
	//	return 1;
	//}
	

	// 设置日志回调
	srt::Logger::setLogCallback(log_report_callback);

	//const auto &zipPath = stdc::path::from_utf8(cmdline[1]);
	//const auto &outputDir = stdc::path::from_utf8(cmdline[2]);

	// 改成键盘输入压缩包路径和输出目录
	stdc::u8println("Enter the path to the zip package:");
	std::string zipPath;
	std::getline(std::cin, zipPath);
	stdc::u8println("Enter the output directory:");
	std::string outputDir;
	std::getline(std::cin, outputDir);

	int ret;
	try {
		// 调用核心逻辑
		ret = installPackage(zipPath, outputDir);
	} catch (const std::exception &e) {
		// 统一处理异常
		std::string msg = exception_message(e);
		stdc::console::critical("Error: %1", msg);
		ret = -1;
	}
	return ret;
}