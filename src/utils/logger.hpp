#pragma once

#include <string>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <thread>

/**
 * @file logger.hpp
 * @brief Thread-safe logging system for Heston stochastic volatility model
 * @author Quantitative Finance Team
 * @version 1.0
 * @date 2025
 */

namespace Utils {

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    CRITICAL = 4
};

std::string to_string(LogLevel level);

class Logger {
private:
    std::string component_name_;
    static std::mutex global_mutex_;
    static std::ofstream log_file_;
    static LogLevel min_level_;
    static bool console_output_;
    static bool file_output_;
    static std::string log_filename_;
    static size_t max_file_size_;
    static size_t current_file_size_;
    static int max_log_files_;
    
    static std::string get_timestamp();
    static std::string get_thread_id();
    static void rotate_log_files();
    void write_log(LogLevel level, const std::string& message);

public:
    explicit Logger(const std::string& component_name);
    ~Logger();
    
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&) = default;
    Logger& operator=(Logger&&) = default;
    
    static void configure(
        LogLevel min_level = LogLevel::INFO,
        bool console_output = true,
        bool file_output = true,
        const std::string& log_filename = "heston_model.log",
        size_t max_file_size = 10 * 1024 * 1024,
        int max_log_files = 5
    );
    
    template<typename... Args>
    void debug(const std::string& format, Args&&... args) {
        if (min_level_ <= LogLevel::DEBUG) {
            log(LogLevel::DEBUG, format, std::forward<Args>(args)...);
        }
    }
    
    template<typename... Args>
    void info(const std::string& format, Args&&... args) {
        if (min_level_ <= LogLevel::INFO) {
            log(LogLevel::INFO, format, std::forward<Args>(args)...);
        }
    }
    
    template<typename... Args>
    void warning(const std::string& format, Args&&... args) {
        if (min_level_ <= LogLevel::WARNING) {
            log(LogLevel::WARNING, format, std::forward<Args>(args)...);
        }
    }
    
    template<typename... Args>
    void error(const std::string& format, Args&&... args) {
        if (min_level_ <= LogLevel::ERROR) {
            log(LogLevel::ERROR, format, std::forward<Args>(args)...);
        }
    }
    
    template<typename... Args>
    void critical(const std::string& format, Args&&... args) {
        if (min_level_ <= LogLevel::CRITICAL) {
            log(LogLevel::CRITICAL, format, std::forward<Args>(args)...);
        }
    }
    
    static void flush();
    static LogLevel get_level() { return min_level_; }
    static bool is_enabled(LogLevel level) { return level >= min_level_; }

private:
    template<typename... Args>
    void log(LogLevel level, const std::string& format, Args&&... args) {
        try {
            std::ostringstream oss;
            format_string(oss, format, std::forward<Args>(args)...);
            write_log(level, oss.str());
        } catch (const std::exception& e) {
            write_log(LogLevel::ERROR, "Log formatting error: " + std::string(e.what()));
        }
    }
    
    void format_string(std::ostringstream& oss, const std::string& format);
    
    template<typename T, typename... Args>
    void format_string(std::ostringstream& oss, const std::string& format, T&& arg, Args&&... args) {
        size_t pos = format.find("{}");
        if (pos != std::string::npos) {
            oss << format.substr(0, pos) << std::forward<T>(arg);
            format_string(oss, format.substr(pos + 2), std::forward<Args>(args)...);
        } else {
            oss << format;
        }
    }
};

class PerformanceTimer {
private:
    Logger& logger_;
    std::string operation_name_;
    std::chrono::high_resolution_clock::time_point start_time_;
    LogLevel log_level_;

public:
    PerformanceTimer(Logger& logger, const std::string& operation_name, 
                    LogLevel log_level = LogLevel::DEBUG);
    ~PerformanceTimer();
    double elapsed_ms() const;
};

#define PERF_TIMER(logger, name) \
    Utils::PerformanceTimer _perf_timer(logger, name)

#define PERF_TIMER_LEVEL(logger, name, level) \
    Utils::PerformanceTimer _perf_timer(logger, name, level)

} // namespace Utils

extern Utils::Logger g_logger;