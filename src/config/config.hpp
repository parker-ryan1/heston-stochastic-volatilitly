#pragma once

#include <string>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include "../utils/logger.hpp"

/**
 * @file config.hpp
 * @brief Configuration management for Heston stochastic volatility model
 * @author Quantitative Finance Team
 * @version 1.0
 * @date 2025
 */

namespace Config {

enum class ValueType {
    STRING,
    INTEGER,
    DOUBLE,
    BOOLEAN
};

class ConfigValue {
private:
    std::string string_value_;
    ValueType type_;
    
public:
    ConfigValue() : type_(ValueType::STRING) {}
    explicit ConfigValue(const std::string& value) : string_value_(value), type_(ValueType::STRING) {}
    explicit ConfigValue(int value) : string_value_(std::to_string(value)), type_(ValueType::INTEGER) {}
    explicit ConfigValue(double value) : string_value_(std::to_string(value)), type_(ValueType::DOUBLE) {}
    explicit ConfigValue(bool value) : string_value_(value ? "true" : "false"), type_(ValueType::BOOLEAN) {}
    
    operator std::string() const { return string_value_; }
    operator int() const { return std::stoi(string_value_); }
    operator double() const { return std::stod(string_value_); }
    operator bool() const { return string_value_ == "true" || string_value_ == "1"; }
    
    ValueType getType() const { return type_; }
    const std::string& getString() const { return string_value_; }
};

class ConfigManager {
private:
    mutable std::mutex mutex_;
    std::map<std::string, ConfigValue> config_map_;
    std::string config_file_path_;
    Utils::Logger logger_;
    
    ConfigManager();
    bool loadFromFile(const std::string& file_path);
    void loadEnvironmentOverrides();
    bool parseJsonContent(const std::string& json_content);
    void setDefaults();
    bool validateConfiguration();

public:
    static ConfigManager& getInstance();
    
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;
    
    bool initialize(const std::string& config_file_path = "config.json");
    
    std::string getString(const std::string& key, const std::string& default_value = "") const;
    int getInt(const std::string& key, int default_value = 0) const;
    double getDouble(const std::string& key, double default_value = 0.0) const;
    bool getBool(const std::string& key, bool default_value = false) const;
    
    void set(const std::string& key, const ConfigValue& value);
    bool hasKey(const std::string& key) const;
    bool saveToFile(const std::string& file_path = "") const;
    bool reload();
    std::vector<std::string> getAllKeys() const;
    void printConfiguration(Utils::LogLevel log_level = Utils::LogLevel::INFO) const;
    
    // Convenience methods for Heston model
    int getMonteCarloSimulations() const { return getInt("monte_carlo.simulations", 100000); }
    int getMonteCarloSteps() const { return getInt("monte_carlo.steps", 252); }
    int getRandomSeed() const { return getInt("monte_carlo.random_seed", 42); }
    
    std::string getDiscretizationScheme() const { return getString("heston.discretization_scheme", "euler"); }
    bool getEnableFellerCheck() const { return getBool("heston.enable_feller_check", true); }
    double getVarianceFloor() const { return getDouble("heston.variance_floor", 1e-8); }
    
    double getConvergenceTolerance() const { return getDouble("numerical.tolerance", 1e-6); }
    int getMaxIterations() const { return getInt("numerical.max_iterations", 1000); }
    
    std::string getLogLevel() const { return getString("logging.level", "INFO"); }
    std::string getLogFile() const { return getString("logging.file", "heston_model.log"); }
    bool getLogToConsole() const { return getBool("logging.console", true); }
    bool getLogToFile() const { return getBool("logging.file_output", true); }
    
    bool getEnableThreadSafety() const { return getBool("threading.enable_safety", true); }
    int getMaxThreads() const { return getInt("threading.max_threads", std::thread::hardware_concurrency()); }
    
    bool getEnableMemoryProfiling() const { return getBool("memory.enable_profiling", false); }
    bool getEnablePerformanceLogging() const { return getBool("performance.enable_logging", true); }
    
    // VIX options specific
    double getVIXScalingFactor() const { return getDouble("vix.scaling_factor", 100.0); }
    int getVIXAveragingDays() const { return getInt("vix.averaging_days", 30); }
    
    // Risk management
    double getVaRConfidence95() const { return getDouble("risk.var_confidence_95", 0.95); }
    double getVaRConfidence99() const { return getDouble("risk.var_confidence_99", 0.99); }
    bool getEnableStressTesting() const { return getBool("risk.enable_stress_testing", true); }
};

#define Config ConfigManager::getInstance()

} // namespace Config