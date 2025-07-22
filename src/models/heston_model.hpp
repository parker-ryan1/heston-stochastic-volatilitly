#pragma once

#include <vector>
#include <memory>
#include <random>
#include <string>
#include <functional>
#include <atomic>
#include <mutex>
#include <complex>
#include "../utils/logger.hpp"
#include "../config/config.hpp"

/**
 * @file heston_model.hpp
 * @brief Heston stochastic volatility model implementation
 * @author Quantitative Finance Team
 * @version 1.0
 * @date 2025
 * 
 * This file contains the implementation of the Heston stochastic volatility model:
 * 
 * Stock price dynamics: dS = (r-q)S dt + √v S dW₁
 * Variance dynamics:    dv = κ(θ-v) dt + σ√v dW₂
 * Correlation:          dW₁ dW₂ = ρ dt
 * 
 * Where:
 * - S: Stock price
 * - v: Instantaneous variance
 * - κ: Mean reversion speed
 * - θ: Long-term variance
 * - σ: Volatility of volatility (vol-of-vol)
 * - ρ: Correlation between stock and volatility
 * - r: Risk-free rate
 * - q: Dividend yield
 */

namespace Heston {

/**
 * @brief Heston model parameters
 */
struct Parameters {
    double S0;      ///< Initial stock price (S > 0)
    double v0;      ///< Initial variance (v > 0)
    double kappa;   ///< Mean reversion speed (κ > 0)
    double theta;   ///< Long-term variance (θ > 0)
    double sigma;   ///< Volatility of volatility (σ > 0)
    double rho;     ///< Correlation (-1 ≤ ρ ≤ 1)
    double r;       ///< Risk-free rate (r ≥ 0)
    double q;       ///< Dividend yield (q ≥ 0)
    
    Parameters(double S0 = 100.0, double v0 = 0.04, double kappa = 2.0,
               double theta = 0.04, double sigma = 0.3, double rho = -0.7,
               double r = 0.05, double q = 0.0)
        : S0(S0), v0(v0), kappa(kappa), theta(theta), sigma(sigma),
          rho(rho), r(r), q(q) {}
    
    /**
     * @brief Validate parameters
     * @return true if all parameters are valid
     */
    bool is_valid() const noexcept;
    
    /**
     * @brief Get validation error message
     * @return Error message if invalid, empty if valid
     */
    std::string validation_error() const noexcept;
    
    /**
     * @brief Check Feller condition for variance process
     * @return true if 2κθ ≥ σ² (ensures variance stays positive)
     */
    bool satisfies_feller_condition() const noexcept {
        return 2.0 * kappa * theta >= sigma * sigma;
    }
    
    /**
     * @brief Print parameters to console
     */
    void print() const;
};

/**
 * @brief Simulation path containing stock prices and variances
 */
struct SimulationPath {
    std::vector<double> stock_prices;   ///< Stock price path
    std::vector<double> variances;      ///< Variance path
    std::vector<double> volatilities;   ///< Volatility path (√variance)
    std::vector<double> times;          ///< Time points
    
    SimulationPath() = default;
    
    /**
     * @brief Reserve memory for path
     * @param size Number of time steps
     */
    void reserve(size_t size) {
        stock_prices.reserve(size);
        variances.reserve(size);
        volatilities.reserve(size);
        times.reserve(size);
    }
    
    /**
     * @brief Get path length
     * @return Number of time points
     */
    size_t size() const { return stock_prices.size(); }
    
    /**
     * @brief Check if path is empty
     * @return true if path has no points
     */
    bool empty() const { return stock_prices.empty(); }
};

/**
 * @brief Pricing result with statistics
 */
struct PricingResult {
    double price;               ///< Option price
    double standard_error;      ///< Monte Carlo standard error
    double confidence_interval; ///< 95% confidence interval
    int simulations_used;       ///< Number of simulations performed
    double execution_time_ms;   ///< Execution time in milliseconds
    bool is_valid;              ///< Whether calculation was successful
    std::string error_message;  ///< Error message if calculation failed
    
    PricingResult() : price(0.0), standard_error(0.0), confidence_interval(0.0),
                     simulations_used(0), execution_time_ms(0.0), is_valid(false) {}
};

/**
 * @brief Thread-safe correlated random number generator
 */
class CorrelatedRNG {
private:
    thread_local static std::mt19937 generator_;
    thread_local static std::normal_distribution<double> normal_dist_;
    thread_local static bool initialized_;
    
    static void initialize_thread_local();

public:
    /**
     * @brief Generate correlated normal random numbers
     * @param correlation Correlation coefficient (-1 ≤ ρ ≤ 1)
     * @return Pair of correlated N(0,1) random variables
     */
    static std::pair<double, double> generate(double correlation);
    
    /**
     * @brief Set random seed for current thread
     * @param seed Random seed
     */
    static void set_seed(unsigned int seed);
};

/**
 * @brief Heston model simulator using various discretization schemes
 */
class HestonSimulator {
public:
    /**
     * @brief Discretization schemes for Heston model
     */
    enum class Scheme {
        EULER,          ///< Euler-Maruyama scheme
        MILSTEIN,       ///< Milstein scheme
        FULL_TRUNCATION, ///< Full truncation scheme
        REFLECTION      ///< Reflection scheme
    };

private:
    Parameters params_;
    Scheme scheme_;
    mutable Utils::Logger logger_;
    
public:
    /**
     * @brief Construct Heston simulator
     * @param params Heston model parameters
     * @param scheme Discretization scheme
     */
    explicit HestonSimulator(const Parameters& params, Scheme scheme = Scheme::EULER)
        : params_(params), scheme_(scheme), logger_("HestonSimulator") {
        if (!params_.is_valid()) {
            throw std::invalid_argument(params_.validation_error());
        }
    }
    
    /**
     * @brief Simulate single path
     * @param T Time to maturity
     * @param num_steps Number of time steps
     * @param full_path Whether to return full path or just endpoints
     * @return Simulation path
     */
    SimulationPath simulate_path(double T, int num_steps, bool full_path = true) const;
    
    /**
     * @brief Price European option using Monte Carlo
     * @param K Strike price
     * @param T Time to maturity
     * @param is_call true for call, false for put
     * @param num_simulations Number of Monte Carlo simulations
     * @param num_steps Number of time steps per simulation
     * @return Pricing result
     */
    PricingResult price_european_option(double K, double T, bool is_call,
                                       int num_simulations = 0, int num_steps = 0) const;
    
    /**
     * @brief Price variance swap
     * @param T Time to maturity
     * @param num_simulations Number of Monte Carlo simulations
     * @param num_steps Number of time steps per simulation
     * @return Fair variance strike
     */
    double price_variance_swap(double T, int num_simulations = 0, int num_steps = 0) const;
    
    /**
     * @brief Generate volatility surface
     * @param strikes Vector of strike prices
     * @param maturities Vector of maturities
     * @param num_simulations Number of simulations per point
     * @return Implied volatility surface
     */
    std::vector<std::vector<double>> generate_volatility_surface(
        const std::vector<double>& strikes,
        const std::vector<double>& maturities,
        int num_simulations = 0) const;
    
    /**
     * @brief Get model parameters
     * @return Heston parameters
     */
    const Parameters& get_parameters() const { return params_; }
    
    /**
     * @brief Get discretization scheme
     * @return Current scheme
     */
    Scheme get_scheme() const { return scheme_; }
    
    /**
     * @brief Set discretization scheme
     * @param scheme New discretization scheme
     */
    void set_scheme(Scheme scheme) { scheme_ = scheme; }

private:
    /**
     * @brief Simulate using Euler-Maruyama scheme
     */
    SimulationPath simulate_euler(double T, int num_steps, bool full_path) const;
    
    /**
     * @brief Simulate using Milstein scheme
     */
    SimulationPath simulate_milstein(double T, int num_steps, bool full_path) const;
    
    /**
     * @brief Simulate using full truncation scheme
     */
    SimulationPath simulate_full_truncation(double T, int num_steps, bool full_path) const;
    
    /**
     * @brief Simulate using reflection scheme
     */
    SimulationPath simulate_reflection(double T, int num_steps, bool full_path) const;
    
    /**
     * @brief Calculate Black-Scholes implied volatility
     */
    double calculate_implied_volatility(double market_price, double S, double K, 
                                       double T, double r, double q, bool is_call) const;
};

/**
 * @brief VIX options pricer using Heston model
 */
class VIXOptionsPricer {
private:
    Parameters params_;
    HestonSimulator simulator_;
    mutable Utils::Logger logger_;
    
public:
    /**
     * @brief Construct VIX options pricer
     * @param params Heston model parameters
     */
    explicit VIXOptionsPricer(const Parameters& params)
        : params_(params), simulator_(params), logger_("VIXOptionsPricer") {}
    
    /**
     * @brief Calculate VIX level from variance
     * @param variance Instantaneous variance
     * @return VIX level (percentage)
     */
    static double calculate_vix(double variance) {
        return std::sqrt(variance) * 100.0;
    }
    
    /**
     * @brief Price VIX option
     * @param K Strike level (VIX points)
     * @param T Time to maturity
     * @param is_call true for call, false for put
     * @param num_simulations Number of Monte Carlo simulations
     * @return Pricing result
     */
    PricingResult price_vix_option(double K, double T, bool is_call, int num_simulations = 0) const;
    
    /**
     * @brief Calculate VIX term structure
     * @param maturities Vector of maturities
     * @param num_simulations Number of simulations per maturity
     * @return VIX levels for each maturity
     */
    std::vector<double> calculate_vix_term_structure(const std::vector<double>& maturities,
                                                    int num_simulations = 0) const;
};

/**
 * @brief Risk analyzer for Heston model
 */
class RiskAnalyzer {
public:
    /**
     * @brief Risk metrics structure
     */
    struct RiskMetrics {
        double var_95;                  ///< 95% Value at Risk
        double var_99;                  ///< 99% Value at Risk
        double expected_shortfall_95;   ///< Expected Shortfall (CVaR)
        double max_drawdown;            ///< Maximum drawdown
        double volatility_of_volatility; ///< Vol-of-vol estimate
        double correlation_stock_vol;   ///< Stock-volatility correlation
        double skewness;                ///< Return skewness
        double kurtosis;                ///< Return kurtosis
        
        RiskMetrics() : var_95(0.0), var_99(0.0), expected_shortfall_95(0.0),
                       max_drawdown(0.0), volatility_of_volatility(0.0),
                       correlation_stock_vol(0.0), skewness(0.0), kurtosis(0.0) {}
    };

private:
    Parameters params_;
    HestonSimulator simulator_;
    mutable Utils::Logger logger_;
    
public:
    /**
     * @brief Construct risk analyzer
     * @param params Heston model parameters
     */
    explicit RiskAnalyzer(const Parameters& params)
        : params_(params), simulator_(params), logger_("RiskAnalyzer") {}
    
    /**
     * @brief Calculate comprehensive risk metrics
     * @param T Analysis period
     * @param num_simulations Number of Monte Carlo simulations
     * @return Risk metrics
     */
    RiskMetrics calculate_risk_metrics(double T, int num_simulations = 0) const;
    
    /**
     * @brief Perform stress testing
     * @param stress_scenarios Vector of parameter stress scenarios
     * @param base_option_price Base case option price
     * @param K Strike price
     * @param T Time to maturity
     * @param is_call Option type
     * @return Vector of stressed option prices
     */
    std::vector<double> stress_test(const std::vector<Parameters>& stress_scenarios,
                                   double base_option_price, double K, double T, bool is_call) const;
    
    /**
     * @brief Calculate Greeks using finite differences
     * @param K Strike price
     * @param T Time to maturity
     * @param is_call Option type
     * @param num_simulations Number of simulations per Greek
     * @return Map of Greek names to values
     */
    std::map<std::string, double> calculate_greeks(double K, double T, bool is_call,
                                                  int num_simulations = 0) const;
};

/**
 * @brief Data exporter for Heston model results
 */
class DataExporter {
public:
    /**
     * @brief Export simulation path to CSV
     * @param path Simulation path
     * @param filename Output filename
     */
    static void export_path(const SimulationPath& path, const std::string& filename);
    
    /**
     * @brief Export volatility surface to CSV
     * @param surface Volatility surface
     * @param strikes Strike prices
     * @param maturities Maturities
     * @param filename Output filename
     */
    static void export_volatility_surface(const std::vector<std::vector<double>>& surface,
                                         const std::vector<double>& strikes,
                                         const std::vector<double>& maturities,
                                         const std::string& filename);
    
    /**
     * @brief Export risk metrics to JSON
     * @param metrics Risk metrics
     * @param filename Output filename
     */
    static void export_risk_metrics(const RiskAnalyzer::RiskMetrics& metrics,
                                   const std::string& filename);
    
    /**
     * @brief Export VIX term structure to CSV
     * @param maturities Maturities
     * @param vix_levels VIX levels
     * @param filename Output filename
     */
    static void export_vix_term_structure(const std::vector<double>& maturities,
                                         const std::vector<double>& vix_levels,
                                         const std::string& filename);
};

/**
 * @brief Utility functions for Heston model
 */
namespace Utils {
    /**
     * @brief Convert discretization scheme to string
     * @param scheme Discretization scheme
     * @return String representation
     */
    std::string scheme_to_string(HestonSimulator::Scheme scheme);
    
    /**
     * @brief Validate Heston parameters for numerical stability
     * @param params Heston parameters
     * @return Vector of warning messages
     */
    std::vector<std::string> validate_numerical_stability(const Parameters& params);
    
    /**
     * @brief Calculate theoretical moments of Heston model
     * @param params Heston parameters
     * @param T Time horizon
     * @return Map of moment names to values
     */
    std::map<std::string, double> calculate_theoretical_moments(const Parameters& params, double T);
    
    /**
     * @brief Estimate parameters from market data (simplified)
     * @param market_prices Vector of option prices
     * @param strikes Vector of strikes
     * @param maturities Vector of maturities
     * @param S0 Current stock price
     * @param r Risk-free rate
     * @return Estimated Heston parameters
     */
    Parameters estimate_parameters(const std::vector<double>& market_prices,
                                  const std::vector<double>& strikes,
                                  const std::vector<double>& maturities,
                                  double S0, double r);
}

} // namespace Heston