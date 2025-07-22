#include "heston_model.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <fstream>

namespace Heston {

// Thread-local RNG initialization
thread_local std::mt19937 CorrelatedRNG::generator_;
thread_local std::normal_distribution<double> CorrelatedRNG::normal_dist_(0.0, 1.0);
thread_local bool CorrelatedRNG::initialized_ = false;

void CorrelatedRNG::initialize_thread_local() {
    if (!initialized_) {
        generator_.seed(std::random_device{}());
        initialized_ = true;
    }
}

std::pair<double, double> CorrelatedRNG::generate(double correlation) {
    initialize_thread_local();
    
    double z1 = normal_dist_(generator_);
    double z2 = normal_dist_(generator_);
    
    double w1 = z1;
    double w2 = correlation * z1 + std::sqrt(1.0 - correlation * correlation) * z2;
    
    return {w1, w2};
}

void CorrelatedRNG::set_seed(unsigned int seed) {
    generator_.seed(seed);
    initialized_ = true;
}

// Parameters implementation
bool Parameters::is_valid() const noexcept {
    return S0 > 0.0 && v0 > 0.0 && kappa > 0.0 && theta > 0.0 && 
           sigma > 0.0 && rho >= -1.0 && rho <= 1.0 && r >= 0.0 && q >= 0.0;
}

std::string Parameters::validation_error() const noexcept {
    std::ostringstream oss;
    
    if (S0 <= 0.0) oss << "Initial stock price must be positive. ";
    if (v0 <= 0.0) oss << "Initial variance must be positive. ";
    if (kappa <= 0.0) oss << "Mean reversion speed must be positive. ";
    if (theta <= 0.0) oss << "Long-term variance must be positive. ";
    if (sigma <= 0.0) oss << "Vol-of-vol must be positive. ";
    if (rho < -1.0 || rho > 1.0) oss << "Correlation must be between -1 and 1. ";
    if (r < 0.0) oss << "Risk-free rate cannot be negative. ";
    if (q < 0.0) oss << "Dividend yield cannot be negative. ";
    
    return oss.str();
}

void Parameters::print() const {
    std::cout << "Heston Model Parameters:" << std::endl;
    std::cout << "  Initial Stock Price (S0): $" << S0 << std::endl;
    std::cout << "  Initial Variance (v0): " << v0 << std::endl;
    std::cout << "  Mean Reversion Speed (κ): " << kappa << std::endl;
    std::cout << "  Long-term Variance (θ): " << theta << std::endl;
    std::cout << "  Vol-of-Vol (σ): " << sigma << std::endl;
    std::cout << "  Correlation (ρ): " << rho << std::endl;
    std::cout << "  Risk-free Rate (r): " << r << std::endl;
    std::cout << "  Dividend Yield (q): " << q << std::endl;
    std::cout << "  Feller Condition: " << (satisfies_feller_condition() ? "Satisfied" : "Not satisfied") << std::endl;
}// He
stonSimulator implementation
SimulationPath HestonSimulator::simulate_path(double T, int num_steps, bool full_path) const {
    switch (scheme_) {
        case Scheme::EULER:
            return simulate_euler(T, num_steps, full_path);
        case Scheme::MILSTEIN:
            return simulate_milstein(T, num_steps, full_path);
        case Scheme::FULL_TRUNCATION:
            return simulate_full_truncation(T, num_steps, full_path);
        case Scheme::REFLECTION:
            return simulate_reflection(T, num_steps, full_path);
        default:
            return simulate_euler(T, num_steps, full_path);
    }
}

SimulationPath HestonSimulator::simulate_euler(double T, int num_steps, bool full_path) const {
    SimulationPath path;
    if (full_path) {
        path.reserve(num_steps + 1);
    }
    
    const double dt = T / num_steps;
    const double sqrt_dt = std::sqrt(dt);
    
    double S = params_.S0;
    double v = params_.v0;
    
    if (full_path) {
        path.stock_prices.push_back(S);
        path.variances.push_back(v);
        path.volatilities.push_back(std::sqrt(v));
        path.times.push_back(0.0);
    }
    
    for (int i = 0; i < num_steps; ++i) {
        auto [dW1, dW2] = CorrelatedRNG::generate(params_.rho);
        
        // Ensure variance stays positive
        double v_pos = std::max(v, 0.0);
        double sqrt_v = std::sqrt(v_pos);
        
        // Update variance using Euler scheme
        double dv = params_.kappa * (params_.theta - v_pos) * dt + 
                   params_.sigma * sqrt_v * sqrt_dt * dW2;
        v += dv;
        
        // Update stock price
        double dS = (params_.r - params_.q) * S * dt + sqrt_v * S * sqrt_dt * dW1;
        S += dS;
        
        if (full_path) {
            path.stock_prices.push_back(S);
            path.variances.push_back(v);
            path.volatilities.push_back(std::sqrt(std::max(v, 0.0)));
            path.times.push_back((i + 1) * dt);
        }
    }
    
    if (!full_path) {
        path.stock_prices.push_back(S);
        path.variances.push_back(v);
        path.volatilities.push_back(std::sqrt(std::max(v, 0.0)));
        path.times.push_back(T);
    }
    
    return path;
}

SimulationPath HestonSimulator::simulate_milstein(double T, int num_steps, bool full_path) const {
    // Milstein scheme with improved accuracy for variance process
    SimulationPath path;
    if (full_path) {
        path.reserve(num_steps + 1);
    }
    
    const double dt = T / num_steps;
    const double sqrt_dt = std::sqrt(dt);
    
    double S = params_.S0;
    double v = params_.v0;
    
    if (full_path) {
        path.stock_prices.push_back(S);
        path.variances.push_back(v);
        path.volatilities.push_back(std::sqrt(v));
        path.times.push_back(0.0);
    }
    
    for (int i = 0; i < num_steps; ++i) {
        auto [dW1, dW2] = CorrelatedRNG::generate(params_.rho);
        
        double v_pos = std::max(v, 0.0);
        double sqrt_v = std::sqrt(v_pos);
        
        // Milstein correction for variance
        double dv = params_.kappa * (params_.theta - v_pos) * dt + 
                   params_.sigma * sqrt_v * sqrt_dt * dW2 +
                   0.25 * params_.sigma * params_.sigma * (dW2 * dW2 - dt);
        v += dv;
        
        // Stock price update (Euler for simplicity)
        double dS = (params_.r - params_.q) * S * dt + sqrt_v * S * sqrt_dt * dW1;
        S += dS;
        
        if (full_path) {
            path.stock_prices.push_back(S);
            path.variances.push_back(v);
            path.volatilities.push_back(std::sqrt(std::max(v, 0.0)));
            path.times.push_back((i + 1) * dt);
        }
    }
    
    if (!full_path) {
        path.stock_prices.push_back(S);
        path.variances.push_back(v);
        path.volatilities.push_back(std::sqrt(std::max(v, 0.0)));
        path.times.push_back(T);
    }
    
    return path;
}Simula
tionPath HestonSimulator::simulate_full_truncation(double T, int num_steps, bool full_path) const {
    // Full truncation scheme - sets negative variance to zero
    SimulationPath path;
    if (full_path) {
        path.reserve(num_steps + 1);
    }
    
    const double dt = T / num_steps;
    const double sqrt_dt = std::sqrt(dt);
    
    double S = params_.S0;
    double v = params_.v0;
    
    if (full_path) {
        path.stock_prices.push_back(S);
        path.variances.push_back(v);
        path.volatilities.push_back(std::sqrt(v));
        path.times.push_back(0.0);
    }
    
    for (int i = 0; i < num_steps; ++i) {
        auto [dW1, dW2] = CorrelatedRNG::generate(params_.rho);
        
        double sqrt_v = std::sqrt(std::max(v, 0.0));
        
        // Update variance with full truncation
        double dv = params_.kappa * (params_.theta - std::max(v, 0.0)) * dt + 
                   params_.sigma * sqrt_v * sqrt_dt * dW2;
        v = std::max(v + dv, 0.0);  // Full truncation
        
        // Update stock price
        double dS = (params_.r - params_.q) * S * dt + sqrt_v * S * sqrt_dt * dW1;
        S += dS;
        
        if (full_path) {
            path.stock_prices.push_back(S);
            path.variances.push_back(v);
            path.volatilities.push_back(std::sqrt(v));
            path.times.push_back((i + 1) * dt);
        }
    }
    
    if (!full_path) {
        path.stock_prices.push_back(S);
        path.variances.push_back(v);
        path.volatilities.push_back(std::sqrt(v));
        path.times.push_back(T);
    }
    
    return path;
}

SimulationPath HestonSimulator::simulate_reflection(double T, int num_steps, bool full_path) const {
    // Reflection scheme - reflects negative variance to positive
    SimulationPath path;
    if (full_path) {
        path.reserve(num_steps + 1);
    }
    
    const double dt = T / num_steps;
    const double sqrt_dt = std::sqrt(dt);
    
    double S = params_.S0;
    double v = params_.v0;
    
    if (full_path) {
        path.stock_prices.push_back(S);
        path.variances.push_back(v);
        path.volatilities.push_back(std::sqrt(v));
        path.times.push_back(0.0);
    }
    
    for (int i = 0; i < num_steps; ++i) {
        auto [dW1, dW2] = CorrelatedRNG::generate(params_.rho);
        
        double sqrt_v = std::sqrt(std::max(v, 0.0));
        
        // Update variance
        double dv = params_.kappa * (params_.theta - v) * dt + 
                   params_.sigma * sqrt_v * sqrt_dt * dW2;
        v += dv;
        
        // Reflection: if variance becomes negative, reflect it
        if (v < 0.0) {
            v = -v;
        }
        
        // Update stock price
        double dS = (params_.r - params_.q) * S * dt + sqrt_v * S * sqrt_dt * dW1;
        S += dS;
        
        if (full_path) {
            path.stock_prices.push_back(S);
            path.variances.push_back(v);
            path.volatilities.push_back(std::sqrt(v));
            path.times.push_back((i + 1) * dt);
        }
    }
    
    if (!full_path) {
        path.stock_prices.push_back(S);
        path.variances.push_back(v);
        path.volatilities.push_back(std::sqrt(v));
        path.times.push_back(T);
    }
    
    return path;
}