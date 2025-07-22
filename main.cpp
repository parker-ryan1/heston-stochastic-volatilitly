#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <string>
#include <map>
#include <memory>
#include <fstream>
#include <complex>

// Mathematical constants
const double PI = 3.14159265358979323846;
const double SQRT_2PI = std::sqrt(2.0 * PI);

// Complex number utilities for characteristic functions
using Complex = std::complex<double>;

// Normal distribution functions
double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double norm_pdf(double x) {
    return std::exp(-0.5 * x * x) / SQRT_2PI;
}

// Heston Model Parameters
struct HestonParams {
    double S0;      // Initial stock price
    double v0;      // Initial variance
    double kappa;   // Mean reversion speed
    double theta;   // Long-term variance
    double sigma;   // Volatility of volatility (vol-of-vol)
    double rho;     // Correlation between stock and volatility
    double r;       // Risk-free rate
    double q;       // Dividend yield
    
    HestonParams(double S0 = 100.0, double v0 = 0.04, double kappa = 2.0, 
                 double theta = 0.04, double sigma = 0.3, double rho = -0.7,
                 double r = 0.05, double q = 0.0)
        : S0(S0), v0(v0), kappa(kappa), theta(theta), sigma(sigma), 
          rho(rho), r(r), q(q) {}
    
    void print() const {
        std::cout << "Heston Model Parameters:" << std::endl;
        std::cout << "  Initial Stock Price (S0): $" << S0 << std::endl;
        std::cout << "  Initial Variance (v0): " << v0 << std::endl;
        std::cout << "  Mean Reversion Speed (κ): " << kappa << std::endl;
        std::cout << "  Long-term Variance (θ): " << theta << std::endl;
        std::cout << "  Vol-of-Vol (σ): " << sigma << std::endl;
        std::cout << "  Correlation (ρ): " << rho << std::endl;
        std::cout << "  Risk-free Rate (r): " << r << std::endl;
        std::cout << "  Dividend Yield (q): " << q << std::endl;
    }
    
    bool isValid() const {
        return (S0 > 0 && v0 > 0 && kappa > 0 && theta > 0 && sigma > 0 &&
                rho >= -1.0 && rho <= 1.0 && r >= 0);
    }
};

// Random number generator for correlated Brownian motions
class CorrelatedBrownianMotion {
private:
    std::mt19937 gen;
    std::normal_distribution<double> normal_dist;
    
public:
    CorrelatedBrownianMotion() : gen(std::random_device{}()), normal_dist(0.0, 1.0) {}
    
    void seed(unsigned int s) { gen.seed(s); }
    
    std::pair<double, double> generate(double rho) {
        double z1 = normal_dist(gen);
        double z2 = normal_dist(gen);
        
        // Generate correlated random variables
        double w1 = z1;
        double w2 = rho * z1 + std::sqrt(1.0 - rho * rho) * z2;
        
        return {w1, w2};
    }
};

// Heston Model Simulator
class HestonSimulator {
private:
    HestonParams params;
    CorrelatedBrownianMotion rng;
    
public:
    HestonSimulator(const HestonParams& p) : params(p) {}
    
    // Euler-Maruyama discretization scheme
    struct SimulationPath {
        std::vector<double> stock_prices;
        std::vector<double> variances;
        std::vector<double> volatilities;
        std::vector<double> times;
    };
    
    SimulationPath simulate(double T, int num_steps, bool full_path = true) {
        double dt = T / num_steps;
        double sqrt_dt = std::sqrt(dt);
        
        SimulationPath path;
        if (full_path) {
            path.stock_prices.reserve(num_steps + 1);
            path.variances.reserve(num_steps + 1);
            path.volatilities.reserve(num_steps + 1);
            path.times.reserve(num_steps + 1);
        }
        
        double S = params.S0;
        double v = params.v0;
        
        if (full_path) {
            path.stock_prices.push_back(S);
            path.variances.push_back(v);
            path.volatilities.push_back(std::sqrt(v));
            path.times.push_back(0.0);
        }
        
        for (int i = 0; i < num_steps; ++i) {
            auto [dW1, dW2] = rng.generate(params.rho);
            
            // Ensure variance stays positive (Feller condition)
            double v_pos = std::max(v, 0.0);
            double sqrt_v = std::sqrt(v_pos);
            
            // Update variance using Euler scheme
            double dv = params.kappa * (params.theta - v_pos) * dt + 
                       params.sigma * sqrt_v * sqrt_dt * dW2;
            v += dv;
            
            // Update stock price
            double dS = (params.r - params.q) * S * dt + sqrt_v * S * sqrt_dt * dW1;
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
    
    // Monte Carlo option pricing
    double priceEuropeanOption(double K, double T, bool is_call, int num_sims = 100000, int num_steps = 252) {
        double total_payoff = 0.0;
        
        for (int sim = 0; sim < num_sims; ++sim) {
            auto path = simulate(T, num_steps, false);
            double S_T = path.stock_prices.back();
            
            double payoff = is_call ? std::max(S_T - K, 0.0) : std::max(K - S_T, 0.0);
            total_payoff += payoff;
        }
        
        return std::exp(-params.r * T) * total_payoff / num_sims;
    }
    
    // Variance swap pricing
    double priceVarianceSwap(double T, int num_sims = 100000, int num_steps = 252) {
        double total_realized_var = 0.0;
        
        for (int sim = 0; sim < num_sims; ++sim) {
            auto path = simulate(T, num_steps, true);
            
            // Calculate realized variance
            double realized_var = 0.0;
            for (size_t i = 1; i < path.stock_prices.size(); ++i) {
                double log_return = std::log(path.stock_prices[i] / path.stock_prices[i-1]);
                realized_var += log_return * log_return;
            }
            realized_var *= (252.0 / num_steps);  // Annualize
            
            total_realized_var += realized_var;
        }
        
        return total_realized_var / num_sims;
    }
    
    // Volatility surface generation
    std::vector<std::vector<double>> generateVolatilitySurface(
        const std::vector<double>& strikes, 
        const std::vector<double>& maturities,
        int num_sims = 50000) {
        
        std::vector<std::vector<double>> vol_surface(maturities.size(), std::vector<double>(strikes.size()));
        
        for (size_t i = 0; i < maturities.size(); ++i) {
            for (size_t j = 0; j < strikes.size(); ++j) {
                double call_price = priceEuropeanOption(strikes[j], maturities[i], true, num_sims);
                
                // Convert to implied volatility using Black-Scholes approximation
                double implied_vol = blackScholesImpliedVol(call_price, params.S0, strikes[j], 
                                                          maturities[i], params.r, params.q);
                vol_surface[i][j] = implied_vol;
            }
        }
        
        return vol_surface;
    }
    
private:
    // Black-Scholes implied volatility (simplified Newton-Raphson)
    double blackScholesImpliedVol(double market_price, double S, double K, double T, double r, double q) {
        double vol = 0.2;  // Initial guess
        const double tolerance = 1e-6;
        const int max_iterations = 100;
        
        for (int i = 0; i < max_iterations; ++i) {
            double d1 = (std::log(S/K) + (r - q + 0.5*vol*vol)*T) / (vol*std::sqrt(T));
            double d2 = d1 - vol*std::sqrt(T);
            
            double bs_price = S*std::exp(-q*T)*norm_cdf(d1) - K*std::exp(-r*T)*norm_cdf(d2);
            double vega = S*std::exp(-q*T)*norm_pdf(d1)*std::sqrt(T);
            
            if (std::abs(vega) < 1e-10) break;
            
            double price_diff = bs_price - market_price;
            if (std::abs(price_diff) < tolerance) break;
            
            vol -= price_diff / vega;
            vol = std::max(vol, 0.001);  // Ensure positive volatility
        }
        
        return vol;
    }
};

// VIX Options Pricer
class VIXOptionsPricer {
private:
    HestonParams params;
    HestonSimulator simulator;
    
public:
    VIXOptionsPricer(const HestonParams& p) : params(p), simulator(p) {}
    
    // VIX calculation from variance
    double calculateVIX(double variance) {
        return std::sqrt(variance) * 100.0;  // Convert to percentage
    }
    
    // Price VIX options using Monte Carlo
    double priceVIXOption(double K, double T, bool is_call, int num_sims = 100000) {
        double total_payoff = 0.0;
        
        for (int sim = 0; sim < num_sims; ++sim) {
            // Simulate path and calculate terminal variance
            auto path = simulator.simulate(T, 252, false);
            double terminal_variance = path.variances.back();
            double vix_level = calculateVIX(terminal_variance);
            
            double payoff = is_call ? std::max(vix_level - K, 0.0) : std::max(K - vix_level, 0.0);
            total_payoff += payoff;
        }
        
        return std::exp(-params.r * T) * total_payoff / num_sims;
    }
};

// Risk Management and Analytics
class HestonRiskAnalyzer {
private:
    HestonParams params;
    HestonSimulator simulator;
    
public:
    HestonRiskAnalyzer(const HestonParams& p) : params(p), simulator(p) {}
    
    struct RiskMetrics {
        double var_95;      // 95% Value at Risk
        double var_99;      // 99% Value at Risk
        double expected_shortfall_95;  // Expected Shortfall (CVaR)
        double max_drawdown;
        double volatility_of_volatility;
        double correlation_stock_vol;
    };
    
    RiskMetrics calculateRiskMetrics(double T, int num_sims = 10000) {
        std::vector<double> returns;
        std::vector<double> max_drawdowns;
        std::vector<double> vol_changes;
        std::vector<double> stock_returns;
        
        returns.reserve(num_sims);
        max_drawdowns.reserve(num_sims);
        
        for (int sim = 0; sim < num_sims; ++sim) {
            auto path = simulator.simulate(T, 252, true);
            
            // Calculate return
            double total_return = (path.stock_prices.back() / path.stock_prices.front()) - 1.0;
            returns.push_back(total_return);
            
            // Calculate maximum drawdown
            double peak = path.stock_prices[0];
            double max_dd = 0.0;
            
            for (size_t i = 1; i < path.stock_prices.size(); ++i) {
                peak = std::max(peak, path.stock_prices[i]);
                double drawdown = (peak - path.stock_prices[i]) / peak;
                max_dd = std::max(max_dd, drawdown);
                
                // Collect data for correlation analysis
                if (i > 1) {
                    double stock_ret = std::log(path.stock_prices[i] / path.stock_prices[i-1]);
                    double vol_change = path.volatilities[i] - path.volatilities[i-1];
                    stock_returns.push_back(stock_ret);
                    vol_changes.push_back(vol_change);
                }
            }
            
            max_drawdowns.push_back(max_dd);
        }
        
        // Sort returns for VaR calculation
        std::sort(returns.begin(), returns.end());
        
        RiskMetrics metrics;
        metrics.var_95 = returns[static_cast<int>(0.05 * num_sims)];
        metrics.var_99 = returns[static_cast<int>(0.01 * num_sims)];
        
        // Expected Shortfall (average of worst 5%)
        double sum_tail = 0.0;
        int tail_count = static_cast<int>(0.05 * num_sims);
        for (int i = 0; i < tail_count; ++i) {
            sum_tail += returns[i];
        }
        metrics.expected_shortfall_95 = sum_tail / tail_count;
        
        // Maximum drawdown
        metrics.max_drawdown = *std::max_element(max_drawdowns.begin(), max_drawdowns.end());
        
        // Volatility of volatility
        double vol_mean = 0.0;
        for (double vol : vol_changes) vol_mean += vol;
        vol_mean /= vol_changes.size();
        
        double vol_var = 0.0;
        for (double vol : vol_changes) {
            vol_var += (vol - vol_mean) * (vol - vol_mean);
        }
        metrics.volatility_of_volatility = std::sqrt(vol_var / (vol_changes.size() - 1));
        
        // Correlation between stock returns and volatility changes
        double mean_stock_ret = 0.0;
        for (double ret : stock_returns) mean_stock_ret += ret;
        mean_stock_ret /= stock_returns.size();
        
        double covariance = 0.0;
        double var_stock = 0.0;
        double var_vol = 0.0;
        
        for (size_t i = 0; i < stock_returns.size(); ++i) {
            double stock_dev = stock_returns[i] - mean_stock_ret;
            double vol_dev = vol_changes[i] - vol_mean;
            
            covariance += stock_dev * vol_dev;
            var_stock += stock_dev * stock_dev;
            var_vol += vol_dev * vol_dev;
        }
        
        metrics.correlation_stock_vol = covariance / std::sqrt(var_stock * var_vol);
        
        return metrics;
    }
};

// Data Export Utilities
class DataExporter {
public:
    static void exportPath(const HestonSimulator::SimulationPath& path, const std::string& filename) {
        std::ofstream file(filename);
        file << "Time,StockPrice,Variance,Volatility\n";
        
        for (size_t i = 0; i < path.times.size(); ++i) {
            file << std::fixed << std::setprecision(6)
                 << path.times[i] << ","
                 << path.stock_prices[i] << ","
                 << path.variances[i] << ","
                 << path.volatilities[i] << "\n";
        }
        
        file.close();
        std::cout << "Path data exported to " << filename << std::endl;
    }
    
    static void exportVolatilitySurface(const std::vector<std::vector<double>>& surface,
                                       const std::vector<double>& strikes,
                                       const std::vector<double>& maturities,
                                       const std::string& filename) {
        std::ofstream file(filename);
        
        // Header
        file << "Strike/Maturity";
        for (double T : maturities) {
            file << "," << T;
        }
        file << "\n";
        
        // Data
        for (size_t j = 0; j < strikes.size(); ++j) {
            file << strikes[j];
            for (size_t i = 0; i < maturities.size(); ++i) {
                file << "," << std::fixed << std::setprecision(4) << surface[i][j];
            }
            file << "\n";
        }
        
        file.close();
        std::cout << "Volatility surface exported to " << filename << std::endl;
    }
};

// Main Application Class
class HestonModelApplication {
private:
    HestonParams params;
    std::unique_ptr<HestonSimulator> simulator;
    std::unique_ptr<VIXOptionsPricer> vix_pricer;
    std::unique_ptr<HestonRiskAnalyzer> risk_analyzer;
    
    void printHeader() {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "           HESTON STOCHASTIC VOLATILITY MODEL" << std::endl;
        std::cout << "                    C++ Implementation" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
    }
    
    void printMenu() {
        std::cout << "\nSelect Operation:" << std::endl;
        std::cout << "1. Configure Heston Parameters" << std::endl;
        std::cout << "2. Simulate Single Path" << std::endl;
        std::cout << "3. Price European Options" << std::endl;
        std::cout << "4. Price VIX Options" << std::endl;
        std::cout << "5. Price Variance Swaps" << std::endl;
        std::cout << "6. Generate Volatility Surface" << std::endl;
        std::cout << "7. Risk Analysis" << std::endl;
        std::cout << "8. Export Simulation Data" << std::endl;
        std::cout << "9. View Current Parameters" << std::endl;
        std::cout << "0. Exit" << std::endl;
        std::cout << "\nChoice: ";
    }
    
    double getInput(const std::string& prompt) {
        double value;
        std::cout << prompt;
        std::cin >> value;
        return value;
    }
    
    bool getBoolInput(const std::string& prompt) {
        char choice;
        std::cout << prompt << " (y/n): ";
        std::cin >> choice;
        return (choice == 'y' || choice == 'Y');
    }
    
    void updateComponents() {
        simulator = std::make_unique<HestonSimulator>(params);
        vix_pricer = std::make_unique<VIXOptionsPricer>(params);
        risk_analyzer = std::make_unique<HestonRiskAnalyzer>(params);
    }

public:
    HestonModelApplication() : params() {
        updateComponents();
    }
    
    void run() {
        printHeader();
        
        int choice;
        do {
            printMenu();
            std::cin >> choice;
            
            switch (choice) {
                case 1: configureParameters(); break;
                case 2: simulatePath(); break;
                case 3: priceEuropeanOptions(); break;
                case 4: priceVIXOptions(); break;
                case 5: priceVarianceSwaps(); break;
                case 6: generateVolatilitySurface(); break;
                case 7: performRiskAnalysis(); break;
                case 8: exportData(); break;
                case 9: params.print(); break;
                case 0: std::cout << "Goodbye!" << std::endl; break;
                default: std::cout << "Invalid choice!" << std::endl; break;
            }
        } while (choice != 0);
    }
    
private:
    void configureParameters() {
        std::cout << "\n--- Configure Heston Parameters ---" << std::endl;
        
        params.S0 = getInput("Initial Stock Price (S0): $");
        params.v0 = getInput("Initial Variance (v0): ");
        params.kappa = getInput("Mean Reversion Speed (κ): ");
        params.theta = getInput("Long-term Variance (θ): ");
        params.sigma = getInput("Vol-of-Vol (σ): ");
        params.rho = getInput("Correlation (ρ, -1 to 1): ");
        params.r = getInput("Risk-free Rate (r): ");
        params.q = getInput("Dividend Yield (q): ");
        
        if (params.isValid()) {
            updateComponents();
            std::cout << "Parameters updated successfully!" << std::endl;
        } else {
            std::cout << "Invalid parameters! Please check your inputs." << std::endl;
        }
    }
    
    void simulatePath() {
        std::cout << "\n--- Simulate Heston Path ---" << std::endl;
        
        double T = getInput("Time to Maturity (years): ");
        int steps = static_cast<int>(getInput("Number of Steps: "));
        
        auto path = simulator->simulate(T, steps, true);
        
        std::cout << "\nPath Summary:" << std::endl;
        std::cout << "Initial Stock Price: $" << std::fixed << std::setprecision(2) << path.stock_prices.front() << std::endl;
        std::cout << "Final Stock Price: $" << path.stock_prices.back() << std::endl;
        std::cout << "Initial Volatility: " << std::setprecision(4) << path.volatilities.front() * 100 << "%" << std::endl;
        std::cout << "Final Volatility: " << path.volatilities.back() * 100 << "%" << std::endl;
        
        double total_return = (path.stock_prices.back() / path.stock_prices.front()) - 1.0;
        std::cout << "Total Return: " << std::setprecision(2) << total_return * 100 << "%" << std::endl;
        
        // Calculate some path statistics
        double max_price = *std::max_element(path.stock_prices.begin(), path.stock_prices.end());
        double min_price = *std::min_element(path.stock_prices.begin(), path.stock_prices.end());
        double max_vol = *std::max_element(path.volatilities.begin(), path.volatilities.end());
        double min_vol = *std::min_element(path.volatilities.begin(), path.volatilities.end());
        
        std::cout << "Price Range: $" << min_price << " - $" << max_price << std::endl;
        std::cout << "Volatility Range: " << min_vol * 100 << "% - " << max_vol * 100 << "%" << std::endl;
    }
    
    void priceEuropeanOptions() {
        std::cout << "\n--- Price European Options ---" << std::endl;
        
        double K = getInput("Strike Price: $");
        double T = getInput("Time to Maturity (years): ");
        bool is_call = getBoolInput("Call option?");
        int num_sims = static_cast<int>(getInput("Number of Simulations (default 100000): "));
        
        if (num_sims <= 0) num_sims = 100000;
        
        std::cout << "\nPricing..." << std::endl;
        double price = simulator->priceEuropeanOption(K, T, is_call, num_sims);
        
        std::cout << "\nOption Price: $" << std::fixed << std::setprecision(4) << price << std::endl;
        
        // Calculate some additional metrics
        double moneyness = params.S0 / K;
        std::cout << "Moneyness (S/K): " << std::setprecision(3) << moneyness << std::endl;
        std::cout << "Option Type: " << (is_call ? "Call" : "Put") << std::endl;
        std::cout << "Simulations Used: " << num_sims << std::endl;
    }
    
    void priceVIXOptions() {
        std::cout << "\n--- Price VIX Options ---" << std::endl;
        
        double K = getInput("Strike Level (VIX points): ");
        double T = getInput("Time to Maturity (years): ");
        bool is_call = getBoolInput("Call option?");
        int num_sims = static_cast<int>(getInput("Number of Simulations (default 100000): "));
        
        if (num_sims <= 0) num_sims = 100000;
        
        std::cout << "\nPricing VIX option..." << std::endl;
        double price = vix_pricer->priceVIXOption(K, T, is_call, num_sims);
        
        std::cout << "\nVIX Option Price: $" << std::fixed << std::setprecision(4) << price << std::endl;
        
        // Show current implied VIX level
        double current_vix = vix_pricer->calculateVIX(params.v0);
        std::cout << "Current VIX Level: " << std::setprecision(2) << current_vix << std::endl;
        std::cout << "Strike Level: " << K << std::endl;
        std::cout << "Moneyness: " << std::setprecision(3) << current_vix / K << std::endl;
    }
    
    void priceVarianceSwaps() {
        std::cout << "\n--- Price Variance Swaps ---" << std::endl;
        
        double T = getInput("Time to Maturity (years): ");
        int num_sims = static_cast<int>(getInput("Number of Simulations (default 100000): "));
        
        if (num_sims <= 0) num_sims = 100000;
        
        std::cout << "\nPricing variance swap..." << std::endl;
        double fair_variance = simulator->priceVarianceSwap(T, num_sims);
        
        std::cout << "\nVariance Swap Results:" << std::endl;
        std::cout << "Fair Variance Strike: " << std::fixed << std::setprecision(6) << fair_variance << std::endl;
        std::cout << "Fair Volatility Strike: " << std::setprecision(4) << std::sqrt(fair_variance) * 100 << "%" << std::endl;
        std::cout << "Current Variance: " << std::setprecision(6) << params.v0 << std::endl;
        std::cout << "Current Volatility: " << std::setprecision(4) << std::sqrt(params.v0) * 100 << "%" << std::endl;
        
        double variance_premium = fair_variance - params.v0;
        std::cout << "Variance Premium: " << std::setprecision(6) << variance_premium << std::endl;
    }
    
    void generateVolatilitySurface() {
        std::cout << "\n--- Generate Volatility Surface ---" << std::endl;
        
        // Define strikes and maturities
        std::vector<double> strikes = {80, 90, 95, 100, 105, 110, 120};
        std::vector<double> maturities = {0.083, 0.25, 0.5, 1.0, 2.0};  // 1M, 3M, 6M, 1Y, 2Y
        
        int num_sims = static_cast<int>(getInput("Simulations per point (default 50000): "));
        if (num_sims <= 0) num_sims = 50000;
        
        std::cout << "\nGenerating volatility surface..." << std::endl;
        std::cout << "This may take a few minutes..." << std::endl;
        
        auto vol_surface = simulator->generateVolatilitySurface(strikes, maturities, num_sims);
        
        // Display the surface
        std::cout << "\nImplied Volatility Surface:" << std::endl;
        std::cout << std::setw(8) << "Strike";
        for (double T : maturities) {
            std::cout << std::setw(10) << (T < 1 ? std::to_string(static_cast<int>(T * 12)) + "M" : 
                                                   std::to_string(static_cast<int>(T)) + "Y");
        }
        std::cout << std::endl;
        
        for (size_t j = 0; j < strikes.size(); ++j) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(0) << strikes[j];
            for (size_t i = 0; i < maturities.size(); ++i) {
                std::cout << std::setw(10) << std::setprecision(2) << vol_surface[i][j] * 100 << "%";
            }
            std::cout << std::endl;
        }
        
        if (getBoolInput("Export to CSV?")) {
            DataExporter::exportVolatilitySurface(vol_surface, strikes, maturities, "volatility_surface.csv");
        }
    }
    
    void performRiskAnalysis() {
        std::cout << "\n--- Risk Analysis ---" << std::endl;
        
        double T = getInput("Analysis Period (years): ");
        int num_sims = static_cast<int>(getInput("Number of Simulations (default 10000): "));
        
        if (num_sims <= 0) num_sims = 10000;
        
        std::cout << "\nPerforming risk analysis..." << std::endl;
        auto metrics = risk_analyzer->calculateRiskMetrics(T, num_sims);
        
        std::cout << "\nRisk Metrics:" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        std::cout << "95% Value at Risk: " << std::fixed << std::setprecision(2) << metrics.var_95 * 100 << "%" << std::endl;
        std::cout << "99% Value at Risk: " << metrics.var_99 * 100 << "%" << std::endl;
        std::cout << "Expected Shortfall (95%): " << metrics.expected_shortfall_95 * 100 << "%" << std::endl;
        std::cout << "Maximum Drawdown: " << metrics.max_drawdown * 100 << "%" << std::endl;
        std::cout << "Volatility of Volatility: " << std::setprecision(4) << metrics.volatility_of_volatility << std::endl;
        std::cout << "Stock-Vol Correlation: " << std::setprecision(3) << metrics.correlation_stock_vol << std::endl;
        
        // Risk interpretation
        std::cout << "\nRisk Interpretation:" << std::endl;
        if (metrics.var_95 < -0.20) {
            std::cout << "⚠️  High risk profile detected" << std::endl;
        } else if (metrics.var_95 < -0.10) {
            std::cout << "⚡ Moderate risk profile" << std::endl;
        } else {
            std::cout << "✅ Conservative risk profile" << std::endl;
        }
        
        if (std::abs(metrics.correlation_stock_vol) > 0.5) {
            std::cout << "📊 Strong stock-volatility correlation detected" << std::endl;
        }
    }
    
    void exportData() {
        std::cout << "\n--- Export Simulation Data ---" << std::endl;
        
        double T = getInput("Time to Maturity (years): ");
        int steps = static_cast<int>(getInput("Number of Steps: "));
        
        auto path = simulator->simulate(T, steps, true);
        
        std::string filename;
        std::cout << "Enter filename (without extension): ";
        std::cin >> filename;
        filename += ".csv";
        
        DataExporter::exportPath(path, filename);
    }
};

int main() {
    try {
        HestonModelApplication app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}