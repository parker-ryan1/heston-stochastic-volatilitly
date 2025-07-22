#include "test_framework.hpp"
#include "../src/models/heston_model.hpp"
#include <cmath>
#include <limits>

using namespace Heston;
using namespace Testing;

/**
 * @file test_heston_model.cpp
 * @brief Comprehensive unit tests for Heston stochastic volatility model
 * 
 * Test Coverage:
 * - Parameter validation
 * - Path simulation with different schemes
 * - European option pricing
 * - VIX options pricing
 * - Variance swap pricing
 * - Risk analytics
 * - Performance benchmarks
 * - Thread safety
 */

class HestonModelTestFixture : public TestFixture {
protected:
    Parameters standard_params;
    Parameters high_vol_params;
    Parameters low_correlation_params;
    
    void setUp() override {
        // Standard Heston parameters
        standard_params = Parameters(100.0, 0.04, 2.0, 0.04, 0.3, -0.7, 0.05, 0.0);
        
        // High volatility parameters
        high_vol_params = Parameters(100.0, 0.09, 1.5, 0.09, 0.5, -0.8, 0.05, 0.0);
        
        // Low correlation parameters
        low_correlation_params = Parameters(100.0, 0.04, 2.0, 0.04, 0.3, -0.1, 0.05, 0.0);
    }
};

// Test suite for parameter validation
TEST_SUITE(HestonParameters) {
    auto suite = std::make_unique<TestSuite>("HestonParameters");
    
    // Test valid parameters
    suite->addTest("ValidParameters", []() {
        ASSERT_NO_THROW(Parameters(100.0, 0.04, 2.0, 0.04, 0.3, -0.7, 0.05, 0.0));
        
        Parameters params(100.0, 0.04, 2.0, 0.04, 0.3, -0.7, 0.05, 0.0);
        ASSERT_TRUE(params.is_valid());
        ASSERT_TRUE(params.validation_error().empty());
    });
    
    // Test Feller condition
    suite->addTest("FellerCondition", []() {
        // Parameters that satisfy Feller condition: 2κθ ≥ σ²
        Parameters good_params(100.0, 0.04, 2.0, 0.04, 0.3, -0.7, 0.05, 0.0);
        ASSERT_TRUE(good_params.satisfies_feller_condition());
        
        // Parameters that violate Feller condition
        Parameters bad_params(100.0, 0.04, 1.0, 0.04, 1.0, -0.7, 0.05, 0.0);
        ASSERT_FALSE(bad_params.satisfies_feller_condition());
    });
    
    // Test invalid parameters
    suite->addTest("InvalidParameters", []() {
        // Invalid initial stock price
        Parameters params1(-100.0, 0.04, 2.0, 0.04, 0.3, -0.7, 0.05, 0.0);
        ASSERT_FALSE(params1.is_valid());
        
        // Invalid correlation
        Parameters params2(100.0, 0.04, 2.0, 0.04, 0.3, 1.5, 0.05, 0.0);
        ASSERT_FALSE(params2.is_valid());
        
        // Invalid variance
        Parameters params3(100.0, -0.04, 2.0, 0.04, 0.3, -0.7, 0.05, 0.0);
        ASSERT_FALSE(params3.is_valid());
    });
    
    TestRegistry::getInstance().registerSuite(std::move(suite));
}

// Test suite for path simulation
TEST_SUITE(HestonSimulation) {
    auto suite = std::make_unique<TestSuite>("HestonSimulation");
    
    // Test Euler scheme simulation
    suite->addTestMethod<HestonModelTestFixture>("EulerSchemeSimulation", [](HestonModelTestFixture& fixture) {
        BENCHMARK("EulerSchemeSimulation");
        
        HestonSimulator simulator(fixture.standard_params, HestonSimulator::Scheme::EULER);
        auto path = simulator.simulate_path(1.0, 252, true);
        
        ASSERT_FALSE(path.empty());
        ASSERT_EQ(path.size(), 253);  // 252 steps + initial point
        ASSERT_GT(path.stock_prices.back(), 0.0);
        ASSERT_GE(path.variances.back(), 0.0);
        ASSERT_EQ(path.stock_prices[0], fixture.standard_params.S0);
        ASSERT_NEAR(path.variances[0], fixture.standard_params.v0, 1e-10);
    });
    
    // Test different discretization schemes
    suite->addTestMethod<HestonModelTestFixture>("DifferentSchemes", [](HestonModelTestFixture& fixture) {
        std::vector<HestonSimulator::Scheme> schemes = {
            HestonSimulator::Scheme::EULER,
            HestonSimulator::Scheme::MILSTEIN,
            HestonSimulator::Scheme::FULL_TRUNCATION,
            HestonSimulator::Scheme::REFLECTION
        };
        
        for (auto scheme : schemes) {
            HestonSimulator simulator(fixture.standard_params, scheme);
            auto path = simulator.simulate_path(0.25, 63, false);
            
            ASSERT_FALSE(path.empty());
            ASSERT_GT(path.stock_prices.back(), 0.0);
            ASSERT_GE(path.variances.back(), 0.0);
        }
    });
    
    // Test variance positivity
    suite->addTestMethod<HestonModelTestFixture>("VariancePositivity", [](HestonModelTestFixture& fixture) {
        HestonSimulator simulator(fixture.standard_params, HestonSimulator::Scheme::FULL_TRUNCATION);
        auto path = simulator.simulate_path(1.0, 252, true);
        
        // All variances should be non-negative with full truncation
        for (double variance : path.variances) {
            ASSERT_GE(variance, 0.0);
        }
        
        // All volatilities should be non-negative
        for (double volatility : path.volatilities) {
            ASSERT_GE(volatility, 0.0);
        }
    });
    
    // Test path consistency
    suite->addTestMethod<HestonModelTestFixture>("PathConsistency", [](HestonModelTestFixture& fixture) {
        HestonSimulator simulator(fixture.standard_params);
        auto path = simulator.simulate_path(1.0, 252, true);
        
        // Check that volatilities are square roots of variances
        for (size_t i = 0; i < path.size(); ++i) {
            double expected_vol = std::sqrt(std::max(path.variances[i], 0.0));
            ASSERT_NEAR(path.volatilities[i], expected_vol, 1e-10);
        }
        
        // Check time progression
        for (size_t i = 1; i < path.times.size(); ++i) {
            ASSERT_GT(path.times[i], path.times[i-1]);
        }
    });
    
    TestRegistry::getInstance().registerSuite(std::move(suite));
}

// Test suite for European option pricing
TEST_SUITE(HestonEuropeanOptions) {
    auto suite = std::make_unique<TestSuite>("HestonEuropeanOptions");
    
    // Test European call option pricing
    suite->addTestMethod<HestonModelTestFixture>("EuropeanCallPricing", [](HestonModelTestFixture& fixture) {
        BENCHMARK("EuropeanCallPricing");
        
        HestonSimulator simulator(fixture.standard_params);
        auto result = simulator.price_european_option(100.0, 0.25, true, 10000, 63);
        
        ASSERT_TRUE(result.is_valid);
        ASSERT_GT(result.price, 0.0);
        ASSERT_LT(result.price, fixture.standard_params.S0);
        ASSERT_GT(result.standard_error, 0.0);
        ASSERT_EQ(result.simulations_used, 10000);
    });
    
    // Test European put option pricing
    suite->addTestMethod<HestonModelTestFixture>("EuropeanPutPricing", [](HestonModelTestFixture& fixture) {
        HestonSimulator simulator(fixture.standard_params);
        auto result = simulator.price_european_option(100.0, 0.25, false, 10000, 63);
        
        ASSERT_TRUE(result.is_valid);
        ASSERT_GT(result.price, 0.0);
        ASSERT_LT(result.price, 100.0);  // Put price should be less than strike for ATM
    });
    
    // Test put-call parity (approximately)
    suite->addTestMethod<HestonModelTestFixture>("PutCallParity", [](HestonModelTestFixture& fixture) {
        HestonSimulator simulator(fixture.standard_params);
        
        double K = 100.0;
        double T = 0.25;
        
        auto call_result = simulator.price_european_option(K, T, true, 20000, 63);
        auto put_result = simulator.price_european_option(K, T, false, 20000, 63);
        
        ASSERT_TRUE(call_result.is_valid);
        ASSERT_TRUE(put_result.is_valid);
        
        // Put-call parity: C - P = S*e^(-q*T) - K*e^(-r*T)
        double left_side = call_result.price - put_result.price;
        double right_side = fixture.standard_params.S0 * std::exp(-fixture.standard_params.q * T) -
                           K * std::exp(-fixture.standard_params.r * T);
        
        // Allow for Monte Carlo error
        ASSERT_NEAR(left_side, right_side, 1.0);
    });
    
    // Test option pricing with different volatility levels
    suite->addTestMethod<HestonModelTestFixture>("VolatilityEffect", [](HestonModelTestFixture& fixture) {
        HestonSimulator low_vol_sim(fixture.standard_params);
        HestonSimulator high_vol_sim(fixture.high_vol_params);
        
        auto low_vol_result = low_vol_sim.price_european_option(100.0, 0.25, true, 10000, 63);
        auto high_vol_result = high_vol_sim.price_european_option(100.0, 0.25, true, 10000, 63);
        
        ASSERT_TRUE(low_vol_result.is_valid);
        ASSERT_TRUE(high_vol_result.is_valid);
        
        // Higher volatility should give higher option price
        ASSERT_GT(high_vol_result.price, low_vol_result.price);
    });
    
    TestRegistry::getInstance().registerSuite(std::move(suite));
}

// Test suite for VIX options
TEST_SUITE(HestonVIXOptions) {
    auto suite = std::make_unique<TestSuite>("HestonVIXOptions");
    
    // Test VIX calculation
    suite->addTest("VIXCalculation", []() {
        double variance = 0.04;  // 20% volatility
        double vix = VIXOptionsPricer::calculate_vix(variance);
        ASSERT_NEAR(vix, 20.0, 1e-10);
        
        variance = 0.0625;  // 25% volatility
        vix = VIXOptionsPricer::calculate_vix(variance);
        ASSERT_NEAR(vix, 25.0, 1e-10);
    });
    
    // Test VIX option pricing
    suite->addTestMethod<HestonModelTestFixture>("VIXOptionPricing", [](HestonModelTestFixture& fixture) {
        BENCHMARK("VIXOptionPricing");
        
        VIXOptionsPricer pricer(fixture.standard_params);
        auto result = pricer.price_vix_option(20.0, 0.25, true, 10000);
        
        ASSERT_TRUE(result.is_valid);
        ASSERT_GE(result.price, 0.0);
        ASSERT_LT(result.price, 50.0);  // Reasonable upper bound
    });
    
    // Test VIX term structure
    suite->addTestMethod<HestonModelTestFixture>("VIXTermStructure", [](HestonModelTestFixture& fixture) {
        VIXOptionsPricer pricer(fixture.standard_params);
        std::vector<double> maturities = {0.083, 0.25, 0.5, 1.0};  // 1M, 3M, 6M, 1Y
        
        auto vix_levels = pricer.calculate_vix_term_structure(maturities, 5000);
        
        ASSERT_EQ(vix_levels.size(), maturities.size());
        
        for (double vix : vix_levels) {
            ASSERT_GT(vix, 0.0);
            ASSERT_LT(vix, 100.0);  // Reasonable bounds
        }
    });
    
    TestRegistry::getInstance().registerSuite(std::move(suite));
}

// Test suite for variance swaps
TEST_SUITE(HestonVarianceSwaps) {
    auto suite = std::make_unique<TestSuite>("HestonVarianceSwaps");
    
    // Test variance swap pricing
    suite->addTestMethod<HestonModelTestFixture>("VarianceSwapPricing", [](HestonModelTestFixture& fixture) {
        BENCHMARK("VarianceSwapPricing");
        
        HestonSimulator simulator(fixture.standard_params);
        double fair_variance = simulator.price_variance_swap(0.25, 10000, 63);
        
        ASSERT_GT(fair_variance, 0.0);
        ASSERT_LT(fair_variance, 1.0);  // Reasonable upper bound
        
        // Should be close to long-term variance for short maturities
        // (this is a rough approximation)
        ASSERT_GT(fair_variance, fixture.standard_params.theta * 0.5);
        ASSERT_LT(fair_variance, fixture.standard_params.theta * 2.0);
    });
    
    // Test variance swap with different maturities
    suite->addTestMethod<HestonModelTestFixture>("VarianceSwapMaturities", [](HestonModelTestFixture& fixture) {
        HestonSimulator simulator(fixture.standard_params);
        
        std::vector<double> maturities = {0.25, 0.5, 1.0};
        std::vector<double> fair_variances;
        
        for (double T : maturities) {
            double fair_var = simulator.price_variance_swap(T, 5000, static_cast<int>(T * 252));
            fair_variances.push_back(fair_var);
            ASSERT_GT(fair_var, 0.0);
        }
        
        // For mean-reverting variance, longer maturities should approach theta
        ASSERT_GT(fair_variances.back(), fair_variances.front() * 0.8);
        ASSERT_LT(fair_variances.back(), fair_variances.front() * 1.2);
    });
    
    TestRegistry::getInstance().registerSuite(std::move(suite));
}

// Test suite for risk analytics
TEST_SUITE(HestonRiskAnalytics) {
    auto suite = std::make_unique<TestSuite>("HestonRiskAnalytics");
    
    // Test risk metrics calculation
    suite->addTestMethod<HestonModelTestFixture>("RiskMetricsCalculation", [](HestonModelTestFixture& fixture) {
        BENCHMARK("RiskMetricsCalculation");
        
        RiskAnalyzer analyzer(fixture.standard_params);
        auto metrics = analyzer.calculate_risk_metrics(1.0, 5000);
        
        // VaR should be negative (losses)
        ASSERT_LT(metrics.var_95, 0.0);
        ASSERT_LT(metrics.var_99, 0.0);
        ASSERT_LT(metrics.var_99, metrics.var_95);  // 99% VaR should be worse than 95%
        
        // Expected shortfall should be worse than VaR
        ASSERT_LT(metrics.expected_shortfall_95, metrics.var_95);
        
        // Maximum drawdown should be positive
        ASSERT_GE(metrics.max_drawdown, 0.0);
        ASSERT_LE(metrics.max_drawdown, 1.0);
        
        // Vol-of-vol should be positive
        ASSERT_GT(metrics.volatility_of_volatility, 0.0);
        
        // Correlation should be between -1 and 1
        ASSERT_GE(metrics.correlation_stock_vol, -1.0);
        ASSERT_LE(metrics.correlation_stock_vol, 1.0);
    });
    
    // Test Greeks calculation
    suite->addTestMethod<HestonModelTestFixture>("GreeksCalculation", [](HestonModelTestFixture& fixture) {
        RiskAnalyzer analyzer(fixture.standard_params);
        auto greeks = analyzer.calculate_greeks(100.0, 0.25, true, 5000);
        
        // Check that we have the expected Greeks
        ASSERT_TRUE(greeks.find("delta") != greeks.end());
        ASSERT_TRUE(greeks.find("gamma") != greeks.end());
        ASSERT_TRUE(greeks.find("theta") != greeks.end());
        ASSERT_TRUE(greeks.find("vega") != greeks.end());
        ASSERT_TRUE(greeks.find("rho") != greeks.end());
        
        // Delta should be between 0 and 1 for ATM call
        ASSERT_GT(greeks["delta"], 0.0);
        ASSERT_LT(greeks["delta"], 1.0);
        
        // Gamma should be positive
        ASSERT_GT(greeks["gamma"], 0.0);
        
        // Vega should be positive
        ASSERT_GT(greeks["vega"], 0.0);
    });
    
    TestRegistry::getInstance().registerSuite(std::move(suite));
}

// Performance benchmark tests
TEST_SUITE(HestonPerformance) {
    auto suite = std::make_unique<TestSuite>("HestonPerformance");
    
    // Benchmark path simulation
    suite->addTestMethod<HestonModelTestFixture>("PathSimulationBenchmark", [](HestonModelTestFixture& fixture) {
        HestonSimulator simulator(fixture.standard_params);
        
        const int num_runs = 100;
        
        {
            BENCHMARK("PathSimulation_100_runs_252_steps");
            for (int i = 0; i < num_runs; ++i) {
                auto path = simulator.simulate_path(1.0, 252, false);
                ASSERT_FALSE(path.empty());
            }
        }
    });
    
    // Benchmark option pricing
    suite->addTestMethod<HestonModelTestFixture>("OptionPricingBenchmark", [](HestonModelTestFixture& fixture) {
        HestonSimulator simulator(fixture.standard_params);
        
        const int num_runs = 10;
        
        {
            BENCHMARK("OptionPricing_10_runs_10k_sims");
            for (int i = 0; i < num_runs; ++i) {
                auto result = simulator.price_european_option(100.0, 0.25, true, 10000, 63);
                ASSERT_TRUE(result.is_valid);
            }
        }
    });
    
    TestRegistry::getInstance().registerSuite(std::move(suite));
}

// Thread safety tests
TEST_SUITE(HestonThreadSafety) {
    auto suite = std::make_unique<TestSuite>("HestonThreadSafety");
    
    // Test concurrent path simulation
    suite->addTestMethod<HestonModelTestFixture>("ConcurrentPathSimulation", [](HestonModelTestFixture& fixture) {
        const int num_threads = 4;
        const int paths_per_thread = 100;
        
        std::vector<std::thread> threads;
        std::vector<bool> results(num_threads, false);
        
        {
            BENCHMARK("ConcurrentPathSimulation_4_threads");
            
            for (int t = 0; t < num_threads; ++t) {
                threads.emplace_back([&fixture, &results, t, paths_per_thread]() {
                    bool all_valid = true;
                    HestonSimulator simulator(fixture.standard_params);
                    
                    for (int i = 0; i < paths_per_thread; ++i) {
                        auto path = simulator.simulate_path(0.25, 63, false);
                        if (path.empty() || path.stock_prices.back() <= 0.0) {
                            all_valid = false;
                            break;
                        }
                    }
                    
                    results[t] = all_valid;
                });
            }
            
            for (auto& thread : threads) {
                thread.join();
            }
        }
        
        // All threads should have succeeded
        for (bool result : results) {
            ASSERT_TRUE(result);
        }
    });
    
    TestRegistry::getInstance().registerSuite(std::move(suite));
}

// Main test runner
int main(int argc, char* argv[]) {
    // Configure logging for tests
    Utils::Logger::configure(
        Utils::LogLevel::INFO,
        true,   // console output
        true,   // file output
        "heston_test.log",
        5 * 1024 * 1024,  // 5MB max file size
        3       // keep 3 log files
    );
    
    // Initialize configuration
    Config::ConfigManager::getInstance().initialize("test_config.json");
    
    std::cout << "Heston Stochastic Volatility Model - Unit Tests" << std::endl;
    std::cout << "===============================================" << std::endl;
    
    // Print test discovery
    Testing::TestRegistry::getInstance().printDiscovery();
    
    // Run all tests
    auto stats = Testing::TestRegistry::getInstance().runAllSuites();
    
    // Return appropriate exit code
    return (stats.failed_tests == 0 && stats.error_tests == 0) ? 0 : 1;
}