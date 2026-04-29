#include "operators.hpp"
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

namespace {

std::vector<int> parse_j_numerators(const std::string& csv) {
    std::vector<int> out;
    std::stringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            out.push_back(std::stoi(token));
        }
    }
    if (out.empty()) {
        throw std::invalid_argument("j-numerators cannot be empty");
    }
    return out;
}

void print_usage(const char* bin) {
    std::cout << "Usage: " << bin << " [options]\n"
              << "  --ks <float>                 Singlet rate (default 4e6)\n"
              << "  --kd <float>                 Triplet/dephasing rate (default 1e6)\n"
              << "  --h-number <int>             Number of hyperfine couplings (default 1)\n"
              << "  --j-numerators <csv>         Nuclear spin numerators n for j=n/2 (default 1)\n"
              << "  --bz-min <float>             Sweep lower bound (default -10)\n"
              << "  --bz-max <float>             Sweep upper bound, exclusive (default 10)\n"
              << "  --bz-step <float>            Sweep step (default 0.1)\n"
              << "  --out <path>                 CSV output path (default data/operators_cpp_sweep.csv)\n"
              << "  --plot-out <path>            Plot image output path (default data/operators_cpp_sweep.png)\n"
              << "  --no-plot                    Skip plotting\n"
              << "  --no-show                    Save image but do not open interactive window\n";
}

} // namespace

int main(int argc, char** argv) {
    magspin::SolverConfig cfg;
    magspin::SweepConfig sweep;
    std::string out_csv = "data/operators_cpp_sweep.csv";
    std::string plot_out = "data/operators_cpp_sweep.png";
    bool do_plot = true;
    bool no_show = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto read_value = [&](const std::string& key) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value for " + key);
            }
            return argv[++i];
        };

        if (arg == "--ks") {
            cfg.k_s = std::stod(read_value(arg));
        } else if (arg == "--kd") {
            cfg.k_d = std::stod(read_value(arg));
        } else if (arg == "--h-number") {
            cfg.h_number = std::stoi(read_value(arg));
        } else if (arg == "--j-numerators") {
            cfg.j_numerators = parse_j_numerators(read_value(arg));
        } else if (arg == "--bz-min") {
            sweep.bz_min = std::stod(read_value(arg));
        } else if (arg == "--bz-max") {
            sweep.bz_max = std::stod(read_value(arg));
        } else if (arg == "--bz-step") {
            sweep.bz_step = std::stod(read_value(arg));
        } else if (arg == "--out") {
            out_csv = read_value(arg);
        } else if (arg == "--plot-out") {
            plot_out = read_value(arg);
        } else if (arg == "--no-plot") {
            do_plot = false;
        } else if (arg == "--no-show") {
            no_show = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }

    try {
        magspin::Solver solver(cfg);
        magspin::SweepResult result = solver.sweep(sweep);
        magspin::write_csv(out_csv, result);
        std::cout << "Wrote sweep CSV to: " << out_csv << "\n";

        if (do_plot) {
            plt::figure_size(1100, 700);
            plt::plot(result.bz_values, result.singlet_traces);
            plt::xlabel("Bz");
            plt::ylabel("Tr(Ps * rho_ss)");
            plt::title("SLE sweep (C++ port)");
            plt::grid(true);
            plt::save(plot_out);
            std::cout << "Wrote plot to: " << plot_out << "\n";
            if (!no_show) {
                plt::show();
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
