#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <sstream>

#include "config.hpp"
#include "operators.hpp"
#include "sle_solver.hpp"

#ifdef USE_CUDA
#include "cuda_solver.hpp"
#endif

void print_usage(const char* program_name)
{
    std::cout << "SLE Solver - Stochastic Liouville Equation for Spin Systems\n\n";
    std::cout << "Usage: " << program_name << " --cpu|--gpu [options]\n\n";
    std::cout << "Required (choose one):\n";
    std::cout << "  --cpu              Use CPU solver (Eigen)\n";
    std::cout << "  --gpu              Use GPU solver (CUDA) - fails if unavailable\n\n";
    std::cout << "Options:\n";
    std::cout << "  -n, --electrons N  Number of electrons (1-4, default: 2)\n";
    std::cout << "  -g G_FACTOR        g-factor (default: 2.003)\n";
    std::cout << "  --mu MU            Bohr magneton in eV/mT (default: 5.788e-8)\n";
    std::cout << "  --hbar HBAR        Reduced Planck constant in eV*s (default: 6.582e-16)\n";
    std::cout << "  -a HYPERFINE       Hyperfine coupling constant (default: 1.0)\n";
    std::cout << "  --ks KS            Singlet recombination rate (default: 4e6)\n";
    std::cout << "  --kd KD            Dephasing rate (default: 1e6)\n";
    std::cout << "  --bz-min MIN       Minimum Bz in mT (default: -10)\n";
    std::cout << "  --bz-max MAX       Maximum Bz in mT (default: 10)\n";
    std::cout << "  --bz-step STEP     Bz step size in mT (default: 0.02)\n";
    std::cout << "  --fudge F          Fudge factor (default: 1.0)\n";
    std::cout << "  -o, --output FILE  Output CSV file (default: output/results.csv)\n";
    std::cout << "  --save             Append timestamp to filename\n";
    std::cout << "  -v, --verbose      Print detailed progress\n";
    std::cout << "  -h, --help         Show this help message\n";
}

std::string get_timestamp()
{
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
    return ss.str();
}

int main(int argc, char* argv[])
{
    // Default values
    bool use_cpu = false;
    bool use_gpu = false;
    bool verbose = false;
    bool save_with_timestamp = false;
    
    int n_electrons = 2;
    PhysicalParams params;
    SweepParams sweep;
    std::string output_file = "output/results.csv";

    // Parse arguments
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "--cpu")
        {
            use_cpu = true;
        }
        else if (arg == "--gpu")
        {
            use_gpu = true;
        }
        else if (arg == "-n" || arg == "--electrons")
        {
            if (i + 1 < argc) n_electrons = std::atoi(argv[++i]);
        }
        else if (arg == "-g")
        {
            if (i + 1 < argc) params.g = std::atof(argv[++i]);
        }
        else if (arg == "--mu")
        {
            if (i + 1 < argc) params.mu = std::atof(argv[++i]);
        }
        else if (arg == "--hbar")
        {
            if (i + 1 < argc) params.hbar = std::atof(argv[++i]);
        }
        else if (arg == "-a")
        {
            if (i + 1 < argc) params.a = std::atof(argv[++i]);
        }
        else if (arg == "--ks")
        {
            if (i + 1 < argc) params.Ks = std::atof(argv[++i]);
        }
        else if (arg == "--kd")
        {
            if (i + 1 < argc) params.Kd = std::atof(argv[++i]);
        }
        else if (arg == "--bz-min")
        {
            if (i + 1 < argc) sweep.Bz_min = std::atof(argv[++i]);
        }
        else if (arg == "--bz-max")
        {
            if (i + 1 < argc) sweep.Bz_max = std::atof(argv[++i]);
        }
        else if (arg == "--bz-step")
        {
            if (i + 1 < argc) sweep.Bz_step = std::atof(argv[++i]);
        }
        else if (arg == "--fudge")
        {
            if (i + 1 < argc) params.fudge = std::atof(argv[++i]);
        }
        else if (arg == "-o" || arg == "--output")
        {
            if (i + 1 < argc) output_file = argv[++i];
        }
        else if (arg == "--save")
        {
            save_with_timestamp = true;
        }
        else if (arg == "-v" || arg == "--verbose")
        {
            verbose = true;
        }
        else if (arg == "-h" || arg == "--help")
        {
            print_usage(argv[0]);
            return 0;
        }
        else
        {
            std::cerr << "Error: Unknown argument '" << arg << "'\n";
            std::cerr << "Use --help for usage information.\n";
            return 1;
        }
    }

    // Validate mode selection
    if (!use_cpu && !use_gpu)
    {
        std::cerr << "Error: Must specify --cpu or --gpu\n";
        std::cerr << "Use --help for usage information.\n";
        return 1;
    }
    if (use_cpu && use_gpu)
    {
        std::cerr << "Error: Cannot specify both --cpu and --gpu\n";
        return 1;
    }

    // Check GPU availability if requested
    if (use_gpu)
    {
#ifdef USE_CUDA
        if (!check_cuda_available())
        {
            std::cerr << "Error: --gpu requested but no CUDA GPU available\n";
            return 1;
        }
        if (verbose)
        {
            print_cuda_info();
        }
#else
        std::cerr << "Error: --gpu requested but compiled without CUDA support\n";
        std::cerr << "Rebuild with: make gpu\n";
        return 1;
#endif
    }

    // Validate parameters
    if (n_electrons < 1 || n_electrons > 4)
    {
        std::cerr << "Error: n_electrons must be between 1 and 4\n";
        return 1;
    }
    if (sweep.Bz_min >= sweep.Bz_max)
    {
        std::cerr << "Error: bz-min must be less than bz-max\n";
        return 1;
    }
    if (sweep.Bz_step <= 0)
    {
        std::cerr << "Error: bz-step must be positive\n";
        return 1;
    }

    // Create system configuration
    SystemConfig config(n_electrons, 1);

    // Modify output filename if --save specified
    if (save_with_timestamp)
    {
        size_t dot_pos = output_file.rfind('.');
        if (dot_pos != std::string::npos)
        {
            output_file = output_file.substr(0, dot_pos) + "_" + get_timestamp() + output_file.substr(dot_pos);
        }
        else
        {
            output_file += "_" + get_timestamp();
        }
    }

    // Print configuration
    if (verbose)
    {
        std::cout << "\nConfiguration:\n";
        std::cout << "  Solver: " << (use_gpu ? "GPU (CUDA)" : "CPU (Eigen)") << "\n";
        std::cout << "  Electrons: " << n_electrons << "\n";
        std::cout << "  Hilbert space dim: " << config.total_dim << "\n";
        std::cout << "  Linear system size: " << config.matrix_size << " x " << config.matrix_size << "\n";
        std::cout << "  g-factor: " << params.g << "\n";
        std::cout << "  mu: " << params.mu << " eV/mT\n";
        std::cout << "  hbar: " << params.hbar << " eV*s\n";
        std::cout << "  Hyperfine a: " << params.a << "\n";
        std::cout << "  Ks: " << params.Ks << "\n";
        std::cout << "  Kd: " << params.Kd << "\n";
        std::cout << "  Bz range: [" << sweep.Bz_min << ", " << sweep.Bz_max << "] mT\n";
        std::cout << "  Bz step: " << sweep.Bz_step << " mT\n";
        std::cout << "  Fudge: " << params.fudge << "\n";
        std::cout << "  Output: " << output_file << "\n\n";
    }

    // Run simulation
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (verbose)
    {
        std::cout << "Running simulation..." << std::endl;
    }

    std::vector<SimulationPoint> results;
    try
    {
        results = run_sweep(config, params, sweep, use_gpu);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error during simulation: " << e.what() << "\n";
        return 1;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Write results to CSV
    std::ofstream outfile(output_file);
    if (!outfile)
    {
        std::cerr << "Error: Cannot open output file '" << output_file << "'\n";
        return 1;
    }

    outfile << "Bz,singlet_population\n";
    outfile << std::setprecision(10);
    for (const auto& point : results)
    {
        outfile << point.Bz << "," << point.singlet_population << "\n";
    }
    outfile.close();

    // Print summary
    if (verbose)
    {
        std::cout << "\nCompleted:\n";
        std::cout << "  Points: " << results.size() << "\n";
        std::cout << "  Time: " << std::fixed << std::setprecision(3) << elapsed.count() << " s\n";
        std::cout << "  Output: " << output_file << "\n";
    }
    else
    {
        // Minimal output for scripted usage
        std::cout << output_file << "\n";
    }

    return 0;
}
