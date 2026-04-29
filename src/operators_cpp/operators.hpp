#ifndef MAGSPIN_OPERATORS_CPP_OPERATORS_HPP
#define MAGSPIN_OPERATORS_CPP_OPERATORS_HPP

#include <complex>
#include <string>
#include <utility>
#include <vector>
#include <Eigen/Dense>
#include <eigen3/unsupported/Eigen/KroneckerProduct>

namespace magspin {

using cplx = std::complex<double>;
using MatXc = Eigen::MatrixXcd;
using VecXc = Eigen::VectorXcd;

struct PhysicalConstants {
    double hbar = 6.582119569e-16;
    double uB = 5.788381806e-8; // eV / mT
    double g = 2.00231930436256;
};

struct SweepConfig {
    double bz_min = -10.0;
    double bz_max = 10.0;
    double bz_step = 0.1;
};

struct SolverConfig {
    double k_s = 4e6;
    double k_d = 1e6;
    int h_number = 1;
    std::vector<int> j_numerators = {1};
    bool use_cuda_if_available = true;
    bool require_cuda = false;
    PhysicalConstants constants;
};

struct SweepResult {
    std::vector<double> bz_values;
    std::vector<double> singlet_traces;
};

struct VerificationSummary {
    double max_abs_error = 0.0;
    double mean_abs_error = 0.0;
    bool within_tolerance = false;
};

class Solver {
public:
    explicit Solver(const SolverConfig& cfg);
    cplx solve_point(double bz) const;
    SweepResult sweep(const SweepConfig& sweep_cfg) const;
    bool cuda_enabled() const;
    int total_dim() const;
    int liouville_dim() const;
    int nuclear_count() const;

private:
    SolverConfig cfg_;
    int total_dim_ = 0;
    std::vector<int> dims_;

    MatXc sx1_;
    MatXc sy1_;
    MatXc sz1_;
    MatXc sx2_;
    MatXc sy2_;
    MatXc sz2_;

    std::vector<MatXc> jx_;
    std::vector<MatXc> jy_;
    std::vector<MatXc> jz_;

    MatXc proj_singlet_;
    MatXc proj_triplet_;
    MatXc h0_;
    MatXc hsum_;

    static MatXc kron_all(const std::vector<MatXc>& mats);
    static MatXc raise_op(const MatXc& op, int index, const std::vector<int>& dims);
    static MatXc spin_j_matrix(int j_numerator, char axis);
    static MatXc make_u_transform();
    static MatXc electron_op_total_basis(const MatXc& op_single, bool first);
    static std::pair<MatXc, VecXc> build_system(const MatXc& h, const MatXc& ps, const MatXc& pt,
                                                double k_s, double k_d, double hbar, int dim);
    bool should_use_cuda() const;
    bool ensure_cuda_status() const;
    mutable bool cuda_checked_ = false;
    mutable bool cuda_available_ = false;
    mutable bool cuda_fallback_warned_ = false;
};

void write_csv(const std::string& out_path, const SweepResult& result);
VerificationSummary compare_sweeps(const SweepResult& a, const SweepResult& b, double tolerance);

} // namespace magspin

#endif
