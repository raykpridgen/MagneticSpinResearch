#include "operators.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

#ifdef MAGSPIN_USE_CUDA
#include "solve.hpp"
#endif

namespace magspin {

MatXc Solver::kron_all(const std::vector<MatXc>& mats) {
    if (mats.empty()) {
        throw std::invalid_argument("kron_all requires at least one matrix");
    }
    MatXc out = mats.front();
    for (size_t i = 1; i < mats.size(); ++i) {
        out = Eigen::kroneckerProduct(out, mats[i]).eval();
    }
    return out;
}

MatXc Solver::raise_op(const MatXc& op, int index, const std::vector<int>& dims) {
    std::vector<MatXc> factors;
    factors.reserve(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        if (static_cast<int>(i) == index) {
            factors.push_back(op);
        } else {
            factors.push_back(MatXc::Identity(dims[i], dims[i]));
        }
    }
    return kron_all(factors);
}

MatXc Solver::spin_j_matrix(int j_numerator, char axis) {
    const int dim = j_numerator + 1;
    const double j = static_cast<double>(j_numerator) / 2.0;

    MatXc jp = MatXc::Zero(dim, dim);
    MatXc jm = MatXc::Zero(dim, dim);

    for (int col = 0; col < dim; ++col) {
        const double m = j - static_cast<double>(col);
        if (col > 0) {
            const double coeff = std::sqrt(std::max(0.0, j * (j + 1.0) - m * (m + 1.0)));
            jp(col - 1, col) = coeff;
        }
        if (col + 1 < dim) {
            const double coeff = std::sqrt(std::max(0.0, j * (j + 1.0) - m * (m - 1.0)));
            jm(col + 1, col) = coeff;
        }
    }

    if (axis == 'x') {
        return 0.5 * (jp + jm);
    }
    if (axis == 'y') {
        return cplx(0.0, -0.5) * (jp - jm);
    }

    MatXc jz = MatXc::Zero(dim, dim);
    for (int i = 0; i < dim; ++i) {
        jz(i, i) = j - static_cast<double>(i);
    }
    return jz;
}

MatXc Solver::make_u_transform() {
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    MatXc u(4, 4);
    u <<
        0.0, inv_sqrt2, -inv_sqrt2, 0.0,
        0.0, inv_sqrt2,  inv_sqrt2, 0.0,
        1.0, 0.0,        0.0,       0.0,
        0.0, 0.0,        0.0,       1.0;
    return u;
}

MatXc Solver::electron_op_total_basis(const MatXc& op_single, bool first) {
    MatXc id2 = MatXc::Identity(2, 2);
    MatXc product = first ? Eigen::kroneckerProduct(op_single, id2).eval()
                          : Eigen::kroneckerProduct(id2, op_single).eval();
    MatXc u = make_u_transform();
    return u * product * u.adjoint();
}

std::pair<MatXc, VecXc> Solver::build_system(
    const MatXc& h, const MatXc& ps, const MatXc& pt, double k_s, double k_d, double hbar, int dim) {
    const int vec_dim = dim * dim;
    MatXc id = MatXc::Identity(dim, dim);

    MatXc liouvillian = (-cplx(0.0, 1.0) / hbar) *
        (Eigen::kroneckerProduct(id, h).eval() - Eigen::kroneckerProduct(h.transpose(), id).eval());
    MatXc sing_anticom = MatXc::Zero(vec_dim, vec_dim);
    if (k_s + k_d > 0.0) {
        sing_anticom = (k_s + k_d) *
            (Eigen::kroneckerProduct(id, ps).eval() + Eigen::kroneckerProduct(ps.transpose(), id).eval());
    }
    MatXc trip_anticom = MatXc::Zero(vec_dim, vec_dim);
    if (k_d > 0.0) {
        trip_anticom = k_d *
            (Eigen::kroneckerProduct(id, pt).eval() + Eigen::kroneckerProduct(pt.transpose(), id).eval());
    }

    MatXc L = liouvillian + 0.5 * sing_anticom + 0.5 * trip_anticom;
    VecXc source = VecXc::Zero(vec_dim);
    for (int i = 0; i < dim; ++i) {
        source(i * (dim + 1)) = 1.0 / static_cast<double>(dim);
    }

    L.row(vec_dim - 1).setZero();
    for (int i = 0; i < dim; ++i) {
        L(vec_dim - 1, i * (dim + 1)) = 1.0;
    }
    source(vec_dim - 1) = 1.0;

    return {L, source};
}

Solver::Solver(const SolverConfig& cfg) : cfg_(cfg) {
    if (cfg_.j_numerators.empty()) {
        throw std::invalid_argument("At least one nuclear spin j_numerator is required");
    }
    if (cfg_.h_number < 1 || cfg_.h_number > static_cast<int>(cfg_.j_numerators.size())) {
        throw std::invalid_argument("h_number must be in [1, number of nuclear spins]");
    }

    dims_.push_back(4);
    for (int jn : cfg_.j_numerators) {
        dims_.push_back(jn + 1);
    }
    total_dim_ = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());

    MatXc sx_single(2, 2), sy_single(2, 2), sz_single(2, 2);
    sx_single << 0.0, 0.5, 0.5, 0.0;
    sy_single << 0.0, cplx(0.0, -0.5), cplx(0.0, 0.5), 0.0;
    sz_single << 0.5, 0.0, 0.0, -0.5;

    MatXc sx1_base = electron_op_total_basis(sx_single, true);
    MatXc sy1_base = electron_op_total_basis(sy_single, true);
    MatXc sz1_base = electron_op_total_basis(sz_single, true);
    MatXc sx2_base = electron_op_total_basis(sx_single, false);
    MatXc sy2_base = electron_op_total_basis(sy_single, false);
    MatXc sz2_base = electron_op_total_basis(sz_single, false);

    sx1_ = raise_op(sx1_base, 0, dims_);
    sy1_ = raise_op(sy1_base, 0, dims_);
    sz1_ = raise_op(sz1_base, 0, dims_);
    sx2_ = raise_op(sx2_base, 0, dims_);
    sy2_ = raise_op(sy2_base, 0, dims_);
    sz2_ = raise_op(sz2_base, 0, dims_);

    for (size_t i = 0; i < cfg_.j_numerators.size(); ++i) {
        const int jn = cfg_.j_numerators[i];
        jx_.push_back(raise_op(spin_j_matrix(jn, 'x'), static_cast<int>(i + 1), dims_));
        jy_.push_back(raise_op(spin_j_matrix(jn, 'y'), static_cast<int>(i + 1), dims_));
        jz_.push_back(raise_op(spin_j_matrix(jn, 'z'), static_cast<int>(i + 1), dims_));
    }

    MatXc lower_singlet = MatXc::Zero(4, 4);
    lower_singlet(0, 0) = 1.0;
    MatXc lower_triplet = MatXc::Identity(4, 4) - lower_singlet;
    proj_singlet_ = raise_op(lower_singlet, 0, dims_);
    proj_triplet_ = raise_op(lower_triplet, 0, dims_);

    h0_ = cfg_.constants.g * cfg_.constants.uB * (sz1_ + sz2_);
    hsum_ = MatXc::Zero(total_dim_, total_dim_);
    for (int i = 0; i < cfg_.h_number; ++i) {
        const MatXc h_i = cfg_.constants.g * cfg_.constants.uB *
            (jz_[i] * (i == 0 ? sz1_ : sz2_) +
             jx_[i] * (i == 0 ? sx1_ : sx2_) +
             jy_[i] * (i == 0 ? sy1_ : sy2_));
        hsum_ += h_i;
    }
}

cplx Solver::solve_point(double bz) const {
    MatXc h = bz * h0_ + hsum_;
    auto [L, source] = build_system(h, proj_singlet_, proj_triplet_, cfg_.k_s, cfg_.k_d,
                                    cfg_.constants.hbar, total_dim_);
    VecXc vec_rho(source.size());
    bool solved_on_cuda = false;

    if (should_use_cuda()) {
#ifdef MAGSPIN_USE_CUDA
        try {
            solve_cuda_complex(L.data(), source.data(), vec_rho.data(), L.rows());
            solved_on_cuda = true;
        } catch (const std::exception& ex) {
            if (!cuda_fallback_warned_) {
                std::cerr << "CUDA solve failed (" << ex.what()
                          << "); falling back to Eigen CPU solver.\n";
                cuda_fallback_warned_ = true;
            }
        }
#endif
    }

    if (!solved_on_cuda) {
        vec_rho = L.fullPivLu().solve(source);
    }

    MatXc rho = Eigen::Map<const MatXc>(vec_rho.data(), total_dim_, total_dim_);
    return (proj_singlet_ * rho).trace();
}

bool Solver::ensure_cuda_status() const {
#ifdef MAGSPIN_USE_CUDA
    if (!cuda_checked_) {
        cuda_available_ = check_cuda_available();
        cuda_checked_ = true;
    }
    return cuda_available_;
#else
    return false;
#endif
}

bool Solver::should_use_cuda() const {
    if (!cfg_.use_cuda_if_available) {
        return false;
    }
    const bool available = ensure_cuda_status();
    if (cfg_.require_cuda && !available) {
        throw std::runtime_error("CUDA was requested but no CUDA device is available");
    }
    return available;
}

bool Solver::cuda_enabled() const {
    return should_use_cuda();
}

int Solver::total_dim() const {
    return total_dim_;
}

int Solver::liouville_dim() const {
    return total_dim_ * total_dim_;
}

int Solver::nuclear_count() const {
    return static_cast<int>(cfg_.j_numerators.size());
}

SweepResult Solver::sweep(const SweepConfig& sweep_cfg) const {
    SweepResult out;
    if (sweep_cfg.bz_step <= 0.0) {
        throw std::invalid_argument("bz_step must be positive");
    }
    const int count = std::max(0, static_cast<int>(
        std::floor((sweep_cfg.bz_max - sweep_cfg.bz_min) / sweep_cfg.bz_step + 1e-12)));
    out.bz_values.reserve(count);
    out.singlet_traces.reserve(count);
    for (int i = 0; i < count; ++i) {
        const double bz = sweep_cfg.bz_min + static_cast<double>(i) * sweep_cfg.bz_step;
        out.bz_values.push_back(bz);
        out.singlet_traces.push_back(solve_point(bz).real());
    }
    return out;
}

void write_csv(const std::string& out_path, const SweepResult& result) {
    std::ofstream out(out_path);
    if (!out.good()) {
        throw std::runtime_error("Failed to open output file: " + out_path);
    }
    out << "Bz,singlet_trace\n";
    for (size_t i = 0; i < result.bz_values.size(); ++i) {
        out << result.bz_values[i] << "," << result.singlet_traces[i] << "\n";
    }
}

VerificationSummary compare_sweeps(const SweepResult& a, const SweepResult& b, double tolerance) {
    if (a.bz_values.size() != b.bz_values.size()) {
        throw std::invalid_argument("Sweep sizes differ");
    }
    VerificationSummary summary;
    const size_t n = a.bz_values.size();
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double err = std::abs(a.singlet_traces[i] - b.singlet_traces[i]);
        summary.max_abs_error = std::max(summary.max_abs_error, err);
        sum += err;
    }
    summary.mean_abs_error = (n == 0) ? 0.0 : sum / static_cast<double>(n);
    summary.within_tolerance = summary.max_abs_error <= tolerance;
    return summary;
}

} // namespace magspin
