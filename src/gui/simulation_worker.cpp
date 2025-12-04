#include "simulation_worker.h"
#include <QDebug>
#include <iostream>
#include <vector>
#include <utility>
#include <eigen3/Eigen/Dense>
#include <complex>

// Include simulation functions from parent directory
// We need to declare the functions we'll use from operators.cpp
#include <eigen3/unsupported/Eigen/KroneckerProduct>

using cplx = std::complex<double>;
using MatXc = Eigen::MatrixXcd;
using VecXc = Eigen::VectorXcd;

// Forward declarations of functions from operators.cpp
struct SystemConfig;
struct SpinOperators;
struct KronOperators;
struct ProjOperators;

// We'll need to link against the compiled operators.cpp
extern std::vector<std::pair<double, double>> sweep_magnetic_field(
    const SystemConfig& config,
    double g, double mu, double a,
    double Ks, double Kd, double hbar,
    double Bz_min, double Bz_max, double Bz_step,
    double fudge);

// SystemConfig needs to be redefined here or included from a header
struct SystemConfig
{
    int n_electrons;
    int n_nuclei;
    int electron_dim;
    int nuclear_dim;
    int total_dim;
    int matrix_size;

    SystemConfig(int n_e, int n_n)
        : n_electrons(n_e), n_nuclei(n_n),
          electron_dim(1 << n_e),
          nuclear_dim(1 << n_n),
          total_dim((1 << n_e) * (1 << n_n)),
          matrix_size(((1 << n_e) * (1 << n_n)) * ((1 << n_e) * (1 << n_n)))
    {}
};

SimulationWorker::SimulationWorker(const SimulationParameters& params, QObject *parent)
    : QObject(parent), m_params(params)
{
}

SimulationWorker::~SimulationWorker()
{
}

void SimulationWorker::runSimulation()
{
    emit simulationStarted();

    try
    {
        // Create system configuration (n_electrons, 1 nuclear spin)
        SystemConfig config(m_params.n_electrons, 1);

        // Estimate number of steps for progress reporting
        int total_steps = static_cast<int>((m_params.Bz_max - m_params.Bz_min) / m_params.Bz_step) + 1;

        qDebug() << "Starting simulation with" << m_params.n_electrons << "electrons";
        qDebug() << "Magnetic field range:" << m_params.Bz_min << "to" << m_params.Bz_max
                 << "mT, step:" << m_params.Bz_step << "mT";
        qDebug() << "Total simulation points:" << total_steps;

        // Run the magnetic field sweep
        // Note: For progress updates, we'd need to modify sweep_magnetic_field to accept a callback
        // For now, we'll just emit 0% and 100%
        emit progressUpdated(0);

        std::vector<std::pair<double, double>> results = sweep_magnetic_field(
            config,
            m_params.g, m_params.mu, m_params.a,
            m_params.Ks, m_params.Kd, m_params.hbar,
            m_params.Bz_min, m_params.Bz_max, m_params.Bz_step,
            m_params.fudge
        );

        emit progressUpdated(100);

        // Convert results to Qt format
        QVector<QPointF> plotData;
        plotData.reserve(results.size());
        for (const auto& point : results)
        {
            plotData.append(QPointF(point.first, point.second));
        }

        // Emit the results
        emit resultReady(plotData);

        QString successMsg = QString("Simulation completed: %1 data points generated")
                                .arg(results.size());
        qDebug() << successMsg;
        emit simulationFinished(true, successMsg);
    }
    catch (const std::exception& e)
    {
        QString errorMsg = QString("Simulation error: %1").arg(e.what());
        qDebug() << errorMsg;
        emit errorOccurred(errorMsg);
        emit simulationFinished(false, errorMsg);
    }
    catch (...)
    {
        QString errorMsg = "Unknown error occurred during simulation";
        qDebug() << errorMsg;
        emit errorOccurred(errorMsg);
        emit simulationFinished(false, errorMsg);
    }
}
