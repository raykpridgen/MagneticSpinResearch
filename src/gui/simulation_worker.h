#ifndef SIMULATION_WORKER_H
#define SIMULATION_WORKER_H

#include <QObject>
#include <QVector>
#include <QPointF>
#include <string>

// Simulation parameters structure
struct SimulationParameters
{
    int n_electrons;
    double g;
    double mu;
    double a;
    double Ks;
    double Kd;
    double hbar;
    double fudge;
    double Bz_min;
    double Bz_max;
    double Bz_step;
    bool use_cuda;

    SimulationParameters()
        : n_electrons(2), g(2.003), mu(5.788e-8), a(1.0),
          Ks(4e6), Kd(1e6), hbar(6.582e-16), fudge(1.0),
          Bz_min(-10.0), Bz_max(10.0), Bz_step(0.02), use_cuda(false)
    {}
};

/**
 * Worker class for running magnetic field sweep simulation in background thread
 * Emits signals to update GUI with results and progress
 */
class SimulationWorker : public QObject
{
    Q_OBJECT

public:
    explicit SimulationWorker(const SimulationParameters& params, QObject *parent = nullptr);
    ~SimulationWorker();

public slots:
    void runSimulation();

signals:
    void simulationStarted();
    void progressUpdated(int percentage);
    void resultReady(const QVector<QPointF>& data);
    void simulationFinished(bool success, const QString& message);
    void errorOccurred(const QString& errorMessage);

private:
    SimulationParameters m_params;
};

#endif // SIMULATION_WORKER_H
