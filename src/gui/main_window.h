#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include <QMainWindow>
#include <QThread>
#include <QVector>
#include <QPointF>
#include "qcustomplot.h"
#include "simulation_worker.h"

class QSpinBox;
class QDoubleSpinBox;
class QLineEdit;
class QSlider;
class QCheckBox;
class QPushButton;
class QLabel;
class QProgressBar;
class QVBoxLayout;
class QFormLayout;

/**
 * Main window for magnetic field simulation GUI
 * Contains plot area, parameter input panel, and simulation controls
 */
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onRunButtonClicked();
    void onSimulationStarted();
    void onProgressUpdated(int percentage);
    void onResultReady(const QVector<QPointF>& data);
    void onSimulationFinished(bool success, const QString& message);
    void onErrorOccurred(const QString& errorMessage);
    void onParameterChanged();
    void onFudgeSliderChanged(int value);

private:
    void setupUI();
    void createPlotArea();
    void createParameterPanel();
    void createStatusBar();
    void updateStatusText();
    SimulationParameters getParameters() const;
    bool validateParameters(QString& errorMsg) const;

    // UI Components
    QCustomPlot *m_plot;

    // Parameter inputs
    QSpinBox *m_nElectronsSpinBox;
    QLineEdit *m_gLineEdit;
    QLineEdit *m_muLineEdit;
    QDoubleSpinBox *m_aSpinBox;
    QLineEdit *m_KsLineEdit;
    QLineEdit *m_KdLineEdit;
    QLineEdit *m_hbarLineEdit;
    QSlider *m_fudgeSlider;
    QLabel *m_fudgeValueLabel;
    QDoubleSpinBox *m_BzMinSpinBox;
    QDoubleSpinBox *m_BzMaxSpinBox;
    QDoubleSpinBox *m_BzStepSpinBox;
    QCheckBox *m_useCudaCheckBox;

    // Controls
    QPushButton *m_runButton;
    QProgressBar *m_progressBar;

    // Status
    QLabel *m_statusLabel;

    // Simulation thread and worker
    QThread *m_workerThread;
    SimulationWorker *m_worker;
};

#endif // MAIN_WINDOW_H
