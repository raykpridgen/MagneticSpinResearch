#include "main_window.h"
#include <QWidget>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QSlider>
#include <QCheckBox>
#include <QPushButton>
#include <QLabel>
#include <QProgressBar>
#include <QStatusBar>
#include <QMessageBox>
#include <QGroupBox>
#include <QValidator>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),
      m_plot(nullptr),
      m_workerThread(nullptr),
      m_worker(nullptr)
{
    setupUI();
    setWindowTitle("Magnetic Field Simulation - Radical Pair Mechanism");
    resize(1200, 700);

    // Initialize status text
    updateStatusText();
}

MainWindow::~MainWindow()
{
    // Clean up worker thread
    if (m_workerThread)
    {
        m_workerThread->quit();
        m_workerThread->wait();
        delete m_workerThread;
    }
}

void MainWindow::setupUI()
{
    // Create central widget with horizontal layout
    QWidget *centralWidget = new QWidget(this);
    QHBoxLayout *mainLayout = new QHBoxLayout(centralWidget);

    // Create plot area (left side)
    createPlotArea();
    mainLayout->addWidget(m_plot, 3); // 3:1 ratio

    // Create parameter panel (right side)
    QWidget *parameterPanel = new QWidget();
    QVBoxLayout *paramLayout = new QVBoxLayout(parameterPanel);
    createParameterPanel();

    // Add all parameter widgets to the panel
    QGroupBox *systemGroup = new QGroupBox("System Configuration");
    QFormLayout *systemForm = new QFormLayout();
    systemForm->addRow("Number of Electrons:", m_nElectronsSpinBox);
    systemGroup->setLayout(systemForm);
    paramLayout->addWidget(systemGroup);

    QGroupBox *physicalGroup = new QGroupBox("Physical Parameters");
    QFormLayout *physicalForm = new QFormLayout();
    physicalForm->addRow("g-factor:", m_gLineEdit);
    physicalForm->addRow("μB (eV/mT):", m_muLineEdit);
    physicalForm->addRow("Hyperfine a:", m_aSpinBox);
    physicalForm->addRow("ℏ (eV·s):", m_hbarLineEdit);
    physicalGroup->setLayout(physicalForm);
    paramLayout->addWidget(physicalGroup);

    QGroupBox *ratesGroup = new QGroupBox("Reaction Rates");
    QFormLayout *ratesForm = new QFormLayout();
    ratesForm->addRow("Ks (singlet):", m_KsLineEdit);
    ratesForm->addRow("Kd (dephasing):", m_KdLineEdit);
    ratesGroup->setLayout(ratesForm);
    paramLayout->addWidget(ratesGroup);

    QGroupBox *sweepGroup = new QGroupBox("Magnetic Field Sweep");
    QFormLayout *sweepForm = new QFormLayout();
    sweepForm->addRow("Bz Min (mT):", m_BzMinSpinBox);
    sweepForm->addRow("Bz Max (mT):", m_BzMaxSpinBox);
    sweepForm->addRow("Bz Step (mT):", m_BzStepSpinBox);
    sweepGroup->setLayout(sweepForm);
    paramLayout->addWidget(sweepGroup);

    QGroupBox *adjustmentGroup = new QGroupBox("Adjustments");
    QFormLayout *adjustmentForm = new QFormLayout();
    QHBoxLayout *fudgeLayout = new QHBoxLayout();
    fudgeLayout->addWidget(m_fudgeSlider);
    fudgeLayout->addWidget(m_fudgeValueLabel);
    adjustmentForm->addRow("Fudge Factor:", fudgeLayout);
    adjustmentForm->addRow("Use CUDA:", m_useCudaCheckBox);
    adjustmentGroup->setLayout(adjustmentForm);
    paramLayout->addWidget(adjustmentGroup);

    // Add run button and progress bar
    paramLayout->addWidget(m_runButton);
    paramLayout->addWidget(m_progressBar);

    // Add stretch to push everything to the top
    paramLayout->addStretch();

    mainLayout->addWidget(parameterPanel, 1); // 3:1 ratio

    setCentralWidget(centralWidget);

    // Create status bar
    createStatusBar();
}

void MainWindow::createPlotArea()
{
    m_plot = new QCustomPlot();

    // Configure plot appearance
    m_plot->setBackground(QBrush(Qt::white));
    m_plot->xAxis->setLabel("Magnetic Field Bz (mT)");
    m_plot->yAxis->setLabel("Singlet Population");

    // Add a graph
    m_plot->addGraph();
    m_plot->graph(0)->setPen(QPen(Qt::blue, 2));
    m_plot->graph(0)->setName("Singlet Population");

    // Enable interactions
    m_plot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);

    // Set initial ranges to match default sweep parameters
    m_plot->xAxis->setRange(-10, 10);
    m_plot->yAxis->setRange(0, 1);

    m_plot->replot();
}

void MainWindow::createParameterPanel()
{
    // Number of electrons
    m_nElectronsSpinBox = new QSpinBox();
    m_nElectronsSpinBox->setRange(1, 4);
    m_nElectronsSpinBox->setValue(2);
    connect(m_nElectronsSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &MainWindow::onParameterChanged);

    // g-factor
    m_gLineEdit = new QLineEdit("2.003");
    m_gLineEdit->setValidator(new QDoubleValidator(0.0, 10.0, 6, this));
    connect(m_gLineEdit, &QLineEdit::textChanged, this, &MainWindow::onParameterChanged);

    // Bohr magneton
    m_muLineEdit = new QLineEdit("5.788e-8");
    connect(m_muLineEdit, &QLineEdit::textChanged, this, &MainWindow::onParameterChanged);

    // Hyperfine coupling
    m_aSpinBox = new QDoubleSpinBox();
    m_aSpinBox->setRange(0.001, 100.0);
    m_aSpinBox->setDecimals(3);
    m_aSpinBox->setValue(1.0);
    m_aSpinBox->setSingleStep(0.1);
    connect(m_aSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &MainWindow::onParameterChanged);

    // Planck constant
    m_hbarLineEdit = new QLineEdit("6.582e-16");
    connect(m_hbarLineEdit, &QLineEdit::textChanged, this, &MainWindow::onParameterChanged);

    // Reaction rates
    m_KsLineEdit = new QLineEdit("4e6");
    connect(m_KsLineEdit, &QLineEdit::textChanged, this, &MainWindow::onParameterChanged);

    m_KdLineEdit = new QLineEdit("1e6");
    connect(m_KdLineEdit, &QLineEdit::textChanged, this, &MainWindow::onParameterChanged);

    // Fudge factor slider
    m_fudgeSlider = new QSlider(Qt::Horizontal);
    m_fudgeSlider->setRange(10, 200); // 0.1 to 2.0 in increments of 0.01
    m_fudgeSlider->setValue(100); // 1.0
    connect(m_fudgeSlider, &QSlider::valueChanged, this, &MainWindow::onFudgeSliderChanged);

    m_fudgeValueLabel = new QLabel("1.00");
    m_fudgeValueLabel->setMinimumWidth(40);

    // Magnetic field sweep range
    m_BzMinSpinBox = new QDoubleSpinBox();
    m_BzMinSpinBox->setRange(-100.0, 100.0);
    m_BzMinSpinBox->setDecimals(2);
    m_BzMinSpinBox->setValue(-10.0);  // Match Mathematica: -10 mT
    m_BzMinSpinBox->setSingleStep(0.5);
    connect(m_BzMinSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &MainWindow::onParameterChanged);

    m_BzMaxSpinBox = new QDoubleSpinBox();
    m_BzMaxSpinBox->setRange(-100.0, 100.0);
    m_BzMaxSpinBox->setDecimals(2);
    m_BzMaxSpinBox->setValue(10.0);  // Match Mathematica: 10 mT
    m_BzMaxSpinBox->setSingleStep(0.5);
    connect(m_BzMaxSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &MainWindow::onParameterChanged);

    m_BzStepSpinBox = new QDoubleSpinBox();
    m_BzStepSpinBox->setRange(0.001, 10.0);
    m_BzStepSpinBox->setDecimals(3);
    m_BzStepSpinBox->setValue(0.02);  // Match Mathematica: 0.02 mT step
    m_BzStepSpinBox->setSingleStep(0.01);
    connect(m_BzStepSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &MainWindow::onParameterChanged);

    // CUDA checkbox
    m_useCudaCheckBox = new QCheckBox();
    m_useCudaCheckBox->setChecked(false);

    // Run button
    m_runButton = new QPushButton("Run Simulation");
    m_runButton->setStyleSheet("QPushButton { font-weight: bold; padding: 8px; }");
    connect(m_runButton, &QPushButton::clicked, this, &MainWindow::onRunButtonClicked);

    // Progress bar
    m_progressBar = new QProgressBar();
    m_progressBar->setRange(0, 100);
    m_progressBar->setValue(0);
    m_progressBar->setVisible(false);
}

void MainWindow::createStatusBar()
{
    m_statusLabel = new QLabel();
    statusBar()->addWidget(m_statusLabel, 1);
}

void MainWindow::updateStatusText()
{
    SimulationParameters params = getParameters();
    QString status = QString("%1 Electron%2 | g=%3 | a=%4 | Ks=%5 | Kd=%6 | fudge=%7 | Sweep: %8-%9 mT")
                        .arg(params.n_electrons)
                        .arg(params.n_electrons > 1 ? "s" : "")
                        .arg(params.g, 0, 'f', 3)
                        .arg(params.a, 0, 'f', 2)
                        .arg(params.Ks, 0, 'e', 1)
                        .arg(params.Kd, 0, 'e', 1)
                        .arg(params.fudge, 0, 'f', 2)
                        .arg(params.Bz_min, 0, 'f', 1)
                        .arg(params.Bz_max, 0, 'f', 1);
    m_statusLabel->setText(status);
}

SimulationParameters MainWindow::getParameters() const
{
    SimulationParameters params;

    params.n_electrons = m_nElectronsSpinBox->value();
    params.g = m_gLineEdit->text().toDouble();
    params.mu = m_muLineEdit->text().toDouble();
    params.a = m_aSpinBox->value();
    params.Ks = m_KsLineEdit->text().toDouble();
    params.Kd = m_KdLineEdit->text().toDouble();
    params.hbar = m_hbarLineEdit->text().toDouble();
    params.fudge = m_fudgeSlider->value() / 100.0;
    params.Bz_min = m_BzMinSpinBox->value();
    params.Bz_max = m_BzMaxSpinBox->value();
    params.Bz_step = m_BzStepSpinBox->value();
    params.use_cuda = m_useCudaCheckBox->isChecked();

    return params;
}

bool MainWindow::validateParameters(QString& errorMsg) const
{
    SimulationParameters params = getParameters();

    if (params.Bz_min >= params.Bz_max)
    {
        errorMsg = "Bz Min must be less than Bz Max";
        return false;
    }

    if (params.Bz_step <= 0)
    {
        errorMsg = "Bz Step must be positive";
        return false;
    }

    if ((params.Bz_max - params.Bz_min) / params.Bz_step > 10000)
    {
        errorMsg = "Too many steps (>10000). Increase step size or reduce range.";
        return false;
    }

    if (params.g <= 0 || params.mu <= 0 || params.hbar <= 0)
    {
        errorMsg = "Physical constants must be positive";
        return false;
    }

    if (params.Ks <= 0 || params.Kd <= 0)
    {
        errorMsg = "Reaction rates must be positive";
        return false;
    }

    return true;
}

void MainWindow::onRunButtonClicked()
{
    // Validate parameters
    QString errorMsg;
    if (!validateParameters(errorMsg))
    {
        QMessageBox::warning(this, "Invalid Parameters", errorMsg);
        return;
    }

    // Disable run button during simulation
    m_runButton->setEnabled(false);
    m_progressBar->setVisible(true);
    m_progressBar->setValue(0);

    // Get parameters
    SimulationParameters params = getParameters();

    // Create worker thread if it doesn't exist
    if (m_workerThread)
    {
        m_workerThread->quit();
        m_workerThread->wait();
        delete m_workerThread;
        m_workerThread = nullptr;
    }

    m_workerThread = new QThread();
    m_worker = new SimulationWorker(params);
    m_worker->moveToThread(m_workerThread);

    // Connect signals
    connect(m_workerThread, &QThread::started, m_worker, &SimulationWorker::runSimulation);
    connect(m_worker, &SimulationWorker::simulationStarted, this, &MainWindow::onSimulationStarted);
    connect(m_worker, &SimulationWorker::progressUpdated, this, &MainWindow::onProgressUpdated);
    connect(m_worker, &SimulationWorker::resultReady, this, &MainWindow::onResultReady);
    connect(m_worker, &SimulationWorker::simulationFinished, this, &MainWindow::onSimulationFinished);
    connect(m_worker, &SimulationWorker::errorOccurred, this, &MainWindow::onErrorOccurred);

    // Clean up thread when finished
    connect(m_worker, &SimulationWorker::simulationFinished, m_workerThread, &QThread::quit);
    connect(m_workerThread, &QThread::finished, m_worker, &QObject::deleteLater);

    // Start the thread
    m_workerThread->start();
}

void MainWindow::onSimulationStarted()
{
    qDebug() << "Simulation started";
    statusBar()->showMessage("Simulation running...");
}

void MainWindow::onProgressUpdated(int percentage)
{
    m_progressBar->setValue(percentage);
}

void MainWindow::onResultReady(const QVector<QPointF>& data)
{
    // Update plot with new data
    QVector<double> x, y;
    x.reserve(data.size());
    y.reserve(data.size());

    for (const QPointF& point : data)
    {
        x.append(point.x());
        y.append(point.y());
    }

    m_plot->graph(0)->setData(x, y);

    // Auto-scale axes
    m_plot->xAxis->rescale();
    m_plot->yAxis->rescale();

    // Add some margin
    m_plot->xAxis->scaleRange(1.1, m_plot->xAxis->range().center());
    m_plot->yAxis->scaleRange(1.1, m_plot->yAxis->range().center());

    m_plot->replot();

    qDebug() << "Plot updated with" << data.size() << "points";
}

void MainWindow::onSimulationFinished(bool success, const QString& message)
{
    m_runButton->setEnabled(true);
    m_progressBar->setVisible(false);

    if (success)
    {
        statusBar()->showMessage(message, 5000);
        QMessageBox::information(this, "Simulation Complete", message);
    }
    else
    {
        statusBar()->showMessage("Simulation failed", 5000);
    }
}

void MainWindow::onErrorOccurred(const QString& errorMessage)
{
    QMessageBox::critical(this, "Simulation Error", errorMessage);
    statusBar()->showMessage("Error: " + errorMessage, 5000);
}

void MainWindow::onParameterChanged()
{
    updateStatusText();
}

void MainWindow::onFudgeSliderChanged(int value)
{
    double fudge = value / 100.0;
    m_fudgeValueLabel->setText(QString::number(fudge, 'f', 2));
    updateStatusText();
}
