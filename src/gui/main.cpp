#include <QApplication>
#include "main_window.h"
#include <QStyleFactory>
#include <QMetaType>
#include <QVector>
#include <QPointF>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    // Register meta types for cross-thread signal/slot connections
    qRegisterMetaType<QVector<QPointF>>("QVector<QPointF>");

    // Set application metadata
    app.setApplicationName("Magnetic Field Simulation");
    app.setApplicationVersion("1.0");
    app.setOrganizationName("MagSpin Research");

    // Use Fusion style for a modern look on Linux
    app.setStyle(QStyleFactory::create("Fusion"));

    // Create and show main window
    MainWindow window;
    window.show();

    return app.exec();
}
