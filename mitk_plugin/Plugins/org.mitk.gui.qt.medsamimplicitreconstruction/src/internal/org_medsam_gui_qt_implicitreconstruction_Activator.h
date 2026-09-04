#pragma once

#include <ctkPluginActivator.h>

class org_mitk_gui_qt_medsamimplicitreconstruction_Activator : public QObject, public ctkPluginActivator
{
  Q_OBJECT
  Q_PLUGIN_METADATA(IID "org_mitk_gui_qt_medsamimplicitreconstruction")
  Q_INTERFACES(ctkPluginActivator)

public:
  void start(ctkPluginContext* context) override;
  void stop(ctkPluginContext* context) override;
};
