#include "org_medsam_gui_qt_implicitreconstruction_Activator.h"
#include "QmitkMedSAMReconstructionView.h"

#include <berryIExtensionRegistry.h>
#include <berryPlatform.h>
#include <ctkPluginContext.h>

void org_mitk_gui_qt_medsamimplicitreconstruction_Activator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(QmitkMedSAMReconstructionView, context)
}

void org_mitk_gui_qt_medsamimplicitreconstruction_Activator::stop(ctkPluginContext*)
{
}
