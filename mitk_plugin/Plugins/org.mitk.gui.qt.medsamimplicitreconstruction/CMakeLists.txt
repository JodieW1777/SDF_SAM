project(org_mitk_gui_qt_medsamimplicitreconstruction)

mitk_create_plugin(
  EXPORT_DIRECTIVE MEDSAM_IMPLICITRECONSTRUCTION_EXPORT
  EXPORTED_INCLUDE_SUFFIXES src
  MODULE_DEPENDS MitkQtWidgets MitkCore
  PACKAGE_DEPENDS Qt6|Network
)
