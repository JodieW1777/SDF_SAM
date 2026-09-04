#pragma once

#include <QmitkAbstractView.h>
#include <QNetworkAccessManager>
#include <QTemporaryDir>

#include <array>

class QLineEdit;
class QPushButton;
class QLabel;
class QSpinBox;
class QmitkSingleNodeSelectionWidget;

class QmitkMedSAMReconstructionView : public QmitkAbstractView
{
  Q_OBJECT

public:
  static const std::string VIEW_ID;

protected:
  void CreateQtPartControl(QWidget* parent) override;
  void SetFocus() override;

private slots:
  void CheckBackend();
  void Reconstruct();
  void OnHealthFinished(QNetworkReply* reply);
  void OnReconstructionFinished(QNetworkReply* reply);

private:
  bool GetIndexBoundingBox(std::array<double, 6>& bbox, QString& error) const;
  void SetBusy(bool busy, const QString& text);

  QmitkSingleNodeSelectionWidget* m_ImageSelector = nullptr;
  QmitkSingleNodeSelectionWidget* m_RoiSelector = nullptr;
  QLineEdit* m_ServerUrl = nullptr;
  QSpinBox* m_NumSlices = nullptr;
  QSpinBox* m_QueryBudget = nullptr;
  QPushButton* m_CheckButton = nullptr;
  QPushButton* m_ReconstructButton = nullptr;
  QLabel* m_Status = nullptr;
  QNetworkAccessManager m_Network;
  QTemporaryDir m_TempDir;
};
