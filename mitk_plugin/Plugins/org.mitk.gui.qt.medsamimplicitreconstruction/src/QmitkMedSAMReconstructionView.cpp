#include "QmitkMedSAMReconstructionView.h"

#include <QmitkSingleNodeSelectionWidget.h>
#include <mitkDataNode.h>
#include <mitkGeometryData.h>
#include <mitkImage.h>
#include <mitkIOUtil.h>
#include <mitkNodePredicateDataType.h>
#include <mitkRenderingManager.h>
#include <mitkSurface.h>

#include <vtkPoints.h>
#include <vtkPolyData.h>

#include <QFile>
#include <QFormLayout>
#include <QHttpMultiPart>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QNetworkReply>
#include <QPushButton>
#include <QSpinBox>
#include <QVBoxLayout>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

const std::string QmitkMedSAMReconstructionView::VIEW_ID =
  "org.medsam.views.implicitreconstruction";

void QmitkMedSAMReconstructionView::CreateQtPartControl(QWidget* parent)
{
  auto* layout = new QVBoxLayout(parent);
  auto* help = new QLabel(
    "1. Select a 3-D CT image.\n"
    "2. Create/edit a Bounding Shape ROI in MITK.\n"
    "3. Select that ROI and reconstruct.", parent);
  help->setWordWrap(true);
  layout->addWidget(help);

  m_ImageSelector = new QmitkSingleNodeSelectionWidget(parent);
  m_ImageSelector->SetDataStorage(GetDataStorage());
  m_ImageSelector->SetNodePredicate(mitk::NodePredicateDataType::New("Image"));
  m_ImageSelector->SetSelectionIsOptional(false);
  m_ImageSelector->SetEmptyInfo("Select input CT");
  layout->addWidget(m_ImageSelector);

  m_RoiSelector = new QmitkSingleNodeSelectionWidget(parent);
  m_RoiSelector->SetDataStorage(GetDataStorage());
  m_RoiSelector->SetNodePredicate(mitk::NodePredicateDataType::New("GeometryData"));
  m_RoiSelector->SetSelectionIsOptional(false);
  m_RoiSelector->SetEmptyInfo("Select Bounding Shape ROI");
  layout->addWidget(m_RoiSelector);

  auto* form = new QFormLayout;
  m_ServerUrl = new QLineEdit("http://127.0.0.1:8765", parent);
  m_NumSlices = new QSpinBox(parent);
  m_NumSlices->setRange(4, 32);
  m_NumSlices->setValue(8);
  m_QueryBudget = new QSpinBox(parent);
  m_QueryBudget->setRange(8000, 2000000);
  m_QueryBudget->setSingleStep(10000);
  m_QueryBudget->setValue(100000);
  form->addRow("Backend URL", m_ServerUrl);
  form->addRow("Sparse slices / plane", m_NumSlices);
  form->addRow("Query budget", m_QueryBudget);
  layout->addLayout(form);

  m_CheckButton = new QPushButton("Check backend", parent);
  m_ReconstructButton = new QPushButton("Reconstruct mesh", parent);
  m_Status = new QLabel("Idle", parent);
  layout->addWidget(m_CheckButton);
  layout->addWidget(m_ReconstructButton);
  layout->addWidget(m_Status);
  layout->addStretch();

  connect(m_CheckButton, &QPushButton::clicked, this, &QmitkMedSAMReconstructionView::CheckBackend);
  connect(m_ReconstructButton, &QPushButton::clicked, this, &QmitkMedSAMReconstructionView::Reconstruct);
}

void QmitkMedSAMReconstructionView::SetFocus()
{
  m_ImageSelector->setFocus();
}

void QmitkMedSAMReconstructionView::SetBusy(bool busy, const QString& text)
{
  m_CheckButton->setEnabled(!busy);
  m_ReconstructButton->setEnabled(!busy);
  m_Status->setText(text);
}

void QmitkMedSAMReconstructionView::CheckBackend()
{
  SetBusy(true, "Checking backend...");
  auto* reply = m_Network.get(QNetworkRequest(QUrl(m_ServerUrl->text() + "/health")));
  connect(reply, &QNetworkReply::finished, this, [this, reply]() { OnHealthFinished(reply); });
}

void QmitkMedSAMReconstructionView::OnHealthFinished(QNetworkReply* reply)
{
  const bool ok = reply->error() == QNetworkReply::NoError;
  SetBusy(false, ok ? "Backend ready" : "Backend error: " + reply->errorString());
  reply->deleteLater();
}

bool QmitkMedSAMReconstructionView::GetIndexBoundingBox(
  std::array<double, 6>& bbox, QString& error) const
{
  auto imageNode = m_ImageSelector->GetSelectedNode();
  auto roiNode = m_RoiSelector->GetSelectedNode();
  auto* image = imageNode.IsNotNull() ? dynamic_cast<mitk::Image*>(imageNode->GetData()) : nullptr;
  auto* roi = roiNode.IsNotNull() ? dynamic_cast<mitk::GeometryData*>(roiNode->GetData()) : nullptr;
  if (image == nullptr || roi == nullptr)
  {
    error = "Select both a 3-D image and a GeometryData bounding shape.";
    return false;
  }
  if (image->GetDimension() != 3)
  {
    error = "Only scalar 3-D images are supported.";
    return false;
  }

  std::array<double, 3> lo = {
    std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
    std::numeric_limits<double>::max()};
  std::array<double, 3> hi = {
    std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(),
    std::numeric_limits<double>::lowest()};
  const auto* roiGeometry = roi->GetGeometry();
  const auto* imageGeometry = image->GetGeometry();
  for (unsigned int corner = 0; corner < 8; ++corner)
  {
    const mitk::Point3D world = roiGeometry->GetCornerPoint(static_cast<int>(corner));
    mitk::Point3D index;
    imageGeometry->WorldToIndex(world, index);
    for (unsigned int axis = 0; axis < 3; ++axis)
    {
      lo[axis] = std::min(lo[axis], index[axis]);
      hi[axis] = std::max(hi[axis], index[axis]);
    }
  }
  bbox = {lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]};
  return true;
}

void QmitkMedSAMReconstructionView::Reconstruct()
{
  std::array<double, 6> bbox;
  QString error;
  if (!GetIndexBoundingBox(bbox, error))
  {
    QMessageBox::warning(nullptr, "MedSAM", error);
    return;
  }
  if (!m_TempDir.isValid())
  {
    QMessageBox::critical(nullptr, "MedSAM", "Could not create a temporary directory.");
    return;
  }

  const QString imagePath = m_TempDir.filePath("mitk_input.nii.gz");
  auto imageNode = m_ImageSelector->GetSelectedNode();
  try
  {
    mitk::IOUtil::Save(imageNode->GetData(), imagePath.toStdString());
  }
  catch (const std::exception& exc)
  {
    QMessageBox::critical(nullptr, "MedSAM", QString("Image export failed: %1").arg(exc.what()));
    return;
  }

  QJsonArray bboxJson;
  for (double value : bbox)
    bboxJson.append(value);

  // Send MITK's index-to-world transform explicitly. PLY carries no RAS/LPS
  // metadata, so reconstructing this transform from a temporary NIfTI in the
  // backend can mirror/translate the returned surface.
  const auto* imageGeometry = imageNode->GetData()->GetGeometry();
  const auto* indexToWorld = imageGeometry->GetIndexToWorldTransform();
  const auto& matrix = indexToWorld->GetMatrix();
  const auto& offset = indexToWorld->GetOffset();
  QJsonArray indexToWorldJson;
  for (unsigned int row = 0; row < 3; ++row)
  {
    for (unsigned int column = 0; column < 3; ++column)
      indexToWorldJson.append(matrix[row][column]);
    indexToWorldJson.append(offset[row]);
  }
  indexToWorldJson.append(0.0);
  indexToWorldJson.append(0.0);
  indexToWorldJson.append(0.0);
  indexToWorldJson.append(1.0);

  QJsonObject options{
    {"bbox_index", bboxJson},
    {"index_to_world", indexToWorldJson},
    {"num_slices", m_NumSlices->value()},
    {"query_budget", m_QueryBudget->value()},
    {"query_chunk_size", 20000},
    {"level", 0.0}};

  auto* multiPart = new QHttpMultiPart(QHttpMultiPart::FormDataType);
  QHttpPart optionsPart;
  optionsPart.setHeader(QNetworkRequest::ContentDispositionHeader, "form-data; name=\"options\"");
  optionsPart.setBody(QJsonDocument(options).toJson(QJsonDocument::Compact));
  multiPart->append(optionsPart);

  auto* file = new QFile(imagePath, multiPart);
  if (!file->open(QIODevice::ReadOnly))
  {
    delete multiPart;
    QMessageBox::critical(nullptr, "MedSAM", "Could not reopen exported NIfTI image.");
    return;
  }
  QHttpPart imagePart;
  imagePart.setHeader(QNetworkRequest::ContentDispositionHeader,
    "form-data; name=\"image\"; filename=\"input.nii.gz\"");
  imagePart.setHeader(QNetworkRequest::ContentTypeHeader, "application/gzip");
  imagePart.setBodyDevice(file);
  multiPart->append(imagePart);

  QNetworkRequest request(QUrl(m_ServerUrl->text() + "/v1/reconstruct"));
  auto* reply = m_Network.post(request, multiPart);
  multiPart->setParent(reply);
  SetBusy(true, "Reconstructing on GPU backend...");
  connect(reply, &QNetworkReply::finished, this,
    [this, reply]() { OnReconstructionFinished(reply); });
}

void QmitkMedSAMReconstructionView::OnReconstructionFinished(QNetworkReply* reply)
{
  if (reply->error() != QNetworkReply::NoError)
  {
    const QString detail = QString::fromUtf8(reply->readAll());
    SetBusy(false, "Reconstruction failed");
    QMessageBox::critical(nullptr, "MedSAM backend", reply->errorString() + "\n" + detail);
    reply->deleteLater();
    return;
  }

  const QString meshPath = m_TempDir.filePath("MedSAM_Reconstruction.ply");
  QFile output(meshPath);
  if (!output.open(QIODevice::WriteOnly) || output.write(reply->readAll()) < 0)
  {
    SetBusy(false, "Could not save returned mesh");
    reply->deleteLater();
    return;
  }
  output.close();

  try
  {
    auto nodes = mitk::IOUtil::Load(meshPath.toStdString(), *GetDataStorage());
    if (nodes.IsNotNull() && nodes->Size() > 0)
    {
      auto node = nodes->GetElement(0);
      auto* surface = dynamic_cast<mitk::Surface*>(node->GetData());
      auto imageNode = m_ImageSelector->GetSelectedNode();
      auto* image = imageNode.IsNotNull() ? dynamic_cast<mitk::Image*>(imageNode->GetData()) : nullptr;
      if (surface == nullptr || image == nullptr || surface->GetVtkPolyData() == nullptr)
        throw std::runtime_error("Returned data is not a surface or the selected CT is unavailable.");

      // The backend deliberately returns vertices in CT continuous-index
      // coordinates. Apply the selected MITK image's own geometry here, after
      // PLY loading, so no NIfTI RAS/LPS or PLY convention can alter placement.
      auto* points = surface->GetVtkPolyData()->GetPoints();
      const auto* imageGeometry = image->GetGeometry();
      for (vtkIdType pointId = 0; pointId < points->GetNumberOfPoints(); ++pointId)
      {
        double raw[3];
        points->GetPoint(pointId, raw);
        mitk::Point3D index;
        index[0] = raw[0];
        index[1] = raw[1];
        index[2] = raw[2];
        mitk::Point3D world;
        imageGeometry->IndexToWorld(index, world);
        points->SetPoint(pointId, world[0], world[1], world[2]);
      }
      points->Modified();
      surface->GetVtkPolyData()->Modified();
      surface->Modified();
      node->SetName("MedSAM implicit reconstruction");
      node->SetColor(0.2f, 0.8f, 0.9f);
      node->SetOpacity(0.65f);
    }
    mitk::RenderingManager::GetInstance()->InitializeViewsByBoundingObjects(GetDataStorage());
    SetBusy(false, QString("Mesh loaded; SDF [%1, %2]")
      .arg(QString::fromUtf8(reply->rawHeader("X-SDF-Min")))
      .arg(QString::fromUtf8(reply->rawHeader("X-SDF-Max"))));
  }
  catch (const std::exception& exc)
  {
    SetBusy(false, "MITK could not load returned PLY");
    QMessageBox::critical(nullptr, "MedSAM", exc.what());
  }
  reply->deleteLater();
}
