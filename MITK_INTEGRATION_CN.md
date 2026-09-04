# MedSAM 隐式网格重建接入 MITK

本接入由两个进程组成：

1. `interactive_backend`：Python/PyTorch 常驻 GPU 服务，接收 NIfTI 和三维框，返回世界坐标 PLY 网格。
2. `org.mitk.gui.qt.medsamimplicitreconstruction`：MITK Workbench C++/Qt 视图，读取 MITK 中的 CT 与 Bounding Shape，通过 HTTP 调用后端，并把返回网格加入 Data Manager。

这种拆分避免在 MITK 进程中同时加载 Qt/VTK、PyTorch、CUDA 和 cuDNN DLL。官方 nnInteractive 的远程模式同样使用常驻服务承载模型。

## 一、准备 Python 后端

在项目当前 Python 环境中安装新增依赖：

```powershell
cd E:\MedSAM
pip install -r interactive_backend\requirements.txt
```

确认训练权重对应当前代码结构。默认权重路径是：

```text
E:\MedSAM\result\best_sdf_sam_slice_24_plane.pth
```

启动服务：

```powershell
cd E:\MedSAM
.\run_interactive_backend.ps1 `
  -Checkpoint result\best_sdf_sam_slice_24_plane.pth `
  -Device cuda `
  -Port 8765




$env:MEDSAM_INTERACTIVE_CHECKPOINT = (Resolve-Path "result\best_sdf_sam_slice_24_plane.pth").Path $env:MEDSAM_INTERACTIVE_DEVICE = "cuda"
python -m uvicorn interactive_backend.app:app --host 127.0.0.1 --port 8765
```

另开 PowerShell 检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8765/health
```

应返回：

```json
{"status":"ready","device":"cuda"}
```

如果 MITK 与 GPU 服务不在同一台机器，将启动参数中的 `--host 127.0.0.1` 改为受控网卡地址，并在防火墙中只允许 MITK 工作站访问。当前示例未实现鉴权和 TLS，不要直接暴露到公网。

## 二、构建 MITK 插件

推荐使用与 MITK Workbench **相同 release tag** 的 [MITK ProjectTemplate](https://github.com/MITK/MITK-ProjectTemplate)。二进制版 Workbench 不能直接加载用不同 ABI/Qt 版本编译的插件。

### 1. 准备源码

```powershell
git clone https://github.com/MITK/MITK.git E:\src\MITK
git clone https://github.com/MITK/MITK-ProjectTemplate.git E:\src\MedSAM-MITK
```

将两者切换到相同 release，例如实际安装的是 MITK 2026.06，就使用相应的 `v2026.06` tag。不要混用 `develop` 与 release 二进制。

复制本项目插件目录：

```powershell
Copy-Item `
  -Recurse `
  E:\MedSAM\mitk_plugin\Plugins\org.mitk.gui.qt.medsamimplicitreconstruction `
  E:\src\MedSAM-MITK\Plugins\
```

### 2. 注册插件

在 ProjectTemplate 顶层 `CMakeLists.txt` 的 `MITK_PLUGINS` 列表中加入：

```cmake
org.mitk.gui.qt.medsamimplicitreconstruction:ON
```

如果模板使用单独的 `Plugins/Plugins.cmake`，则把同一项加入该文件的插件列表。插件目录名必须与上面的 symbolic name 完全一致。

### 3. 配置和编译

Windows 示例：

```powershell
cmake --S E:\src\MITK\MITK-master `
  -B E:\build\MITK-superbuild `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -DMITK_EXTENSION_DIRS:PATH=E:\src\MedSAM-MITK `
  -DDOXYGEN_EXECUTABLE:FILEPATH="E:\Program Files\doxygen\bin\doxygen.exe" `
  -DQt6_DIR:PATH="E:\Qt\6.11.2\msvc2022_64\lib\cmake\Qt6" `
  -DOPENSSL_ROOT_DIR:PATH="E:\OpenSSL-Win64"

cmake --build E:\build\MITK-superbuild --config Release --parallel
```

MITK ProjectTemplate 的具体顶层入口随 release 略有差异；优先按照该 tag 自带的 `README.md` 配置。如果已经有可运行的 MITK source build，只需把插件加入对应自定义 application 的 `MITK_PLUGINS` 后重新配置 CMake。

编译完成后，从该 build tree 启动生成的 Workbench。不要只复制一个 DLL 到官方安装目录，因为 CTK 插件还需要 manifest、plugin.xml 和匹配依赖。

## 三、在 MITK 中交互重建

1. 先启动 Python 后端，等待模型加载完成。
2. 启动包含本插件的 MITK Workbench。
3. 在 MITK 中打开原始 `.nii` 或 `.nii.gz` CT。
4. 打开 `View -> Image Cropper`。
5. 在 Data Manager 选中 CT，在 Image Cropper 中点击 `New` 创建立方体 Bounding Shape。
6. 在轴位、冠状位、矢状位窗口拖动框面上的红色 handles；拖动框本体可平移。框应包含器官并留出约 10 voxel 外部背景。无需点击 `Crop` 或 `Mask`。
7. 打开 `View -> Segmentation -> MedSAM Implicit Reconstruction`。
8. 第一个选择器选择原始 CT，第二个选择器选择刚建立的 Bounding Shape 节点。
9. Backend URL 保持 `http://127.0.0.1:8765`，点击 `Check backend`。
10. 建议先使用 `8` slices/plane 和 `100000` query budget，点击 `Reconstruct mesh`。
11. 后端完成后，`MedSAM implicit reconstruction` Surface 节点会自动出现在 Data Manager，并在 3D render window 中显示。

MITK 官方 Image Cropper 对 Bounding Shape 的创建和拖动说明见：
[MITK Image Cropper 文档](https://docs.mitk.org/nightly/org_mitk_views_imagecropper.html)。

## 四、空间坐标如何保持一致

- 插件不把 MITK world/LPS/RAS 数值直接发送给 Python。
- 插件把 Bounding Shape 的 8 个世界坐标角点通过当前 CT 的 `WorldToIndex()` 转为连续图像 index。
- 后端以 `[x0,y0,z0,x1,y1,z1]` index bbox 构造三视图切片和 query。
- marching cubes 顶点先从局部 `[z,y,x]` 恢复为全图 `[x,y,z]` index，再通过上传 NIfTI 的 affine 转回 world coordinates。
- 返回 PLY 已经在 CT 的世界空间中，因此 MITK 加载后无需手动平移、缩放或翻转。

这套流程要求插件上传的 NIfTI 保留 MITK 图像几何；代码使用 `mitk::IOUtil::Save()`，不经过截图或 DICOM 窗宽窗位图像。

## 五、参数与排错

### `TSDF does not cross level 0`

模型在框内输出没有同时出现正负值。通常表示 checkpoint 尚未学到有效零等值面，或框与训练时差异太大。后端会拒绝生成伪网格。

### 网格被截断

扩大 Bounding Shape，确保器官与框每个面之间都有背景。模型训练使用约 10 voxel padding。

### 网格位置正确但表面粗糙

提高 query budget，例如从 `100000` 增加到 `300000`。这提高空间采样密度，也会增加推理时间。

### CUDA OOM

后端默认每次预测 20000 个 query。可在插件源码的 JSON 参数中降低 `query_chunk_size`，例如改为 5000。图像特征目前会在每个 query chunk 重算，因此较小 chunk 会更慢但更省显存。

### 插件列表中没有该 View

检查：

- 插件是否加入 `MITK_PLUGINS`；
- `plugin.xml` 是否被打包；
- 是否从同一个 build tree 启动 Workbench；
- MITK、Qt、MSVC 的版本和配置（Release/Debug）是否一致。

### ROI 选择器为空

必须先通过 Image Cropper 的 `New` 创建 Bounding Shape。它在 Data Manager 中是 `GeometryData`，不是 Image 或 Surface。

## 六、当前交互边界

当前版本实现一次 3D 框提示对应一次完整网格重建。移动框后再次点击 `Reconstruct mesh` 即可产生新结果。尚未实现：

- 正负点、scribble、lasso；
- 请求取消与进度流；
- session 级图像特征缓存；
- 服务端鉴权；
- 自动替换上一张网格。

若要达到 nnInteractive 那种拖动/点击后近实时更新，下一步应把 API 改成 `create session -> upload image/cache features -> update bbox -> decode mesh`，从而避免每次重新上传 CT 和编码全部 sparse slices。
