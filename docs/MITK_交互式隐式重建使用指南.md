# MedSAM–MITK 交互式隐式网格重建使用指南

本文介绍如何从零启动 MedSAM 推理后端和 MITK Workbench，在 MITK 中加载三维 CT、绘制三维提示框，并将模型重建的网格加载回 CT 的正确世界坐标位置。

## 1. 当前目录约定~~~~

本文按以下已构建环境编写：

```text
项目代码：       E:\MedSAM
模型权重：       E:\MedSAM\result\best_sdf_sam_slice_24_plane.pth
MITK 构建目录：  E:\build\MITK-superbuild\MITK-build
MITK 启动脚本：  E:\build\MITK-superbuild\MITK-build\bin\startMitkWorkbench_release.bat
后端地址：       http://127.0.0.1:8765
```

如果目录不同，请替换命令中的路径。

## 2. 每次使用时的整体流程

每次交互重建需要同时运行两个程序：

1. Python 后端：加载模型、接收 CT 和提示框、预测 SDF 并生成 PLY。
2. MITK Workbench：显示 CT、编辑提示框、调用后端并显示返回网格。

建议始终先启动后端，再启动 MITK。后端运行期间不要关闭其 PowerShell 窗口。

## 3. 启动 Python 推理后端

### 3.1 打开第一个 PowerShell 窗口

进入项目目录：

```powershell
Set-Location "E:\MedSAM"
```

如果使用 conda 或虚拟环境，先激活训练和测试所用的同一个环境。确认模型权重存在：

```powershell
Test-Path "E:\MedSAM\result\best_sdf_sam_slice_24_plane.pth"
```

结果应为 `True`。

### 3.2 启动后端

如果 PowerShell 允许执行脚本：

```powershell
.\run_interactive_backend.ps1 `
  -Checkpoint "E:\MedSAM\result\best_sdf_sam_slice_24_plane.pth" `
  -Device cuda `
  -Port 8765
```

如果出现“禁止运行脚本”，无需修改全局执行策略，可仅对本次启动绕过：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "E:\MedSAM\run_interactive_backend.ps1" `
  -Checkpoint "E:\MedSAM\result\best_sdf_sam_slice_24_plane.pth" `
  -Device cuda `
  -Port 8765
```

也可以直接启动：

```powershell
$env:MEDSAM_INTERACTIVE_CHECKPOINT = "E:\MedSAM\result\best_sdf_sam_slice_24_plane.pth"
$env:MEDSAM_INTERACTIVE_DEVICE = "cuda"
python -m uvicorn interactive_backend.app:app --host 127.0.0.1 --port 8765
```

看到类似下面的信息表示服务已经启动：

```text
Uvicorn running on http://127.0.0.1:8765
Application startup complete
```

首次启动会加载整个 checkpoint，可能需要等待一段时间。

### 3.3 检查后端

另开一个 PowerShell 窗口执行：

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8765/health"
```

正常结果应包含：

```text
status  ready
device  cuda
```

注意：命令中只能填写普通 URL 字符串，不能复制 Markdown 形式的 `[文字](网址)`。

## 4. 启动 MITK Workbench

### 4.1 打开第二个 PowerShell 窗口

MITK 生成的启动脚本内部使用相对路径，因此应先进入它所在的 `bin` 目录：

```powershell
Push-Location "E:\build\MITK-superbuild\MITK-build\bin"
.\startMitkWorkbench_release.bat
Pop-Location
```

也可以使用：

```powershell
cmd.exe /c "cd /d E:\build\MITK-superbuild\MITK-build\bin && call startMitkWorkbench_release.bat"
```

不要优先直接双击 `MitkWorkbench.exe`。启动脚本会同时设置 Qt、MITK 插件、Python 和第三方动态库的搜索路径。

## 5. 打开所需视图

MITK 启动后，打开以下两个视图：

1. 在菜单中选择 `Window/View → Open View`（不同版本菜单文字可能略有差异）。
2. 搜索并打开 `Image Cropper`。
3. 再次打开视图列表，搜索 `MedSAM` 或 `Implicit Reconstruction`，打开 MedSAM 重建视图。

MedSAM 面板应包含以下控件：

```text
Select input CT
Select Bounding Shape ROI
Backend URL
Sparse slices / plane
Query budget
Check backend
Reconstruct mesh
```

如果搜索不到 MedSAM 视图，请检查插件文件是否存在：

```powershell
Test-Path "E:\build\MITK-superbuild\MITK-build\bin\plugins\Release\liborg_mitk_gui_qt_medsamimplicitreconstruction.dll"
```

## 6. 加载三维 CT

可通过以下任一方式加载：

- 将 `.nii` 或 `.nii.gz` 拖入 MITK。
- 使用 `File → Open File`。
- 对 DICOM 序列使用 MITK 的 DICOM Browser 导入完整序列。

加载后确认：

- Data Manager 中出现一个 CT 图像节点。
- 轴位、冠状位、矢状位窗口能正常显示。
- 图像是标量三维图像，而不是 4D 序列或 RGB 图像。
- 三个方向没有明显翻转、错位或异常 spacing。

在 MedSAM 面板的 `Select input CT` 中选择该 CT 节点。

## 7. 创建并调整三维提示框

### 7.1 创建 Bounding Shape

1. 在 Data Manager 中选中 CT 节点。
2. 转到 `Image Cropper` 视图。
3. 在 `Image` 中确认选择的是当前 CT。
4. 在 `Bounding Box` 一栏点击 `New`。
5. Data Manager 中会出现一个 `GeometryData` 类型的 Bounding Shape 节点。

这里只借用 Image Cropper 创建和编辑提示框，不需要执行 `Crop` 或 `Mask`。

### 7.2 调整提示框

在二维切片窗口和三维窗口中拖动提示框的控制柄，使其包围目标器官。建议：

- 框完整覆盖器官，不要截断边界。
- 在器官外保留少量背景，让模型获得内外符号信息。
- 不要将框扩大到几乎整个 CT，否则 query 分辨率下降且显存、耗时增加。
- 同时检查轴位、冠状位和矢状位，确保三维范围正确。
- 当前后端最终使用的是包围该 GeometryData 八个角点的轴对齐索引框；旋转提示框通常不会带来旋转 ROI 的效果，因此优先通过平移和缩放调整。

完成后，在 MedSAM 面板的 `Select Bounding Shape ROI` 中选择这个 Bounding Shape 节点。

## 8. 设置推理参数

### Backend URL

本机默认填写：

```text
http://127.0.0.1:8765
```

末尾不要添加 `/health` 或 `/v1/reconstruct`，插件会自动补充接口路径。

### Sparse slices / plane

表示在 XY、XZ、YZ 三个方向各抽取多少张稀疏切片。

- 首次验证：`8`
- 边界复杂或跨度较大的器官：可尝试 `12`、`16` 或训练时使用的数量
- 上限：`32`

该值越大，跨 slice 信息越充分，但显存占用和推理时间也会增加。推理设置最好与训练时的 slice 数量和采样逻辑一致。

### Query budget

控制 ROI 内粗 SDF 网格的总采样预算：

- 快速测试：`50000`～`100000`
- 正常重建：从 `100000` 开始
- 更细表面：逐步提高，但不要一次升到很大

预算越大，表面细节通常越好，同时推理更慢、显存占用更高。模型内部每次按 `20000` 个 query 分块推理。

## 9. 执行交互式重建

1. 点击 `Check backend`。
2. 状态变为 `Backend ready` 后，确认 CT 和 Bounding Shape 都已选中。
3. 点击 `Reconstruct mesh`。
4. 状态会显示 `Reconstructing on GPU backend...`。
5. 推理期间不要关闭后端窗口，也不要重复点击按钮。

成功后：

- 后端返回 PLY 网格。
- MITK 自动将其加入 Data Manager。
- 节点名称为 `MedSAM implicit reconstruction`。
- 默认颜色为青色，透明度约为 `0.65`。
- 状态栏显示预测 SDF 的最小值和最大值。

如果 SDF 最小值小于 `0` 且最大值大于 `0`，说明预测场穿过了零水平集，可以执行 marching cubes。若全正或全负，后端会拒绝生成伪网格并报告 `TSDF does not cross level 0.0`。

## 10. 为什么网格能回到正确空间位置

空间转换链如下：

```text
MITK ROI 世界坐标
    → CT WorldToIndex
    → bbox_index [x0,y0,z0,x1,y1,z1]
    → 后端在原 CT 索引空间抽取 XY/XZ/YZ 切片和 query
    → marching cubes 得到索引空间顶点
    → 后端以原始 CT 索引坐标写入 PLY
    → PLY 加载回 MITK
    → 插件逐顶点调用当前 CT geometry 的 IndexToWorld
```

插件不是把框的屏幕像素坐标直接发送给模型，而是先用 CT geometry 做 `WorldToIndex`。后端返回的 PLY 顶点保持为 CT 索引坐标；插件加载 PLY 后，对每个顶点调用当前 CT 自身的 `IndexToWorld` 转换。这使提示框的 `WorldToIndex` 和网格的 `IndexToWorld` 构成同一个 MITK geometry 下的互逆过程，也避免了 NIfTI RAS/LPS 与无坐标元数据 PLY 之间的约定歧义。

为保持这种对应关系：

- 不要在 MITK 导出后、后端读取前手工修改 NIfTI header 或 affine。
- 不要只 resize 图像而不对坐标执行完全一致的变换。
- 训练和推理必须使用相同的轴顺序、归一化、slice 位置及 box 定义。
- 若对 CT 做重采样，应让图像 geometry/affine 同步更新。

## 11. 检查和保存结果

### 检查空间对齐

1. 在 Data Manager 中保留 CT 和重建网格可见。
2. 在三个二维切片窗口移动十字光标。
3. 检查网格轮廓是否位于对应器官边缘附近。
4. 在三维窗口旋转观察是否出现整体平移、轴交换、镜像或尺度错误。

如果网格形状基本正确但整体错位，优先检查 affine、spacing、origin、direction 和 XYZ/ZYX 转换，而不是继续训练模型。

### 保存网格

在 Data Manager 中右键 `MedSAM implicit reconstruction`，选择保存并导出为 MITK 支持的表面格式，例如 `.ply` 或 `.stl`。插件收到的临时 PLY 位于临时目录，会随 MITK 会话清理，因此需要保留结果时应主动保存。

## 12. 常见问题

### Check backend 显示连接失败

依次检查：

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8765/health"
Get-NetTCPConnection -LocalPort 8765 -ErrorAction SilentlyContinue
```

确认后端窗口仍在运行、端口一致，并且 Backend URL 没有多余路径。

### 后端启动时找不到模块

确认从 `E:\MedSAM` 启动，并使用安装了项目依赖的 Python：

```powershell
Set-Location "E:\MedSAM"
python -c "import torch, nibabel, scipy, skimage, trimesh, fastapi, uvicorn; print('OK')"
```

### CUDA out of memory

优先按顺序处理：

1. 缩小 Bounding Shape，但仍完整覆盖器官。
2. 降低 `Query budget`。
3. 降低 `Sparse slices / plane`。
4. 关闭占用 GPU 的其他程序后重启后端。

`query_chunk_size` 当前固定为 `20000`。若仍然 OOM，可以在插件或后端默认参数中进一步降低该值。

### TSDF 全正或全负

错误通常类似：

```text
TSDF does not cross level 0.0: min=..., max=...
```

这表示 ROI 内没有预测到零水平面。检查：

- 框是否真的覆盖目标器官并包含少量背景。
- 模型 checkpoint 是否与当前代码结构和预处理一致。
- 训练与推理的坐标范围、SDF 正负号和 truncation 定义是否一致。
- `Sparse slices / plane` 是否与训练设置相近。

不要仅通过改变 marching-cubes level 掩盖全正或全负问题；这通常是模型或数据坐标逻辑问题。

### 网格是方块、有孔洞或贴着 ROI 边界

检查 SDF 的正负范围和零面是否真实存在，并提高 query budget 做对照。若网格长期贴框，常见原因是边界填充值、SDF 饱和区、坐标归一化或训练标签符号不一致，而不只是 marching cubes 分辨率不足。

### MITK 找不到 MedSAM 视图

确认使用的是构建目录中的启动脚本，并检查 DLL：

```powershell
Test-Path "E:\build\MITK-superbuild\MITK-build\bin\plugins\Release\liborg_mitk_gui_qt_medsamimplicitreconstruction.dll"
```

若修改过插件源码，需要重新编译：

```powershell
$env:CL = "/utf-8"
cmake --build "E:\build\MITK-superbuild\MITK-build" `
  --config Release `
  --target org_mitk_gui_qt_medsamimplicitreconstruction `
  --parallel 4
```

然后完全退出并重新启动 MITK。

## 13. 推荐的最短日常操作清单

```text
1. 在 E:\MedSAM 启动 Python 后端
2. 用 /health 确认 status=ready、device=cuda
3. 从 MITK bin 目录运行 startMitkWorkbench_release.bat
4. 加载三维 CT
5. 打开 Image Cropper，选择 CT，点击 Bounding Box → New
6. 在三个切面中调整提示框，使其完整覆盖器官并保留少量背景
7. 打开 MedSAM 视图，选择 CT 和 Bounding Shape
8. 点击 Check backend
9. 从 8 slices/plane、100000 queries 开始
10. 点击 Reconstruct mesh
11. 在三切面和 3D 视图检查空间位置
12. 右键保存重建网格
```
