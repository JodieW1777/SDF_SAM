import numpy as np
import nibabel as nib

# 填写你的nii文件路径
nii_path = r"E:\MedSAM\data_test\FLARE22_Tr_0002_0000.nii.gz"
# 你从Slicer导出的ROI RAS边界
ras_bounds = [-327.2622618804106, -218.14057015083935, -298.1088912551468, -219.79803013157198, 54.86060366213076, 178.87227597654112]

# 加载nii，自动获取空间信息
nii_img = nib.load(nii_path)
affine = nii_img.affine
origin = affine[:3, 3]
spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))

rxmin, rxmax, rymin, rymax, rzmin, rzmax = ras_bounds
corners_ras = np.array([
    [rxmin, rymin, rzmin],
    [rxmax, rymin, rzmin],
    [rxmin, rymax, rzmin],
    [rxmax, rymax, rzmin],
    [rxmin, rymin, rzmax],
    [rxmax, rymin, rzmax],
    [rxmin, rymax, rzmax],
    [rxmax, rymax, rzmax],
])

ijk_all = []
for rx, ry, rz in corners_ras:
    i = (rx - origin[0]) / spacing[0]
    j = (ry - origin[1]) / spacing[1]
    k = (rz - origin[2]) / spacing[2]
    ijk_all.append([i, j, k])

ijk_arr = np.array(ijk_all)
# 计算I/J/K三个维度的min/max（整数化）
i_min = int(np.floor(ijk_arr[:,0].min()))
i_max = int(np.ceil(ijk_arr[:,0].max()))
j_min = int(np.floor(ijk_arr[:,1].min()))
j_max = int(np.ceil(ijk_arr[:,1].max()))
k_min = int(np.floor(ijk_arr[:,2].min()))  # 新增Z轴（K）最小值
k_max = int(np.ceil(ijk_arr[:,2].max()))  # 新增Z轴（K）最大值

# 保留原2D BOX（兼容旧逻辑）
GLOBAL_BOX = [i_min, j_min, i_max, j_max]
# 新增3D BOX（包含I/J/K的完整范围）
GLOBAL_BOX_3D = [i_min, j_min, k_min, i_max, j_max, k_max]

# 打印输出（清晰展示各维度范围）
print(f"原始2D GLOBAL_BOX (i_min,j_min,i_max,j_max) = {GLOBAL_BOX}")
print(f"新增3D GLOBAL_BOX (i_min,j_min,k_min,i_max,j_max,k_max) = {GLOBAL_BOX_3D}")
print(f"单独输出Z轴(K)范围：k_min={k_min}, k_max={k_max}")


'''
volumeName = "R"
scene = slicer.app.mrmlScene()
volNode = scene.GetFirstNodeByName(volumeName)
roiNode = scene.GetFirstNodeByClass("vtkMRMLMarkupsROINode")

if volNode and roiNode:
    bounds = [0]*6
    roiNode.GetBounds(bounds)
    print("ROI RAS坐标边界(xmin,xmax,ymin,ymax,zmin,zmax):")
    print(bounds)
    print("三维像素范围：i_min,j_min,k_min,i_max,j_max,k_max =", GLOBAL_BOX_3D)
'''