# GeoMono

GeoMono 用 GCDID 从光度监督中排除运动实例，再用单帧 GCIDC 为通过接地检查的实例生成整掩码逆深度监督。

## 运行

需要 NVIDIA GPU、CUDA Toolkit（含 `nvcc`），以及 Python 3.8 和 3.10。深度训练和 WAFT 光流各用一个 Python 环境。

```bash
bash run_train.sh
```

可在命令后依次提供实验名和 GPU 编号。省略 GPU 编号时，脚本选择空闲显存最多的 GPU。

启动脚本从以下位置查找输入，并在需要时创建 Python 环境和几何缓存。

| 输入 | 默认位置 |
|---|---|
| KITTI Raw 整流图像和相机光学标定 | `data/kitti/` |
| 由图像分割生成的实例候选 | `candidates/kitti_dynamic_mask/` |
| 由图像分割生成的道路与标线掩码 | `data/road_masks/` |
| MPViT-Small 初始化权重 | `models/mpvit_small.pth` |
| WAFT-a1 的 tar-c-t 光流权重 | `models/waft/tar-c-t.pth` |

初始化文件应与相应模型架构匹配。MPViT 文件包含 `model` 状态字典。ResNet-18 初始化由 torchvision 下载或从 `models/resnet18-f37072fd.pth` 读取。WAFT 将其特征提取器的中间特征用于光流，GCIDC 的深度输入来自 GeoMono 自己的网络。 [MonoViT](https://github.com/zxcqlf/MonoViT) · [WAFT](https://github.com/princeton-vl/WAFT)

可用 `GCDID_DATA`、`GCDID_CANDIDATES` 和 `GCDID_ROAD_MASKS` 指向已有输入。可用 `GEOMONO_TRAIN_PYTHON` 和 `GEOMONO_FLOW_PYTHON` 指定两个 Python 可执行文件。KITTI Raw 的 JPEG 图像按日期、行程及 `image_02/data` 或 `image_03/data` 放置。 [KITTI Raw](https://www.cvlibs.net/datasets/kitti/raw_data.php)

每帧掩码以 `<日期>__<行程>__<10 位帧号>__<l 或 r>.npz` 命名。候选文件提供 `labels`、`height`、`width`、`boxes`、`masks_off` 和 `masks_bits`。候选框坐标为包含端点的 `x0, y0, x1, y1`。每个裁剪掩码按行展开，以最高位优先的顺序打包。`masks_off` 记录各掩码在字节数组中的起始偏移。道路文件提供 `labels`、`image_height`、`image_width` 和整图 `masks`。标签名称见加载代码。 [候选实例](gcdid/instances.py) · [道路掩码](gcdid/ground/settings.py)

缓存准备在一次遍历中生成 GCDID 运动掩码和目标帧道路掩码。WAFT 由 GCDID 离线调用来判定运动。训练进程读取运动掩码和道路掩码。首次运行在本地编译 CUDA 算子，相同配置下重跑启动命令可复用已有缓存。

## 动态实例掩码

GCDID（Ground-Constrained Dynamic Instance Detection，地面约束动态实例检测）默认处理机动车、行人、自行车、摩托车及骑行者。实例类别由图像候选掩码提供。每个候选都用同一个静止重投影检验，判定运动后，其可见像素从目标帧的光度监督中排除。

从相机光学内参、可见路面和背景光流估计相机运动。两侧图像共用中心帧中同一背景点的逆深度，两次相机运动与路面平面一起求解。同一车身或目标上的竖直列在两个时间方向上共享逆深度。目标下轮廓的每一段直边由可见路面限定接触区间。轮间轮廓沿下包络连接，缺少路面观测的直边使用单侧约束。即使一个实例没有直接可见接触列，只要场景位姿和实例光流有证据，也可检出。整帧路面不足以估计位姿时保留该帧的候选。

用静止重投影预测首尾两帧间的位移，与实测双向光流比较。将光流沿三帧回投到中心帧，按回投像素落在原实例掩码内的比例计入首尾位移证据。像素残差按静止平移视差、往返光流误差和路面重投影误差归一化。首次构建缓存时，在训练集的无标签图像上选择使两类分数间方差最大的划分，两类均值的中点是该数据集的动静分界。标定按分割器的原生实例取样，限定在候选前景内。训练掩码保留人车融合后的实例。三帧光流在这两个步骤间复用。项目随附 Eigen-Zhou 训练集的参考图像清单，分界由 CUDA 自动估计并写入本地缓存。预测依赖图像及光学内参，独立于训练中的深度和位姿预测。 [实例证据](gcdid/ground/detector.py) · [无标签标定](gcdid/ground/reference.py) · [训练参考图像](splits/eigen_zhou/motion_reference.txt)

## 训练配置

| 项目 | 默认设置 |
|---|---|
| 数据与划分 | KITTI Raw，Eigen Zhou 训练划分 |
| 训练分辨率 | 640 × 192 |
| Batch size / epoch | 12 / 20 |
| 训练帧 / 尺度 | 光度重投影使用 0、−1、+1，单帧 GCIDC 使用 0，均为尺度 0 |
| 编码器初始化 | MPViT-Small 和 ResNet-18 权重 |
| 优化器 | AdamW，初始学习率 0.0001，权重衰减 0.01 |
| 学习率调度 | ExponentialLR，每轮乘以 0.9 |
| 后处理与评估 | 启用 `--post_process` 和 `--no_eval` |
| 数据目录 | `data/kitti/` |
| 实例候选 | `candidates/kitti_dynamic_mask/` |
| 路面分割 | `data/road_masks/` |
| 缓存 | `cache/geometry/` 下的 GCDID 掩码和目标帧道路掩码 |
| 训练输出 | `tmp/<实验名>/models/weights_*` |

默认保存每轮的训练权重，日志点会使用验证批次写 TensorBoard。`--no_eval` 控制每轮末的完整深度评估。训练数据加载器只读取训练需要的图像和几何缓存。

## 逆深度约束

GCIDC（Ground Contact Inverse Depth Constraint，地面接触逆深度约束）用第 0 帧的背景深度拟合路面。在 GCDID 选中的实例底缘，路面掩码和拟合残差为接地列提供观测。算法从接地列及下轮廓估计逆深度，沿可见支撑线延伸到悬空边缘，并利用画面下缘处理截断列。每个通过接地检查的实例在整个可见掩码内都有逐像素点目标。

同一张逆深度监督图用于计算 GCIDC 损失。光度损失先对每张图的有效像素取平均，再在批次内求和。GCIDC 先对每张图有目标像素的相对逆深度误差取平均，再在批次内取平均，并以固定权重 `0.1` 加入总损失。GCIDC 目标停止梯度，无目标的图像在该项贡献零。 [目标与像素损失](gcidc.py) · [训练聚合](trainer.py)

掩码采样、运动判决、GCIDC 目标生成与损失均在 CUDA 上计算。训练发生水平翻转时，目标帧深度、实例网格和道路掩码使用同一朝向。

在 384 个 KITTI 留出帧上，用同一组掩码对比固定检查点的几何目标。下表统计道路运动实例上的 LiDAR 点。

| 检查点 | 目标覆盖率 GCP → GCIDC | δ₁ GCP → GCIDC | AbsRel GCP → GCIDC |
|---|---:|---:|---:|
| `weights_0` | 48.83% → 50.47% | 84.67% → 84.58% | 13.79% → 13.53% |
| `weights_5` | 49.66% → 52.72% | 91.24% → 90.25% | 12.19% → 11.65% |

对于 `weights_5`，触及画面下缘的新增点 AbsRel 从 6.60% 升至 20.08%。其余新增点从 10.17% 降至 8.71%。

## 查看训练与评估

```bash
bash tensorboard.sh 6006
```

`gcdid/panel_N` 同时显示目标图像、候选、运动掩码、光度损失排除区和光度误差。`gcidc/panel_N` 显示同一样本的选区、预测深度、几何目标和约束方向。单图深度推理只需要深度网络。可用 `bash evaluate.sh <权重目录> <GPU编号>` 对已保存的权重做独立深度评估。评估前可运行 `.venv_train/bin/python export_gt_depth.py --data_path data/kitti --split eigen` 生成评估真值。


缓存构建和自监督训练使用图像、图像派生掩码与光流、相机光学内参与模型预测。官方轨迹、OXTS 和 LiDAR 由独立评分流程读取。默认训练参数包含 `--no_eval`。

## 许可

Monodepth2 部分保留原始许可及署名。WAFT 和相关特征提取代码的许可随源码分发。 [Monodepth2 许可](LICENSE) · [WAFT 许可](third_party/WAFT/LICENSE) · [特征提取代码许可](third_party/WAFT/thirdparty/DepthAnythingV2/LICENSE)
