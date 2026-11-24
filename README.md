# GeoMono

GeoMono 用 GCDID 从光度监督中排除运动实例，再用 GCIDC 为其中有几何依据的像素生成逆深度监督。

## 运行

需要 NVIDIA GPU、CUDA Toolkit（含 `nvcc`），以及 Python 3.8 和 3.10。深度训练和 WAFT 光流各用一个 Python 环境。

```bash
bash setup_env.sh
bash run_train.sh
```

在用默认配置启动前，请准备以下输入。

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

启动脚本自动选择空闲显存最多的 GPU。一次遍历同时生成 GCDID 运动掩码、GCIDC 道路掩码和四向光流缓存。两项方法共用道路像素的读取、合并与采样以及相邻帧的双向 WAFT 光流。GCDID 还利用首尾帧光流判定运动。全部缓存完成后自动启动训练。首次运行在本地编译 CUDA 算子。相同配置的完整缓存可复用，中断后以相同命令重新启动可接续缓存构建。

## 动态实例掩码

GCDID（Ground-Constrained Dynamic Instance Detection，地面约束动态实例检测）默认处理机动车、行人、自行车、摩托车及骑行者。实例类别由图像候选掩码提供。每个候选都用同一个静止重投影检验，判定运动后，其可见像素从目标帧的光度监督中排除。

从相机光学内参、可见路面和背景光流估计相机运动。两侧图像共用中心帧中同一背景点的逆深度，两次相机运动与路面平面一起求解。同一车身或目标上的竖直列在两个时间方向上共享逆深度。目标下轮廓的每一段直边由可见路面限定接触区间。轮间轮廓沿下包络连接，缺少路面观测的直边使用单侧约束。即使一个实例没有直接可见接触列，只要场景位姿和实例光流有证据，也可检出。整帧路面不足以估计位姿时保留该帧的候选。

用静止重投影预测首尾两帧间的位移，与实测双向光流比较。将光流沿三帧回投到中心帧，按回投像素落在原实例掩码内的比例计入首尾位移证据。像素残差按静止平移视差、往返光流误差和路面重投影误差归一化。动静分界由无标签运动分数的最大类间方差划分得到，已随项目配置。运动掩码的生成依赖图像及光学内参，独立于训练中深度和位姿网络的预测。 [实例判决](gcdid/ground/detector.py) · [相机运动与路面估计](gcdid/ground/visual_pose.py) · [判决配置](gcdid/ground/calibration.json)

## 训练配置

| 项目 | 默认设置 |
|---|---|
| 数据与划分 | KITTI Raw，Eigen Zhou 训练划分 |
| 训练分辨率 | 640 × 192 |
| Batch size / epoch | 12 / 20 |
| 训练帧 / 尺度 | 0、−1、+1 / 尺度 0 |
| 编码器初始化 | MPViT-Small 和 ResNet-18 权重 |
| 优化器 | AdamW，初始学习率 0.0001，权重衰减 0.01 |
| 学习率调度 | ExponentialLR，每轮乘以 0.9 |
| 后处理与评估 | 启用 `--post_process` 和 `--no_eval` |
| 数据目录 | `data/kitti/` |
| 实例候选 | `candidates/kitti_dynamic_mask/` |
| 路面分割 | `data/road_masks/` |
| 缓存 | `cache/geometry/` 下的 GCDID 掩码、GCIDC 道路掩码及四向光流 |
| 训练输出 | `tmp/<实验名>/models/weights_*` |

默认保存每轮的训练权重，日志点会使用验证批次写 TensorBoard。`--no_eval` 控制每轮末的完整深度评估。训练数据加载器只读取训练需要的图像和几何缓存。

## 逆深度约束

GCIDC（Ground Contact Inverse Depth Constraint，地面接触逆深度约束）在 GCDID 选中的实例底缘寻找可观测接触。目标帧和前后相邻帧的双向光流、路面拟合残差和像素边界共同确定接触逆深度区间。整辆车找不到可信接地列时保留原预测。已有区间支撑且需要校正的像素投影到最近边界。同一接地实例的其他可观测像素沿拟合的接触轮廓扩展，校正幅度由接触观测区间和留一拟合误差自动计算。

同一张逆深度监督图用于计算 GCIDC 损失。光度损失先对每张图的有效像素取平均，再在批次内求和。GCIDC 先对每张图有目标像素的相对逆深度误差取平均，再在批次内取平均，并以固定权重 `0.1` 加入总损失。GCIDC 目标停止梯度，无目标的图像在该项贡献零。 [目标与像素损失](gcidc.py) · [训练聚合](trainer.py)

缓存准备还会用已有实例标签和三帧光流检查分开的骑行者与自行车或摩托车。人体与车轮位置吻合，且交界在前后帧保持邻接时，两个掩码共用一个实例编号。GCIDC 从车轮侧取可信接地证据，为整个人车实例生成目标。融合后的实例网格和训练索引写在原来的四向光流记录里，训练从同一缓存自动读取。 [人车配对](gcdid/riders.py)

WAFT 光流、运动判决、道路和实例掩码的采样、GCIDC 实例索引整理、几何估计及两项方法的训练损失均在 CUDA 上计算。数据加载进程读取和解压缓存，计算在训练进程选定的 GPU 上执行。GCIDC 缓存的四向光流依次为目标帧到前帧、前帧到目标帧、目标帧到后帧、后帧到目标帧。训练发生水平翻转时同步变换掩码、坐标及光流水平分量。

## 查看训练与评估

```bash
bash tensorboard.sh 6006
```

`gcdid/panel_N` 同时显示目标图像、候选、运动掩码、光度损失排除区和光度误差。`gcidc/panel_N` 显示同一样本的选区、预测深度、几何目标和约束方向。单图深度推理只需要深度网络。可用 `bash evaluate.sh <权重目录> <GPU编号>` 对已保存的权重做独立深度评估。评估前可运行 `.venv_train/bin/python export_gt_depth.py --data_path data/kitti --split eigen` 生成评估真值。


缓存构建和自监督训练使用图像、图像派生掩码与光流、相机光学内参与模型预测。官方轨迹、OXTS 和 LiDAR 由独立评分流程读取。默认训练参数包含 `--no_eval`。

## 许可

Monodepth2 部分保留原始许可及署名。WAFT 和相关特征提取代码的许可随源码分发。 [Monodepth2 许可](LICENSE) · [WAFT 许可](third_party/WAFT/LICENSE) · [特征提取代码许可](third_party/WAFT/thirdparty/DepthAnythingV2/LICENSE)
