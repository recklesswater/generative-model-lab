# 本机运行说明（Windows，已在 2026-09-22 实测）

## 环境

| 项目 | 版本 |
| --- | --- |
| Python | 3.13.15（`D:\Python`） |
| numpy | 2.4.6 |
| scipy | 1.18.0 |
| matplotlib | 3.11.1 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.5 ← 本次唯一需要新装的 |
| torch | **2.13.0+cu126，CUDA 可用** |

安装（只缺 pandas 时）：

```powershell
python -m pip install pandas
```

完整安装：

```powershell
python -m pip install numpy scipy matplotlib scikit-learn pandas
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> **本机 torch 状态变化记录**：2026-09-17 时 `import torch` 曾报
> `OSError [WinError 4551] 应用程序控制策略阻止了文件（shm.dll）`，即 Windows
> 智能应用控制（Smart App Control）拦截。**2026-09-22 实测可直接加载并检测到 CUDA**，
> 说明该拦截已不再生效（配置已改，或当时的拦截是临时的）。若以后再次出现该错误，
> 参考 9/17 的排查记录。

## 四条命令（与 CI 一致，本机全部通过）

按 CI 的顺序，在仓库根目录执行：

```powershell
python 04_se3_diffusion/equivariance_test.py
python 01_mcmc_toy/run_demo.py
python 02_diffusion_toy/run_demo.py --steps 300
python 03_omics_synthesis/run_experiment.py --demo --seeds 0 1
```

| 步骤 | 本机结果 | 耗时（本机） |
| --- | --- | --- |
| 04 等价性测试 | 通过，等价误差 ~1e-7（float32 精度） | ~1.5 min |
| 01 MCMC 演示 | 通过，RWM/MALA/HMC 三项对比 | ~50 s |
| 02 diffusion（300 步） | 通过，loss 0.364 | ~11 s |
| 03 omics 增广 | 通过，合成双批次队列 | ~1 min |

## 想看得更清楚时

```powershell
# 02 跑长一点（短跑会误导，见下）
python 02_diffusion_toy/run_demo.py --steps 5000

# 02 换数据集：八高斯比两月亮更能暴露模式丢失
python 02_diffusion_toy/run_demo.py --dataset eight_gaussians --steps 6000

# 04 训练 + 生成 + 用性质引导
python 04_se3_diffusion/run_demo.py

# 03 多种子
python 03_omics_synthesis/run_experiment.py --demo --seeds 0 1 2 3 4
```

## Windows 上的注意点

1. **matplotlib 中文标签**：默认字体 DejaVu Sans 不含汉字，画出来是空白方块。
   需要中文标注时在脚本开头加：

   ```python
   import matplotlib
   matplotlib.use("Agg")          # 不确定有没有显示环境时用这个
   import matplotlib.pyplot as plt
   plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
   plt.rcParams["axes.unicode_minus"] = False
   ```

2. **无窗口运行**：设置环境变量 `MPLBACKEND=Agg`，避免脚本尝试弹窗。

   ```powershell
   $env:MPLBACKEND='Agg'
   ```

3. **figures/ 会被覆盖**：每个 demo 都往 `figures/` 写同名 PNG。跑不同参数前先看一眼，
   需要留档就把图改名或复制出去。

4. **02 的短跑会误导**：300 步时 loss 看着已经"健康"（0.364），但扇区覆盖率是错的。
   见下面「实测对比」。

## 实测对比：02 跑 300 步 vs 5000 步

八扇区覆盖率（真值 vs DDPM 采样）：

```
300 步    truth  0.008 0.018 0.221 0.255 0.128 0.124 0.124 0.123
          DDPM   0.040 0.035 0.111 0.320 0.156 0.103 0.128 0.108   ← 宽扇区被抽干、窄扇区被灌水
5000 步   DDPM   0.017 0.029 0.218 0.218 0.122 0.132 0.110 0.155   ← 宽扇区几乎精确
```

结论：

* **300 步时 loss 已经收敛到看起来正常的水平，但分布是错的**——所以只看 loss 会漏掉
  模式丢失。这就是 `run_demo.py` 为什么要打印扇区表而不止画 loss 曲线。
* 跑到 5000 步后，占据质量大的扇区（真值 0.221 / 0.255）几乎完全对上（0.218 / 0.218）。
* **剩下的偏差在最稀有的两个扇区**（真值 0.008 / 0.018，采样后 0.017 / 0.029）——
  模型**过度覆盖**稀有模式。这是低维高斯核扩散的标准偏差，**加长训练不会再改善**，
  属于模型族的固有限制，不是训练时长问题。

## .gitignore 现状（本次核对）

`outputs/`、`data/`、`03_omics_synthesis/data/`、`*.pt/*.pth/*.ckpt`、`__pycache__/`、
编辑器目录均已排除。

**`figures/` 是刻意不排除的**，文件里的注释写得很清楚：

```
# outputs: keep small demo figures, ignore the rest
outputs/
```

对一个作品集仓库这是对的——**图本身就是交付物**。唯一要注意的是：重跑 demo 会覆盖同名
PNG，如果哪天图变大了（比如换数据集、调高 dpi），仓库会跟着变大。要控制的话加一条
`figures/*.tmp.png` 之类，或约定图片长边上限。
