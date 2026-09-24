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
python 01_mcmc_toy/run_demo.py --seeds 3
python 01_mcmc_toy/show_one_step.py
python 02_diffusion_toy/run_demo.py --steps 300
python 03_omics_synthesis/run_experiment.py --demo --seeds 0 1
```

| 步骤 | 本机结果 | 耗时（本机） |
| --- | --- | --- |
| 04 等价性测试 | 通过，等价误差 ~1e-7（float32 精度） | ~1.5 min |
| 01 MCMC 演示（3 种子） | 通过，RWM/MALA/HMC 三项对比 | ~90 s |
| 01 手算一步 + 自校验 | 通过，三个不变量全部成立 | ~2 s |
| 02 diffusion（300 步） | 通过，loss 0.364 | ~11 s |
| 03 omics 增广 | 通过，合成双批次队列 | ~1 min |

## 想看得更清楚时

```powershell
# 01 多种子：单种子的结论没有误差棒
python 01_mcmc_toy/run_demo.py --seeds 10 --skip-figures

# 01 把某一步的接受率逐项打印出来（含三个不变量自校验）
python 01_mcmc_toy/show_one_step.py

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

---

## 2026-09-23：模块 01 加深（hops 报率 + 手算一步 + 多种子 + 事实性更正）

起因：同行对 `01_mcmc_toy` 的评审指出两件事——**成本假设没有写成显式约定**，以及
**报绝对跨井次数会让读者误读**（三个 sampler 的步数不同）。本次把这两条和"单种子"
一起处理掉。

### 代码与文档改动

| 文件 | 改动 |
| --- | --- |
| `01_mcmc_toy/diagnostics.py` | `summarize()` 新增 `hops per 1k steps` / `hops per 1k evals` |
| `01_mcmc_toy/run_demo.py` | 拆成"混合效率"和"探索"两张表；新增 `--seeds N`、`--skip-figures`；逐种子写 `outputs_mcmc_summary.csv`；势垒扫描改画**率**；轨迹图标题改报率 |
| `01_mcmc_toy/show_one_step.py` | **新增**：把每个采样器一步里的全部数字（两个 log 密度、正反向 log q、接受率、判别用的随机数）打印出来，并做三项自校验，失败时返回退出码 1 |
| `01_mcmc_toy/README.md` | 重写：成本约定单列一节、hops 报率、10 种子误差棒、教学路径五课 |
| `.github/workflows/ci.yml` | 01 改为 `--seeds 3`，并新增 `show_one_step.py` 一步（自校验即冒烟测试） |
| `README.md`（根） | 模块 01 的描述改成与实测一致 |

### 10 种子实测（`python 01_mcmc_toy/run_demo.py --seeds 10 --skip-figures`，约 2–3 min，随 CPU 波动；
同一命令在两台机器上跑出的 `outputs_mcmc_summary.csv` 已核对为**逐字节一致**）

| sampler | ESS(x0) | ESS / 1k evals | hops / 1k steps | longest dwell |
| --- | --- | --- | --- | --- |
| RWM | 3753 ± 207 | **18.8 ± 1.0** | 33.0 ± 0.5 | 368 ± 52 |
| MALA | 2142 ± 113 | 3.6 ± 0.2 | 15.6 ± 0.5 | 739 ± 161 |
| HMC (L=10) | 2082 ± 234 | 3.5 ± 0.4 | **64.8 ± 1.5** | **166 ± 32** |

结论比单种子时更硬：RWM 的成本效率优势约 15 个标准差；MALA 与 HMC 在成本效率上是**平手**
（误差棒重叠），但机制完全不同（HMC 跨井率是 MALA 的 4 倍、停留时间是 1/4）。

### 两处事实性更正（重要）

1. **势垒高度写错了**：`01_mcmc_toy/README.md` 原来写"约 19.7 能量单位、落在势垒上的概率
   质量 ≈ 3×10⁻⁹"。实算 $\Delta U = h a^4 = 2.53$，鞍点密度是井底的
   $e^{-2.53}\approx 0.080$；19.7 是另一组参数的残留，而"势垒上的概率质量"这种说法本身
   也不成立（势垒高度不是概率质量）。这个错误数字已经被同行评审的 LLM 原样引用过一次。
2. **根 README 里两句话不成立**：原文说"mode-hopping 表现为轨迹卡在单井上千步"（实测最长
   停留 RWM 474 步，不是上千），以及"MALA 在同一步长下接受率明显更高"（实测调优步长两者
   都是 1.0，而 MALA 接受率 0.195 **低于** RWM 的 0.283）。两处已改。

### 势垒扫描的新结论（与旧说法相反）

旧 README 说"势垒升高后便宜的采样器先停止跨井"。按**率**算，实测是**HMC 先崩溃**：

| ΔU | 1.27 | 2.53 | 3.80 | 5.06 | 6.33 | 7.59 |
| --- | --- | --- | --- | --- | --- | --- |
| RWM | 58.5 | 33.0 | 18.5 | 10.8 | 6.6 | 4.6 |
| MALA | 38.6 | 15.1 | 9.7 | 7.2 | 7.8 | 8.0 |
| HMC | 164.4 | 65.2 | 23.0 | 7.2 | 1.5 | 0.3 |

机制上说得通：HMC 的动量取自 $N(0,I)$，能用于翻越势垒的动能是有上限的；而随机游走靠
大步长的提议不断注入能量。**注意**：扫描时步长固定在 ΔU=2.53 处调优的值，所以每条曲线
的衰减里有一部分是"没重新调参"，不是方法本身。

### 下一步（按性价比）

1. **维度扫描**：$d$ 维高斯 + $d$ 维漏斗，画 ESS/评估 对 $d$ —— 这是评审必问的第一条。
2. **逐势垒重调步长**，把"方法"和"步长"分开。
3. **replica exchange / 并行回火**：能否救回高势垒的跨井率，代价多少。
4. **故意写错的采样器**：去掉 MALA 的反向提议项，偏差在轨迹图上看不出来、在样本云上一眼
   看出——和模块 02 的"loss 收敛但分布错"是同一课。
