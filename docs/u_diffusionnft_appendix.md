# U-DiffusionNFT：将 DiffusionNFT 扩展到 Dense Reward Map（引入位置变量 $U$）

> 目标：在**不改变 DiffusionNFT 主体算法结构**（implicit positive/negative policy + forward-process SL loss）的前提下，把原先每张图一个标量 reward $r(x_0,c)$ 扩展到**与图像同形状的 reward map** $r_{\text{map}}(x_0,c)\in[0,1]^{H\times W}$（或二值 mask）。
>
> 核心做法：引入“位置变量” $U\in\mathcal U$（像素/patch 索引），把 dense reward map 转为**标量最优性概率** $r(x_0,U,c)\in[0,1]$，从而完全复用 DiffusionNFT Appendix A 的分布拆分与证明技巧。

---

## 0. 与原论文的对应关系（符号对照）

- 原论文（DiffusionNFT）标量 reward：  
  $$
  r(x_0,c)\in[0,1],\qquad r(x_0,c)=p(o=1\mid x_0,c),
  $$
  并据此定义正/负分布 $\pi^+,\pi^-$ 与系数 $\alpha(x_t)$，得到“改进方向” $\Delta$。

- 本文档（U-DiffusionNFT）dense reward map：  
  $$
  r_{\text{map}}(x_0,c)\in[0,1]^{H\times W}.
  $$
  引入 $U\in\mathcal U$ 后定义
  $$
  r(x_0,U,c):=r_{\text{map}}(x_0,c)[U]\in[0,1],
  \qquad r(x_0,U,c)=p(o=1\mid x_0,U,c),
  $$
  于是 $(x_0,U)$ 上的所有分布拆分、posterior split、以及 velocity 关系的推导，形式上与原 Appendix A 保持一致，只是所有对象都需要带上 $U$。

---

## 1. 设定与记号

### 1.1 位置集合与采样分布

- 位置集合 $\mathcal U$：可取为所有像素位置 $\{1,\dots,H\}\times\{1,\dots,W\}$，也可取 patch 网格 $\{1,\dots,H'\}\times\{1,\dots,W'\}$（强烈建议 patch 以降低方差）。
- 位置采样分布：$p(U)$（默认均匀分布）。  
  若工程上使用非均匀采样 $q(U)$（例如更偏向“坏区域”），则训练时需加重要性权重 $p(U)/q(U)$ 来保持无偏。

### 1.2 Forward noising kernel（与原论文一致）

记 forward/noising kernel 为
$$
\pi(x_t\mid x_0),
$$
并强调：**$\pi(x_t\mid x_0)$ 与“reward/最优性/正负分裂”无关，对 $\pi^{old},\pi^{+},\pi^{-}$ 完全相同。**

### 1.3 扩展后的“旧策略”联合分布

原论文的旧策略（数据收集策略）是 $\pi^{old}_0(x_0\mid c)$。  
引入 $U$ 后，定义扩展联合分布
$$
\tilde\pi^{old}_0(x_0,U\mid c):=\pi^{old}_0(x_0\mid c)\,p(U).
$$
这表示在旧策略下，“生成图像 $x_0$”与“抽取位置 $U$”是独立的。

### 1.4 Dense reward map 与最优性变量

给定任意黑盒 reward map（或 mask）$r_{\text{map}}(x_0,c)\in[0,1]^{H\times W}$，定义标量最优性概率
$$
r(x_0,U,c) := r_{\text{map}}(x_0,c)[U]\in[0,1].
$$
并定义二值最优性变量 $o\in\{0,1\}$：
$$
p(o=1\mid x_0,U,c)=r(x_0,U,c),\qquad p(o=0\mid x_0,U,c)=1-r(x_0,U,c).
$$

> 特例：当 reward map 是二值 mask（你提到的情形）  
> $$
> r_{\text{map}}(x_0,c)[U]\in\{0,1\},
> $$
> 则 $o$ 在给定 $(x_0,U,c)$ 时是确定的（hard label）。

---

## 2. 正/负分布的定义与“扩展到 $t$ 时刻”的推导

这一节对应原论文 Appendix A 的 Lemma A.1、Lemma A.2，并显式写出你要求的
$\tilde\pi_t^{\pm}(x_t,U\mid c)$ 与 $\tilde\pi_{0\mid t}^{\pm}(x_0\mid x_t,U,c)$。

### 定义 2.1（U-扩展的正/负分布：对应原论文 Eq.(7)(8)）

定义三元组 $(\tilde\pi^+,\tilde\pi^-,\tilde\pi^{old})$：

$$
\tilde\pi_0^{+}(x_0,U\mid c)
:=\tilde\pi_0^{old}(x_0,U\mid o=1,c)
=\frac{p(o=1\mid x_0,U,c)\,\tilde\pi_0^{old}(x_0,U\mid c)}{p_{\tilde\pi^{old}}(o=1\mid c)}
=\frac{r(x_0,U,c)}{p_{\tilde\pi^{old}}(o=1\mid c)}\tilde\pi_0^{old}(x_0,U\mid c),
$$

$$
\tilde\pi_0^{-}(x_0,U\mid c)
:=\tilde\pi_0^{old}(x_0,U\mid o=0,c)
=\frac{1-r(x_0,U,c)}{1-p_{\tilde\pi^{old}}(o=1\mid c)}\tilde\pi_0^{old}(x_0,U\mid c),
$$
其中
$$
p_{\tilde\pi^{old}}(o=1\mid c)
=\mathbb E_{(x_0,U)\sim \tilde\pi^{old}_0(\cdot,\cdot\mid c)}\left[r(x_0,U,c)\right]
=\mathbb E_{x_0\sim \pi_0^{old}(\cdot\mid c)}\mathbb E_{U\sim p(U)}\left[r_{\text{map}}(x_0,c)[U]\right].
$$

---

### 引理 2.1（Distribution Split：对应原论文 Lemma A.1 / Eq.(9)）

$$
\tilde\pi_0^{old}(x_0,U\mid c)
= p_{\tilde\pi^{old}}(o=1\mid c)\,\tilde\pi_0^{+}(x_0,U\mid c)
+\left[1-p_{\tilde\pi^{old}}(o=1\mid c)\right]\tilde\pi_0^{-}(x_0,U\mid c).
$$

**证明：** 由定义 2.1 的两式直接代入即可（与原论文 Lemma A.1 同步）。

---

### 定义 2.2（扩散到时刻 $t$：$\tilde\pi_t^{\pm}(x_t,U\mid c)$）

对任意 $\star\in\{old,+,-\}$，定义时刻 $t$ 的联合边缘分布：

$$
\tilde\pi_t^{\star}(x_t,U\mid c)
:=\int \pi(x_t\mid x_0)\,\tilde\pi_0^{\star}(x_0,U\mid c)\,dx_0.
$$

> 备注（你之前截图里看到的“分母一致”）：  
> 这里的 $\pi(x_t\mid x_0)$ 就是 forward process 的 kernel，**对 $\star\in\{old,+,-\}$ 完全一致**。  
> 三个 reverse policy 的不同，只会反映在 $\tilde\pi_0^{\star}(x_0,U\mid c)$ 上，而不会改变 forward kernel。

---

### 推论 2.1（时刻 $t$ 的 mixture split）

对引理 2.1 两边同时“扩散”（对 $x_0$ 积分）得到：

$$
\tilde\pi_t^{old}(x_t,U\mid c)
= p_{\tilde\pi^{old}}(o=1\mid c)\,\tilde\pi_t^{+}(x_t,U\mid c)
+\left[1-p_{\tilde\pi^{old}}(o=1\mid c)\right]\tilde\pi_t^{-}(x_t,U\mid c).
$$

---

### 定义 2.3（扩散后验：$\tilde\pi_{0\mid t}^{\pm}(x_0\mid x_t,U,c)$）

对任意 $\star\in\{old,+,-\}$，定义后验：

$$
\tilde\pi_{0\mid t}^{\star}(x_0\mid x_t,U,c)
:=\frac{\tilde\pi_0^{\star}(x_0,U\mid c)\,\pi(x_t\mid x_0)}{\tilde\pi_t^{\star}(x_t,U\mid c)}.
$$

---

### 引理 2.2（Posterior Split：对应原论文 Lemma A.2）

存在系数 $0\le \alpha(x_t,U)\le 1$，使得：

$$
\tilde\pi_{0\mid t}^{old}(x_0\mid x_t,U,c)
=\alpha(x_t,U)\tilde\pi_{0\mid t}^{+}(x_0\mid x_t,U,c)
+\left[1-\alpha(x_t,U)\right]\tilde\pi_{0\mid t}^{-}(x_0\mid x_t,U,c),
$$
其中
$$
\alpha(x_t,U)
:=\frac{p_{\tilde\pi^{old}}(o=1\mid c)\,\tilde\pi_t^{+}(x_t,U\mid c)}{\tilde\pi_t^{old}(x_t,U\mid c)}.
$$

**证明（严格仿照原 Appendix A 的 Bayes 操作）：**

由 Bayes 公式，
$$
\tilde\pi_0^{old}(x_0,U\mid c)
=\frac{\tilde\pi_t^{old}(x_t,U\mid c)\,\tilde\pi_{0\mid t}^{old}(x_0\mid x_t,U,c)}{\pi(x_t\mid x_0)}.
$$
对引理 2.1（旧分布的 mixture）中的 $\tilde\pi_0^{old}(x_0,U\mid c)$ 进行替换，并用同样的 Bayes 公式把
$\tilde\pi_0^{\pm}(x_0,U\mid c)$ 改写为 $\tilde\pi_{0\mid t}^{\pm}(x_0\mid x_t,U,c)$，得到：

$$
\tilde\pi_{0\mid t}^{old}(x_0\mid x_t,U,c)
=
\frac{p_{\tilde\pi^{old}}(o=1\mid c)\tilde\pi_t^{+}(x_t,U\mid c)}{\tilde\pi_t^{old}(x_t,U\mid c)}\tilde\pi_{0\mid t}^{+}(x_0\mid x_t,U,c)
+\frac{(1-p_{\tilde\pi^{old}}(o=1\mid c))\tilde\pi_t^{-}(x_t,U\mid c)}{\tilde\pi_t^{old}(x_t,U\mid c)}\tilde\pi_{0\mid t}^{-}(x_0\mid x_t,U,c).
$$

令
$$
\alpha(x_t,U)
:=\frac{p_{\tilde\pi^{old}}(o=1\mid c)\tilde\pi_t^{+}(x_t,U\mid c)}{\tilde\pi_t^{old}(x_t,U\mid c)},
$$
并用推论 2.1 可验证第二个系数等于 $1-\alpha(x_t,U)$。证毕。

---

## 3. Theorem A.3 的 U-版本：含 $U$ 的 velocity 改进方向

这一节对应原论文 Appendix A 的 Theorem A.3，要求你提出的：**所有量都显式包含 $U$ 的依赖**。

### 3.1 最优 velocity predictor 与后验均值的关系（原论文同款）

原论文在 Theorem A.3 的证明里使用了一个关键事实：最优 velocity predictor 与 $\mathbb E[x_0\mid x_t,\cdot]$ 线性相关。

对任意分布 $\tilde\pi^\star$（$\star\in\{old,+,-\}$），其对应的最优 velocity predictor 满足（同原论文）：

$$
v^{\star}(x_t,c,t,U)
= a_t x_t + b_t\,\mathbb E_{\tilde\pi_{0\mid t}^{\star}(x_0\mid x_t,U,c)}[x_0],
$$
其中
$$
a_t=\frac{\dot\sigma_t}{\sigma_t},\qquad
b_t=\dot\alpha_t-\frac{\dot\sigma_t\,\alpha_t}{\sigma_t}.
$$

> 注意：对旧策略 $\tilde\pi^{old}$，因为 $\tilde\pi_0^{old}(x_0,U\mid c)=\pi_0^{old}(x_0\mid c)p(U)$，从条件独立可推出
> $\tilde\pi_{0\mid t}^{old}(x_0\mid x_t,U,c)=\pi_{0\mid t}^{old}(x_0\mid x_t,c)$，因此
> $$
> v^{old}(x_t,c,t,U)=v^{old}(x_t,c,t)
> $$
> 实际上不依赖 $U$。我们仍保留 $U$ 以统一记号。

---

### 定理 3.1-U（Improvement Direction with $U$：对应原论文 Theorem A.3 / Theorem 3.1）

令 $v^{old},v^{+},v^{-}$ 分别是分布三元组 $(\tilde\pi^{old},\tilde\pi^+,\tilde\pi^-)$ 的最优 velocity predictor（按上式定义）。  
则对任意 $(x_t,c,t,U)$，有：

$$
\Delta(x_t,c,t,U)
:=[1-\alpha(x_t,U)]\Big(v^{old}(x_t,c,t)-v^{-}(x_t,c,t,U)\Big)
=\alpha(x_t,U)\Big(v^{+}(x_t,c,t,U)-v^{old}(x_t,c,t)\Big),
$$
其中系数 $\alpha(x_t,U)$ 由引理 2.2 给出。

**证明（严格仿照原 Appendix A）：**

由引理 2.2（Posterior Split）对 $x_0$ 取期望，得到
$$
\mathbb E_{\tilde\pi_{0\mid t}^{old}}[x_0]
=\alpha(x_t,U)\mathbb E_{\tilde\pi_{0\mid t}^{+}}[x_0]
+[1-\alpha(x_t,U)]\mathbb E_{\tilde\pi_{0\mid t}^{-}}[x_0].
$$
两边乘 $b_t$ 并加上 $a_t x_t$，利用 3.1 的 velocity–posterior-mean 关系，得到：
$$
v^{old}(x_t,c,t)
=\alpha(x_t,U)\,v^{+}(x_t,c,t,U)
+[1-\alpha(x_t,U)]\,v^{-}(x_t,c,t,U).
$$
移项即得：
$$
[1-\alpha(x_t,U)](v^{old}-v^{-})=\alpha(x_t,U)(v^{+}-v^{old}),
$$
即定理结论。证毕。

---

## 4. Policy Optimization（Eq.(5)(6) 的 U-版本）与训练目标

本节对应原论文 Theorem 3.2 / Appendix A 的 Theorem A.4。

### 4.1 隐式正/负 velocity（与原论文完全一致）

保持原论文的 implicit parameterization（只训练一个 $v_\theta$）：

$$
v_{\theta}^{+}(x_t,c,t):=(1-\beta)v^{old}(x_t,c,t)+\beta v_\theta(x_t,c,t),
$$
$$
v_{\theta}^{-}(x_t,c,t):=(1+\beta)v^{old}(x_t,c,t)-\beta v_\theta(x_t,c,t),
$$
其中 $\beta>0$ 为超参数（同原论文）。

### 4.2 U-DiffusionNFT 的训练目标

对 minibatch 中每个样本 $(c,x_0,r_{\text{map}})$，采样 $t\sim\mathcal U[0,1]$、$\epsilon\sim\mathcal N(0,I)$，构造 forward pairing（同原论文 Algorithm 1）：

$$
x_t=\alpha_t x_0+\sigma_t\epsilon,
\qquad
v=\dot\alpha_t x_0+\dot\sigma_t\epsilon.
$$

再采样 $m$ 个位置 $U_1,\dots,U_m\sim p(U)$，定义 $r_j:=r(x_0,U_j,c)$。  
训练目标定义为（Monte Carlo 版本）：

$$
\mathcal L_U(\theta)
:=\mathbb E\left[
\frac{1}{m}\sum_{j=1}^m
r_j\,\left\|v_{\theta}^{+}(x_t,c,t)[U_j]-v[U_j]\right\|_2^2
+(1-r_j)\,\left\|v_{\theta}^{-}(x_t,c,t)[U_j]-v[U_j]\right\|_2^2
\right].
$$

当 $m=|\mathcal U|$（或直接对全图做逐像素/逐 patch 加权求和）时，上式等价于全量版本：

$$
\mathcal L_U(\theta)
=\mathbb E\left[
\mathbb E_{U\sim p(U)}
\Big(
r(x_0,U,c)\|v_{\theta}^{+}[U]-v[U]\|_2^2
+(1-r(x_0,U,c))\|v_{\theta}^{-}[U]-v[U]\|_2^2
\Big)
\right].
$$

> 关键点：该目标在数学结构上与原论文 Eq.(5) 完全同型，只是把“标量 $r$”替换为“位置条件标量 $r(x_0,U,c)$”，并用对 $U$ 的期望/采样实现 dense-to-scalar 的桥接。

---

### 定理 4.1-U（U-Policy Optimization：对应原论文 Eq.(6) / Theorem A.4）

在无限数据、无限模型容量的理想条件下，$\mathcal L_U(\theta)$ 的最优解满足（对每个位置 $u\in\mathcal U$ 逐点成立）：

$$
v_\theta^{*}(x_t,c,t)[u]
=
v^{old}(x_t,c,t)[u]
+\frac{2}{\beta}\Delta(x_t,c,t,u),
$$
其中 $\Delta(x_t,c,t,u)$ 由定理 3.1-U 给出。

**证明（严格仿照原 Appendix A 的 Theorem A.4 结构）：**

1) 将 $\mathcal L_U$ 写成对 $\tilde\pi_{0\mid t}^{old}$ 的期望（与原论文同型）：
$$
\mathcal L_U(\theta)=\mathbb E_{c,t,(x_t,U)\sim \tilde\pi_t^{old}}
\mathbb E_{x_0\sim \tilde\pi_{0\mid t}^{old}(\cdot\mid x_t,U,c)}
\Big[
r(x_0,U,c)\|v_\theta^{+}[U]-v[U]\|^2+(1-r(x_0,U,c))\|v_\theta^{-}[U]-v[U]\|^2
\Big].
$$

2) 用引理 2.1/2.2 把 “$r\tilde\pi_{0\mid t}^{old}$” 与 “$(1-r)\tilde\pi_{0\mid t}^{old}$” 转换成正/负后验（同原论文把 $r\pi^{old}$ 换成 $\alpha\pi^+$ 的步骤）：
$$
r(x_0,U,c)\,\tilde\pi_{0\mid t}^{old}(x_0\mid x_t,U,c)=\alpha(x_t,U)\,\tilde\pi_{0\mid t}^{+}(x_0\mid x_t,U,c),
$$
$$
(1-r(x_0,U,c))\,\tilde\pi_{0\mid t}^{old}(x_0\mid x_t,U,c)=[1-\alpha(x_t,U)]\,\tilde\pi_{0\mid t}^{-}(x_0\mid x_t,U,c).
$$

3) 代回去并利用“$v_\theta^{\pm}(x_t,c,t)$ 在给定 $(x_t,c,t)$ 时不依赖 $x_0$”这一点，把它从内层期望中提出来（原论文同样用这一步从而出现 $\mathbb E[v]$）：
$$
\mathcal L_U(\theta)
=
\mathbb E_{c,t,(x_t,U)\sim \tilde\pi_t^{old}}
\Big[
\alpha(x_t,U)\,\mathbb E_{\tilde\pi_{0\mid t}^{+}}\|v_\theta^{+}[U]-v[U]\|^2
+[1-\alpha(x_t,U)]\,\mathbb E_{\tilde\pi_{0\mid t}^{-}}\|v_\theta^{-}[U]-v[U]\|^2
\Big].
$$

4) 对固定的 $(x_t,c,t,U)$，最小化 $\mathbb E\|a-v[U]\|^2$ 的最优解是 $a=\mathbb E[v[U]]$。  
于是最优条件为：
$$
v_\theta^{+}(x_t,c,t)[U]=v^{+}(x_t,c,t,U)[U],\qquad
v_\theta^{-}(x_t,c,t)[U]=v^{-}(x_t,c,t,U)[U],
$$
其中
$$
v^{+}(x_t,c,t,U):=\mathbb E_{x_0\sim \tilde\pi_{0\mid t}^{+}(\cdot\mid x_t,U,c)}[v],
\quad
v^{-}(x_t,c,t,U):=\mathbb E_{x_0\sim \tilde\pi_{0\mid t}^{-}(\cdot\mid x_t,U,c)}[v].
$$

5) 将 implicit 定义
$v_\theta^{+}=(1-\beta)v^{old}+\beta v_\theta$、$v_\theta^{-}=(1+\beta)v^{old}-\beta v_\theta$
与上一步的最优条件联立，可解得（对每个位置逐点成立）：
$$
v_\theta^{*}=v^{old}+\frac{2}{\beta}\Delta,
$$
其中 $\Delta$ 由定理 3.1-U 得出。证毕。

---

## 5. 算法描述（pseudocode）与每一步的采样/计算细节

本节对应原论文 Algorithm 1，但把标量 reward 替换为 reward map，并引入 $U$ 的采样。

### 5.1 Reward map 的“最优性概率化”（对应原论文的 group normalization）

原论文将 raw scalar reward $r_{\text{raw}}\in\mathbb R$ 归一化到 $r\in[0,1]$（optimality probability）。  
对 reward map 的对应做法是“逐位置元素级”归一化（推荐按 prompt group）：

给定同一个 prompt 的 $K$ 张生成图 $x_0^{1:K}$，得到 $K$ 个 raw map：
$$
R_{\text{raw}}^{k}(u) := R_{\text{raw}}(x_0^{k},c)[u].
$$
定义组内均值（逐位置）：
$$
\mu_c(u):=\frac{1}{K}\sum_{k=1}^{K}R_{\text{raw}}^{k}(u).
$$
归一化：
$$
R_{\text{norm}}^{k}(u):=R_{\text{raw}}^{k}(u)-\mu_c(u).
$$
再映射到 $[0,1]$：
$$
r_{\text{map}}^{k}(u)
:=\frac{1}{2}+\frac{1}{2}\,\mathrm{clip}\!\left(\frac{R_{\text{norm}}^{k}(u)}{Z_c},-1,1\right).
$$

> 特例（mask）：若你直接用二值 mask，当作 $r_{\text{map}}\in\{0,1\}$，这一步可以省略。

---

### 5.2 Algorithm：U-DiffusionNFT（在现有 DiffusionNFT 代码上最小改动）

**输入：**
- 参考/初始化 diffusion policy $v_{\text{ref}}$（或 $v^{old}$ 初值）
- prompt 数据集 $\{c\}$
- 黑盒 reward map 函数 $R_{\text{raw}}(x_0,c)$
- 位置采样分布 $p(U)$
- 超参数：组大小 $K$、每张图采样位置数 $m$、$\beta$、学习率 $\lambda$、EMA 系数 $\eta_i$、归一化因子 $Z_c$

---

#### Pseudocode

```text
Algorithm: U-DiffusionNFT (Dense Reward Map via Location Variable U)

Require:
    v_ref               # pretrained diffusion policy (velocity parameterization)
    R_raw(x0, c)        # black-box reward map function, returns H×W (or H'×W')
    {c}                 # prompt dataset
Hyperparams:
    K                   # images per prompt in rollout
    m                   # number of sampled locations per image in training
    beta                # implicit-policy hyperparameter
    lr                  # learning rate
    {eta_i}             # EMA schedule for sampling policy update
    Z_c                 # reward normalization scale (scalar or map-scale)

Initialize:
    v_old <- v_ref
    v_theta <- v_ref
    buffer D <- empty

for iteration i = 1,2,... do

  # (A) Rollout / Data Collection (same as DiffusionNFT, but store reward maps)
  for each sampled prompt c do
      Sample K images x0^{1:K} ~ pi_old(·|c) using ANY black-box solver
      For each k:
          R_raw^k <- R_raw(x0^k, c)                    # reward map
      Normalize within group (elementwise):
          R_norm^k <- R_raw^k - mean_k(R_raw^k)
      Map to optimality probability (elementwise):
          r_map^k <- 0.5 + 0.5 * clip(R_norm^k / Z_c, -1, 1)    # in [0,1]^{H×W}
      Push {c, x0^k, r_map^k} into buffer D

  # (B) Gradient / Policy Optimization
  for each minibatch {c, x0, r_map} from buffer D do
      Sample timestep t and Gaussian eps
      Forward process:
          x_t = alpha_t * x0 + sigma_t * eps
          v   = dot(alpha_t) * x0 + dot(sigma_t) * eps          # flow-matching target

      Sample locations:
          U_1,...,U_m ~ p(U)

      Compute per-location optimality probs:
          r_j = r_map[U_j]     for j=1..m

      Implicit velocities (identical to DiffusionNFT):
          v_theta^+ = (1 - beta) * v_old(x_t,c,t) + beta * v_theta(x_t,c,t)
          v_theta^- = (1 + beta) * v_old(x_t,c,t) - beta * v_theta(x_t,c,t)

      Loss (Monte Carlo estimator over U):
          L = (1/m) * sum_j [
                r_j * || v_theta^+[U_j] - v[U_j] ||^2
              + (1-r_j) * || v_theta^-[U_j] - v[U_j] ||^2
              ]

      Update theta by gradient descent on L

  # (C) Online update of sampling policy (same as DiffusionNFT)
  theta_old <- eta_i * theta_old + (1 - eta_i) * theta
  clear buffer D

Output: v_theta
```

---

## 6. 实现层面：对已有 DiffusionNFT 代码的“最小改动建议”（不依赖具体代码结构）

> 你提到代码库已实现原始 DiffusionNFT，所以这里**只列“必需改动点”**，不做过度推测。

1) **Reward 数据结构改动（核心）**  
   - 原：每条样本存一个标量 $r\in[0,1]$。  
   - 现：每条样本存一个 $r_{\text{map}}\in[0,1]^{H\times W}$（或 patch grid）。
   - 若存全分辨率 map 太占内存，可只存低分辨率（patch/grid）或在收集时就采样若干 $U_j$ 并只存 $(U_j,r_j)$ 对（这会改变 buffer 格式，但不改主算法逻辑）。

2) **训练 step 的 loss 计算**  
   - 原 loss：标量 $r$ 广播到整张图（或整条向量）上做加权。  
   - 现 loss：对每张图采样 $U_1..U_m$，从 $r_{\text{map}}$ gather 出 $r_j$，再对预测张量同样 gather 对应位置，计算上面的 Monte Carlo loss。

3) **其他逻辑不变**  
   - $v_\theta^+$、$v_\theta^-$ 的构造、forward pairing $(x_t,v)$、以及 EMA 更新 $v^{old}$ 都与原 Algorithm 1 完全一致。

---

## 7. 可选的工程增强（不影响理论正确性）

- **patch 化 $U$**：把 $\mathcal U$ 设为 patch 网格（例如 16×16），方差更小，也更接近 latent diffusion 的空间分辨率。
- **非均匀采样 $U$**：例如更常采“坏区域”，但务必做重要性重权：
  $$
  \mathbb E_{U\sim p(U)}[\cdot] = \mathbb E_{U\sim q(U)}\left[\frac{p(U)}{q(U)}(\cdot)\right].
  $$
- **reward map 平滑**：对二值 mask 可做 dilation/blur 得到 soft mask，提高梯度信号连续性。

---

## 8. 参考（建议放在项目 README / notes 中）

- DiffusionNFT: *Online Diffusion Reinforcement with Forward Process* (arXiv:2509.16117)
- 官方实现：NVlabs/DiffusionNFT
