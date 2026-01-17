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

$$
r_{\text{map}}(x_0,c)[U]\in\{0,1\},
$$

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
= \frac{p_{\tilde\pi^{old}}(o=1\mid c)\tilde\pi_t^{+}(x_t,U\mid c)}{\tilde\pi_t^{old}(x_t,U\mid c)}\tilde\pi_{0\mid t}^{+}(x_0\mid x_t,U,c)
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

### 定理 4.1-U（Reinforcement Guidance Optimization，U-版；对应原论文 Theorem A.4）

这一部分是你指出的 *tricky* 点：在定理 3.1-U（A.3-U）里，理论上的 $v^{+},v^{-},\Delta$ 都带有 $U$ 依赖。
但 **实现与优化时的模型 $v_\theta$ 不需要把 $U$ 作为输入**——因为 $U$ 在这里仅仅是**输出张量的坐标/索引**（或 loss 中被采样的位置），而不是“额外条件变量”。

为避免混淆，我们显式区分：
- 训练的模型：$v_\theta(x_t,c,t)$（输出一个张量场），**不输入 $U$**；
- 证明里出现的 $U$：用于把 dense reward map 转成标量 $r(x_0,U,c)$，并在 loss 中选择坐标 $[U]$。

---

#### 训练目标（对应原论文 Eq.(10)，U-版）

定义“坐标投影”与其范数：
- 对任意张量 $z$，记 $z[U]$ 为在位置 $U$ 的向量（通常是通道维上的向量）；
- $\|z\|_{U}^{2}:=\|z[U]\|_{2}^{2}$。

考虑训练目标：
$$
\boxed{
\mathcal L_U(\theta)
= \mathbb E_{c,\,(x_0,U)\sim \tilde\pi^{old}_0(\cdot,\cdot\mid c),\,t}
\Big[
r(x_0,U,c)\,\|v^{+}_\theta(x_t,c,t)-v\|_{U}^{2}
+\bigl(1-r(x_0,U,c)\bigr)\,\|v^{-}_\theta(x_t,c,t)-v\|_{U}^{2}
\Big]
}
$$
(10-U)
其中（与原论文一致的隐式正/负策略）：
$$
v^{+}_\theta(x_t,c,t):=(1-\beta)v^{old}(x_t,c,t)+\beta v_\theta(x_t,c,t),
\qquad
v^{-}_\theta(x_t,c,t):=(1+\beta)v^{old}(x_t,c,t)-\beta v_\theta(x_t,c,t).
$$
注意：这里 **$v_\theta,v^{\pm}_\theta$ 都不依赖 $U$**；$U$ 只出现在 reward 与 $\|\cdot\|_U$ 的坐标选择中。

---

#### 结论（对应原论文 Theorem A.4）

在无限数据与无限模型容量下，$(10\text{-U})$ 的最优解满足（对任意 $U$ 逐点成立）：
$$
\boxed{
v_\theta^{*}(x_t,c,t)[U]
= v^{old}(x_t,c,t)[U]+\frac{2}{\beta}\,\Delta(x_t,c,t,U)[U]
}
$$
(A.4-U)
其中 $\Delta(x_t,c,t,U)$ 是定理 3.1-U（A.3-U）定义的 reinforcement guidance：
$$
\Delta(x_t,c,t,U)
:=[1-\alpha(x_t,U)]\bigl(v^{old}(x_t,c,t)-v^{-}(x_t,c,t,U)\bigr)
=\alpha(x_t,U)\bigl(v^{+}(x_t,c,t,U)-v^{old}(x_t,c,t)\bigr).
$$

---

### 证明（严格按原论文 A.4 proof 的结构，逐步把 $U$ 带进去）

**Step 1：把 $\mathcal L_U$ 写成 $(x_t,U)$ 的边缘与后验上的期望（同原论文第一行）**

由定义，
$$
\mathcal L_U(\theta)
= \mathbb E_{c,t,\,(x_t,U)\sim \tilde\pi_t^{old}(\cdot,\cdot\mid c)}
\mathbb E_{x_0\sim \tilde\pi^{old}_{0|t}(\cdot\mid x_t,U,c)}
\Big[
r(x_0,U,c)\|v_\theta^{+}-v\|_{U}^{2}
+(1-r(x_0,U,c))\|v_\theta^{-}-v\|_{U}^{2}
\Big],
$$
其中 $x_t\sim\pi(\cdot\mid x_0)$，且 $\tilde\pi_t^{old},\tilde\pi_{0|t}^{old}$ 来自前面 Lemma 2.1/2.2 的定义。

---

**Step 2：用 Lemma 2.1-U / 2.2-U 把 $r\,\tilde\pi^{old}_{0|t}$ 改写成 $\alpha\,\tilde\pi^{+}_{0|t}$（对应原论文中 “From Lemma A.1 … therefore” 那步）**

由 Lemma 2.1-U（即 $r\,\tilde\pi_0^{old}=p(o=1\mid c)\tilde\pi_0^{+}$）有：
$$
r(x_0,U,c)\,\tilde\pi_{0|t}^{old}(x_0\mid x_t,U,c)
= r(x_0,U,c)\,\frac{\tilde\pi_0^{old}(x_0,U\mid c)\,\pi(x_t\mid x_0)}{\tilde\pi_t^{old}(x_t,U\mid c)}
= p(o=1\mid c)\,\frac{\tilde\pi_0^{+}(x_0,U\mid c)\,\pi(x_t\mid x_0)}{\tilde\pi_t^{old}(x_t,U\mid c)}.
$$
再乘除 $\tilde\pi_t^{+}(x_t,U\mid c)$ 并识别出 $\tilde\pi^{+}_{0|t}$：
$$
r(x_0,U,c)\,\tilde\pi_{0|t}^{old}(x_0\mid x_t,U,c)
= \underbrace{\frac{p(o=1\mid c)\,\tilde\pi_t^{+}(x_t,U\mid c)}{\tilde\pi_t^{old}(x_t,U\mid c)}}_{:=\alpha(x_t,U)}
\tilde\pi_{0|t}^{+}(x_0\mid x_t,U,c)
= \alpha(x_t,U)\,\tilde\pi_{0|t}^{+}(x_0\mid x_t,U,c).
$$
同理可得：
$$
(1-r(x_0,U,c))\,\tilde\pi_{0|t}^{old}(x_0\mid x_t,U,c)
= (1-\alpha(x_t,U))\,\tilde\pi_{0|t}^{-}(x_0\mid x_t,U,c).
$$

因此
$$
\mathcal L_U(\theta)
= \mathbb E_{c,t,(x_t,U)\sim \tilde\pi_t^{old}}
\Big[
\alpha(x_t,U)\,\mathbb E_{\tilde\pi^{+}_{0|t}}\|v_\theta^{+}-v\|_U^2
+(1-\alpha(x_t,U))\,\mathbb E_{\tilde\pi^{-}_{0|t}}\|v_\theta^{-}-v\|_U^2
\Big].
$$

---

**Step 3：把“回归到随机变量 $v[U]$”改写成“回归到其条件均值” + 常数（对应原论文把 $E\|a-v\|^2$ 改写为 $\|a-E[v]\|^2+C_1$）**

注意到对固定 $(x_t,c,t,U)$，$v_\theta^{\pm}(x_t,c,t)[U]$ 不依赖 $x_0$，因此
$$
\mathbb E_{\tilde\pi^{+}_{0|t}}\|v_\theta^{+}-v\|_U^2
= \|v_\theta^{+}(x_t,c,t)[U]-\mathbb E_{\tilde\pi^{+}_{0|t}}[v[U]]\|_2^2 + C_{+},
$$
$$
\mathbb E_{\tilde\pi^{-}_{0|t}}\|v_\theta^{-}-v\|_U^2
= \|v_\theta^{-}(x_t,c,t)[U]-\mathbb E_{\tilde\pi^{-}_{0|t}}[v[U]]\|_2^2 + C_{-},
$$
其中 $C_{+},C_{-}$ 与 $\theta$ 无关。

定义（注意这里的 $v^{\pm}$ 是理论上的“最优 teacher”，允许依赖 $U$）：
$$
v^{+}(x_t,c,t,U)[U]:=\mathbb E_{x_0\sim \tilde\pi^{+}_{0|t}(\cdot\mid x_t,U,c)}[v[U]],
\quad
v^{-}(x_t,c,t,U)[U]:=\mathbb E_{x_0\sim \tilde\pi^{-}_{0|t}(\cdot\mid x_t,U,c)}[v[U]].
$$
于是
$$
\mathcal L_U(\theta)
= \mathbb E_{c,t,(x_t,U)\sim \tilde\pi_t^{old}}
\Big[
\alpha(x_t,U)\,\|v_\theta^{+}[U]-v^{+}[U]\|_2^2
+(1-\alpha(x_t,U))\,\|v_\theta^{-}[U]-v^{-}[U]\|_2^2
\Big]+C_1.
$$

---

**Step 4：代入隐式正/负策略，并用定理 3.1-U（A.3-U）把它们化成关于 $v_\theta-v^{old}$ 与 $\Delta$ 的平方（对应原论文 “Combining Theorem A.3 … Substituting …” 那几行）**

由隐式定义：
$$
v_\theta^{+}[U]-v^{+}[U]
=(1-\beta)v^{old}[U]+\beta v_\theta[U]-v^{+}[U]
= \beta\Big(v_\theta[U]-v^{old}[U]-\frac{1}{\beta}\bigl(v^{+}[U]-v^{old}[U]\bigr)\Big).
$$
由 A.3-U 中 $\Delta=\alpha(v^{+}-v^{old})$，在位置 $U$ 上有
$$
v^{+}[U]-v^{old}[U]=\frac{\Delta[U]}{\alpha(x_t,U)} \quad(\alpha>0),
$$
因此
$$
v_\theta^{+}[U]-v^{+}[U]
= \beta\Big(v_\theta[U]-v^{old}[U]-\frac{1}{\beta}\frac{\Delta[U]}{\alpha(x_t,U)}\Big).
$$

同理，
$$
v_\theta^{-}[U]-v^{-}[U]
=(1+\beta)v^{old}[U]-\beta v_\theta[U]-v^{-}[U]
= -\beta\Big(v_\theta[U]-v^{old}[U]-\frac{1}{\beta}\frac{\Delta[U]}{1-\alpha(x_t,U)}\Big),
$$
其中使用了 A.3-U 的另一半 $\Delta=(1-\alpha)(v^{old}-v^{-})$。

将两式代回 $\mathcal L_U$，得到
$$
\mathcal L_U(\theta)
= \beta^2\,
\mathbb E_{c,t,(x_t,U)\sim \tilde\pi_t^{old}}
\Big[
\alpha\,\big\|v_\theta[U]-\big(v^{old}[U]+\frac{1}{\beta}\frac{\Delta[U]}{\alpha}\big)\big\|_2^2
+(1-\alpha)\,\big\|v_\theta[U]-\big(v^{old}[U]+\frac{1}{\beta}\frac{\Delta[U]}{1-\alpha}\big)\big\|_2^2
\Big]+C_1.
$$

---

**Step 5：配方（complete the square），把两项合成一项（对应原论文最后几行）**

对任意标量权重 $\alpha\in[0,1]$ 与两个“目标点” $a,b$，有恒等式
$$
\alpha\|z-a\|^2+(1-\alpha)\|z-b\|^2
= \|z-(\alpha a+(1-\alpha)b)\|^2+\text{const},
$$
其中常数项与 $z$ 无关。

令
$$
a:=v^{old}[U]+\frac{1}{\beta}\frac{\Delta[U]}{\alpha},\qquad
b:=v^{old}[U]+\frac{1}{\beta}\frac{\Delta[U]}{1-\alpha},
$$
则其加权平均为
$$
\alpha a+(1-\alpha)b
= v^{old}[U]+\frac{1}{\beta}\Delta[U]+\frac{1}{\beta}\Delta[U]
= v^{old}[U]+\frac{2}{\beta}\Delta[U].
$$
因此
$$
\mathcal L_U(\theta)
= \beta^2\,
\mathbb E_{c,t,(x_t,U)\sim \tilde\pi_t^{old}}
\big\|v_\theta(x_t,c,t)[U]-\big(v^{old}(x_t,c,t)[U]+\frac{2}{\beta}\Delta(x_t,c,t,U)[U]\big)\big\|_2^2
+C_2.
$$
从而显然最优解满足：
$$
v_\theta^{*}(x_t,c,t)[U]=v^{old}(x_t,c,t)[U]+\frac{2}{\beta}\Delta(x_t,c,t,U)[U].
$$
证毕。

---

**关于“会不会迫使模型输入 $U$”的 double-check：**

- 证明里的 $v^{+}(x_t,c,t,U)$、$v^{-}(x_t,c,t,U)$ 的确是“条件在 $U$”下的理论最优 teacher；
- 但训练目标只回归其在同一位置的分量 $[U]$，且 $v_\theta$ 本身输出的是整张图/整张 latent 网格的张量，天然就包含“每个 $U$ 的分量”；
- 因此 **实现上无需把 $U$ 作为网络输入**：只需要在 loss 中对输出张量做 gather / 加权即可。
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
