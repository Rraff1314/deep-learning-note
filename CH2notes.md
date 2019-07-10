[TOC]

# 支持向量机(Support Vector Machine, SVM)

## 1.什么是支持向量机SVM？

 支持向量机（SVM）是90年代中期发展起来的基于统计学习理论的一种机器学习方法，通过寻求结构化风险最小来提高学习机泛化能力，实现经验风险和置信范围的最小化，从而达到在统计样本量较少的情况下，亦能获得良好统计规律的目的。

SVM能够对训练集之外的数据点做出很好的分类决策。通俗来讲，它是一种**二类**分类模型。SVM有很多种实现方式，本文只介绍最流行的一种实现，即**序列最小优化**。**其基本模型定义为特征空间上的间隔最大的线性分类器**，即支持向量机的学习策略便是间隔最大化，最终可转化为一个凸二次规划问题的求解。

## 2.需要掌握的SVM知识点

SVM整体可以分成三个部分：

1. **SVM理论本身**：包括最大间隔超平面（Maximum Margin Classifier），拉格朗日对偶（Lagrange Duality），支持向量（Support Vector），核函数（Kernel）的引入，松弛变量的软间隔优化（Outliers），最小序列优化（Sequential Minimal Optimization）等。

2. **核方法（Kernel）**：其实核方法的发展是可以独立于SVM来看待的，核方法在很多其它算法中也会应用到。

3. **优化理论**：这里主要介绍的是最小序列优化（Sequential Minimal Optimization），优化理论的发展也是独立于SVM的。

接下来，就这三部分分别进行介绍：

### 2.1SVM理论

#### 2.1.1基于最大间隔分隔数据

1. 分隔超平面：分类的决策边界。（确定分隔超平面的依据：如果数据点离决策边界越远，那么其最后的预测结果也就越可信。所以我们希望找到离分隔超平

   面最近的点，确保它们离分隔面的距离尽可能远，以让分类器尽可能健壮。）

2. 间隔：点到分隔面的间隔。

3. 支持向量：离分隔超平面最近的那些点。

#### 2.1.2寻找最大间隔

1.分隔超平面（即分类器）的形式可以写成：
$$
f(x)=w^T+b
$$
2.点到分隔面的距离为：
$$
|w^A+b|/||w||
$$

3.**分类器求解的优化问题**：

目标：找出分类器（即分隔超平面）定义中的w和b。

（1）目标函数：
$$
arg \quad max_{w,b} \quad \lbrace (min_n (label \cdot (w^T+b))) \cdot \frac{1}{||w||} \rbrace
$$
（2）直接求解上述问题相当困难，所以我们将它转换成为另一种更容易求解的形式：
$$
arg \quad max_{a,b} \quad \frac{1}{||w||}\\
 s.t. \quad label \cdot (w^T+b) \geq 1
$$
（3）进一步，转化为更容易求解的形式：
$$
arg \quad min_{a,b} \quad \frac{1}{||w||^2}\\
 s.t. 1 -label \cdot (w^T+b) \leq 0
$$
 （3）因为该问题是一个带约束条件的优化问题，对于这类优化问题，有一个非常著名的求解方法，即拉格朗日乘子法。通过引入拉格朗日乘子，我们就可以基于约束条件来表述原来的问题。所以，目标函数可转化为如下形式：
$$
\text{拉格朗日方程式：}L(w,b,\alpha)=\frac{1}{2}||w||^2+\sum_{i=1}^m \alpha_i-\sum_{i=1}^m\alpha_i\cdot lable \cdot(w^T+b)\\
\text{目标函数为：}min_{w,b}max_\alpha L(w,b,\alpha)
$$
（4）当问题满足KKT条件时，原问题的解与其对偶问题的解相同。现在我们要求解的原问题符合KKT条件，而原问题难以求解，我们可以把它转化成对偶问题进行求解：
$$
\text{拉格朗日方程式：}L(w,b,\alpha)=\frac{1}{2}||w||^2+\sum_{i=1}^m \alpha_i-\sum_{i=1}^m\alpha_i\cdot lable \cdot(w^T+b)\\
\text{目标函数为：}max_\alpha min_{w,b} L(w,b,\alpha)
$$
> **KKT条件简要介绍**
>
> 一般地，一个最优化数学模型能够表示成下列标准形式：
> $$
> minf(x)\\
> s.t. \quad h_j(x)=0,j=1,\cdots,p,\\
> g_k(x)\leq0,k=1,\cdots,q,\\
> x\in X \subset R^n
> $$
> 其中，f(x)是需要最小化的函数，h(x)是等式约束，g(x)是不等式约束，p和q分别为等式约束和不等式约束的数量。同时，我们得明白以下两个定理：
>
> - 凸优化的概念：![\mathcal{X} \subset \mathbb{R}^n](http://upload.wikimedia.org/math/d/0/1/d01e9255365440ae709190fafc071951.png) 为一凸集， ![f:\mathcal{X}\to \mathbb{R}](http://upload.wikimedia.org/math/3/4/5/345879b44bce56b80552389916fa67fe.png) 为一凸函数。凸优化就是要找出一点 ![x^\ast \in \mathcal{X}](http://upload.wikimedia.org/math/7/a/b/7ab2b524ce2a695903b81d45d27d5242.png) ，使得每一 ![x \in \mathcal{X}](http://upload.wikimedia.org/math/6/2/4/624cf12f420fb0f373cda9f7b216b2f3.png) 满足 ![f(x^\ast)\le f(x)](http://upload.wikimedia.org/math/d/a/0/da0d27822f8c98efc3d1a39ae37f30e1.png) 。
> - KKT条件的意义：它是一个非线性规划（Nonlinear Programming）问题能有最优化解法的必要和充分条件。
>
> 那到底什么是所谓Karush-Kuhn-Tucker条件呢？KKT条件就是指上面最优化数学模型的标准形式中的最小点 x* 必须满足下面的条件：
>
> （1）相应的拉格朗日函数对x的一阶导数为0。
>
> （2）h(x)=0。
>
> （3）言下之意是要么α为0，要么函数g(x)为0。
> $$
> \alpha\cdot g(x)=0,\alpha \geq 0
> $$

（5）化简，上述目标函数变成：<!--化简过程之后会补充！-->
$$
max_\alpha [\sum_{i=1}^m \alpha - \frac{1}{2}\sum_{i,j=1}^mlabel^{(i)}\cdot label^{(j)} \cdot \alpha_i \cdot \alpha_j \langle x^{(i)},x^{(j)} \rangle]\\
\alpha \geq0,和\sum_{i-1}^m \alpha_i \cdot label^{(i)}=0
$$
（6）至此，一切都很完美，但是这里有个假设：数据必须100%线性可分。目前为止，我们知道几乎所有数据都不那么“干净”。这时我们就可以通过引入所谓松弛变量，来允许有些数 点可以处于分隔面的错误一侧。这样我们的优化目标就能保持仍然不变，但是此时（2）中目标函数的新的约束条件则变为：

$$
arg \quad max_{a,b} \quad \frac{1}{||w||}\\\\
s.t. \quad label \cdot (w^T+b) \geq 1- \varepsilon_i
$$
（7）经过相同的求解思路，最终的目标函数及约束条件为：
$$
max_\alpha [\sum_{i=1}^m \alpha - \frac{1}{2}\sum_{i,j=1}^mlabel^{(i)}\cdot label^{(j)} \cdot \alpha_i \cdot \alpha_j \langle x^{(i)},x^{(j)} \rangle]\\C
\geq \alpha \geq0,和\sum_{i-1}^m \alpha_i \cdot label^{(i)}=0
$$
这里的C是个很重要的参数，它从本质上说是用来**折中经验风险和置信风险**的，C越大，置信风险越大，经验风险越小；并且所有的拉格朗日乘子都被限制在了以C为边长的大盒子里。

到目前为止，我们已经了解了一些理论知识，我们当然希望能够通过编程，在数据集上讲这些理论付诸实践。接下来将介绍一个简单但很强大的实现算法。

### 2.2优化理论（待续写）

#### SMO高效优化算法

SMO表示序列最小优化。该算法是将大优化问题分解为多个小优化问题来求解的。这些小优化问题往往很容易求解，并且对它们顺序求解的结果与将它们作为整体来求解的结果是完全一致的。在结果完全相同的同时，SMO算法的求解时间短很多。

SMO算法的目标是求出一系列alpha和b，一旦求出了alpha，就很容易计算出权重向量w并得到分割超平面。

SMO算法的工作原理是：每次循环中选择两个alpha进行优化处理。一旦找到一对合适的alpha，那么就增大其中一个同时减少另一个。这里所谓的“合适”就是指两个alpha必须要符合一定的条件，条件之一就是这两个alpha必须要在间隔边界之外，而其第二个条件则是这两个alpha还没有进行过区间化处理或者不在边界上。

 

### 2.3核函数

使用一种称为核函数的方式将SVM扩展到更多数据集上。





## 待深入学习的问题

### KKT条件

### SMO算法的实现



