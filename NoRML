# No-Reward Meta Learning
[代码地址](https://github.com/google-research/google-research/tree/master/norml)

[论文地址](https://arxiv.org/pdf/1903.01063.pdf)

## 目的与动机：传统MAML-RL面临挑战
传统的MAML-RL极度依赖外部的奖励信号来进行策略的适应微调。当出现dynamics变化、传感器偏差、丢失奖励信号等外部环境变化时，传统的MAML-RL将难以适应。

实际上，在执行一件任务时，奖励函数往往是固定的，当环境发生了变化之后，人通常能能很快地对应环境的变化做出调整而不依赖于奖励反馈，而agent也应当能够如此。因此，元学习算法本身需要学习出奖励的内部规律之类的，当意识到环境dynamics发生变化时，应能很好地做出适应。

简而言之，本文工作的主要目的是要实现一种在adaptation阶段不需要奖励反馈也能实现迅速调整的元学习算法，即所谓的No-Reward Meta Learning

## 主要思想
### 1、MAML-RL的大框架
论文中的算法1、2简单介绍了MAML-RL的方法，是在Gradient-based的元学习框架MAML下应用传统model-free方法Policy Gradient（其中采用了优势函数 $A^\pi(s_t, a_t)$来减少方差）。

而在这个框架下，NoRML做出了两个改进：
* 引入learned advantage function  $A_\Phi(s_t, a_t, s_{t+1})$ 来代替原本的优势函数 $A^\pi(s_t, a_t)$
* 增加了一个 learned parameter offset 来确保更好的进行探索
 
### 2、Learned Advantage Function
$$ A_\Phi(s_t, a_t, s_{t+1}) $$
NoRML中的优势函数相比起MAML-RL中的优势函数有所不同：

* 优势函数$A_\Phi$接受$(s_t, a_t, s_{t+1})$作为输入，是端到端地训练一个前馈神经网络，而非通过奖励r和值函数V计算出来。
* 优势函数$A_\Phi$只用于微调阶段，而meta training时不需要（详见论文算法3）。
* 事实上，$A_\Phi$并不是真实含义的优势函数，而只是为了调整PG的梯度，使其更快适应的一个函数。

这样的优势函数主要有以下好处：

* 接受（st, at, st+1）作为输入，相比起只接受（st, at）而言，更加容易让agent意识到环境dynamics的改变，从而提供一个更有信息量的动作评价。
* 测试的微调（adaptation）阶段，直接使用而不需要训练（详见算法3,算法4）
* 测试阶段不需要拟合值函数，而在MAML用少量数据难以准确拟合一个值函数，从而使得PG也没那么准确、高效。
* 优势函数直接作用于微调的梯度，可以在即使采样步长很小（梯度不准确）的情况下提供准确的调整信息。

### 3、Learned Offset
大部分情况下，单步的PG并不能充分地让meta-policy适应到新任务中，因为局部的梯度方向并不一定能指向最优值点。因此额外增加一个parameter offset $\theta_{offset}$到adaptation中则可一定程度地校正梯度的更新。

adapted policy计算方式：

$$ \theta_i=\theta + \theta_{offset} -\sum_{D_i^{train}}A_\Phi(s_t, a_t, s{t+1})\nabla_\theta log \pi_\theta(a_t|s_t)  $$

关于learned offset的图示：
![](https://github.com/JasonTOKO/RL_paper_reading/blob/master/figure/NoRML_fig1.png)

## 算法
最后，整个算法如下所示，元训练阶段，奖励$r_t$只用于训练优势函数$A^\pi$，而$A^\pi$只用于meta-train的时候，adaptation阶段的时候使用的是LAF，因此，在元测试阶段，只需要使用训练好的offset和LAF即可实现无奖励的自适应。
![](https://github.com/JasonTOKO/RL_paper_reading/blob/master/figure/NoRML_algo3&4.png)

## 实验
### 实验设置
对比算法：纯MAML；Domain Randomization（DR）；NoRML w/o offset；NoRML w/o LAF

实现细节：
* 在NoRML算法上，直接设置$\alpha$和$\theta_{offset}$一直为0即可相当于实现DR（相当于在不同环境下train同一个策略而没有进行adaptation）。
* 策略$\pi_\theta(a_t|s_t)$用一个多维高斯分布来表示，网络输出分布均值$\theta_\mu$，而标准差$\theta_\sigma$则用另一个独立的变量来表示。（实验发现这样比网络同时输出均值、标准差时训练更稳定）
* 实现MAML时，采用Meta-SGD的方案，用一个向量去替代学习率$\alpha$
* PPO应用在MAML的adaptation和meta-learning阶段，而只应用在NoRML的meta-learning阶段（因为$A_\phi$本身就有调整PG梯度的作用）
* 优势函数中的值函数使用多项式回归来拟合。

### Point agent control 
实验目的：探究LAF、offset的作用

任务设置：agent需要从点（0,0）移动到点（1,0），但其运动会受到旋转角$\phi$的影响，不同任务dynamics有着不同的$\phi$

实验结果：

![](https://github.com/JasonTOKO/RL_paper_reading/blob/master/figure/NoRML_fig2.png)

![](https://github.com/JasonTOKO/RL_paper_reading/blob/master/figure/NoRML_fig3.png)

显然，在奖励比较稀疏（更难学习）的时候，MAML效果不佳，而采用LAF能有明显的效果。Fig.2 表明了有offset可以使得最终微调收敛的更好。

而Fig.3 也说明了加上offset可以大大减少方差，并且可以很好地缩减微调的次数。

### Continuous control task
实验1：传感器带有噪声的Cartpole，每个任务都会有-10°到10°的传感器读数偏移。

实验结果：
![](https://github.com/JasonTOKO/RL_paper_reading/blob/master/figure/NoRML_fig4.png)
由Fig.4b可见，NoRML即使在没有外部奖励的条件下，依然能够收敛的比MAML快，而DR则不能很好地完成这个任务（因为这个时候传感器噪声已经不是能够忽略的环境因素）。同时也能看到没有offset最终表现效果会下降，没有LAF则收敛速度更慢。

实验2：在Half Cheetah环境上，特意允许交换两个髋关节的扭矩输出，并且去掉observation中的位置和线速度从而使得奖励更难计算

实验结果：
![](https://github.com/JasonTOKO/RL_paper_reading/blob/master/figure/NoRML_fig5.png)

由Fig.5a可见，显然在没有奖励的情况下，MAML无法进行adaptation，因此表现较差（和没有LAF的NoRML差不多），NoRML无论是收敛速度和最终效果都比MAML好。而DR虽然能达到和NoRML相当的效果，但是其步态看起来却没有NoRML稳定（Fig.5b&c）
