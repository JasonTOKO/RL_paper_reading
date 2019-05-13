# Robots that Learn to Adapt
BAIR实验室所著，论文发布在ICLR 2019

[博客地址](https://bair.berkeley.edu/blog/2019/05/06/robot-adapt/)

[论文地址](https://arxiv.org/pdf/1803.11347.pdf)

## 目的和动机
前人工作使用trial-and-error、MF meta-RL等方法来使得agent可以只需少许尝试即可快速适应。

现在这份工作想做到极致，实现**基于模型**的、**在线**的、只需**几个timesteps**即可完成的快速adaptation。

整个方法的框架：
![23b594f0dea4c087847d26dc3918c7f3.png](http://bair.berkeley.edu/static/blog/adapt/fig3.png)

## 主要思想：MAML式的元学习框架
### 1、采用model-based而非model-free
理由：MF通常需要收集奖励数据来进行adapting，往往需要经历多次rollout，而MB数据则是每个timestep都可收集，这样的小数据更新更有意义。

### 2、采用最近且连续的数据进行训练而不是随机数据集
理由：假定在rollout过程中环境设置和细节都在改变，只有比较近期的时间得到的信息才能够告知当前的任务情况。

实现：与MAML的随机采数据不同，在meta-training阶段，选择（M+K）长度的数据序列，前M个数据作为训练集，根据更新规则u学习一个adapted model，后K个数据则作为测试集，评估adapting过程的损失，如下所示：

$$ L=\Sigma_{tasks}||f_{\theta'}(s,a)-s'||^2|_{data_K} $$

其中：

$$ \theta^\prime=u(\theta,data_M) $$

整个过程可用下图来描述：
![b23de237b53464033ca6cc3111dafcd4.png](http://bair.berkeley.edu/static/blog/adapt/fig4.png)

### 3、自适应的更新方式
理由：对于非线性、表达能力高的FA（如NN），单纯使用SGD比较低效，（NN需要大量的数据才能学习出效果）因此在复杂、快速变化的环境中，往往难以实现快速适应。

实现：提出两种更新方式，GrBAL和ReBAL

**Gradient-Based Adaptive Learner (GrBAL)**:使用类似MAML的梯度方法。

$$ \theta_{\epsilon} = u_\psi(\tau_\epsilon(t-M, t-1), \theta)= \theta_\epsilon + \psi\nabla_\theta\frac{1}{M} \sum_{m=t-M}^{t-1} log \hat{p}_{\theta_\epsilon}(s_{m+1}|s_m, a_m) $$


**Recurrence-Based Adaptive Learner (ReBAL)**:实际上利用LSTM里面的门结构，用RNN模拟学习这个过程，用其中的隐含状态和元胞状态表示待学习的模型参数等，从而学习出更新规则（方向和步长等），具体原理可见[Optimization as a moder for FSL](https://openreview.net/pdf?id=rJY0-Kcll)

## 实验(更详细可参见论文)
### 仿真实验
1、浮力随机时变的环境 

![](http://bair.berkeley.edu/static/blog/adapt/fig5.gif)

2、随机裁剪掉一只脚

![](http://bair.berkeley.edu/static/blog/adapt/fig6.gif)
### 实物实验
1、断足后走直线

![](http://bair.berkeley.edu/static/blog/adapt/fig8.gif)

2、负载后走曲线

![](http://bair.berkeley.edu/static/blog/adapt/fig9.gif)

3、失调姿态下走直线

![](http://bair.berkeley.edu/static/blog/adapt/fig10.gif)

### 结果对比
![](http://bair.berkeley.edu/static/blog/adapt/fig11.png)
