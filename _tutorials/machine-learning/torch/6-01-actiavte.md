---
youku_id: XMjc4ODM5MzQyNA
youtube_id: NGO0oxdz-zs
bilibili_id: 15997678&page=35
title: 激活函数
publish-date: 2018-11-04
thumbnail: "/static/thumbnail-small/torch/5.4_batch_normalization.jpg"
chapter: 6
description: "激活函数（Activation functions）对于人工神经网络模型去学习、理解非常复杂和非线性的函数来说具有十分重要的作用。它们将非线性特性引入到我们的网络中。如图1，在神经元中，输入的 inputs 通过加权，求和后，还被作用了一个函数，这个函数就是激活函数。引入激活函数是为了增
              加神经网络模型的非线性。没有激活函数的每层都相当于矩阵相乘。就算你叠加了若干层之后，无非还是个矩阵相乘罢了。"
post-headings:
  - 要点
  - 做点数据
  - 搭建神经网络
  - 训练
  - 画图
  - 对比结果
---

pytorch中实现了大部分激活函数，你也可以自定义激活函数，激活函数的实现在torch.nn.functional中，每个激活函数都对应激活模块类，但最终还是调用torch.nn.functional，看了定义，你也能自定义激活函数,我们从最早的激活函数来看
# sigmoid
```
def sigmoid(input):
    r"""sigmoid(input) -> Tensor

    Applies the element-wise function :math:`\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}`

    See :class:`~torch.nn.Sigmoid` for more details.
    """
    warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")
    return input.sigmoid()
```
![Sigmoid](https://upload-images.jianshu.io/upload_images/3802398-0c4b0bb7c3037f81.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
源码显示这个激活函数直接调用tensor.sigmoid函数，值域在[0,1]之间，也就是把数据的所有值都压缩在[0,1]之间，映射概率不错，如果作为激活函数有如下缺点
* 神经元容易饱和，其值不在[-5, 5]之间，梯度基本为0，导致权重更新非常缓慢
* 值域中心不是0，相当于舍弃负值部分
* 计算有点小贵，毕竟每次都算两个exp，一定要做内存和计算的葛朗台

# tanh
```
def tanh(input):
    r"""tanh(input) -> Tensor

    Applies element-wise,
    :math:`\text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}`

    See :class:`~torch.nn.Tanh` for more details.
    """
    warnings.warn("nn.functional.tanh is deprecated. Use torch.tanh instead.")
    return input.tanh()
```
![tanh](https://upload-images.jianshu.io/upload_images/3802398-c48703662552dac8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
这个函数的值域正常了，避免了sigmoid的问题，是[-1, 1]，以0为中心，但是依然存在一些问题梯度消失的神经元饱和问题，而且计算更贵！
# relu
```
def relu(input, inplace=False):
    if inplace:
        return torch.relu_(input)
    return torch.relu(input) 
```
![ReLu](https://upload-images.jianshu.io/upload_images/3802398-c128b4e11ff5e37c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


relu的函数定义就是max(0, x)，解决了梯度消失的饱和问题，计算高效，线性值，一般来说比Sigmoid/tanh快6倍左右。而且有资料显示，和生物神经激活机制非常相近。但是引入了新的问题，就是负值容易引起神经死亡，也就是说每次这个激活函数会撸掉负值的部分。
# Leaky Relu
```
def leaky_relu(input, negative_slope=0.01, inplace=False):
    r"""
    leaky_relu(input, negative_slope=0.01, inplace=False) -> Tensor

    Applies element-wise,
    :math:`\text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)`

    See :class:`~torch.nn.LeakyReLU` for more details.
    """
    if inplace:
        return torch._C._nn.leaky_relu_(input, negative_slope)
    return torch._C._nn.leaky_relu(input, negative_slope)
```
![LReLu](https://upload-images.jianshu.io/upload_images/3802398-94c02e481cf21dea.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
为了处理负值的情况，Relu有了变种，其函数是max(0.01*x, x),这个函数解决了神经饱和问题，计算高效，而且神经不死了。
# PRelu
```
def prelu(input, weight):
    r"""prelu(input, weight) -> Tensor

    Applies element-wise the function
    :math:`\text{PReLU}(x) = \max(0,x) + \text{weight} * \min(0,x)` where weight is a
    learnable parameter.

    See :class:`~torch.nn.PReLU` for more details.
    """
    return torch.prelu(input, weight)
```
![PRelu](https://upload-images.jianshu.io/upload_images/3802398-5c424ff056c43219.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这个函数的定义是max(ax, x)，其中参数a可以随时调整。

# Elu Exponential Line Unit
```
def elu(input, alpha=1., inplace=False):
    r"""Applies element-wise,
    :math:`\text{ELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x) - 1))`.

    See :class:`~torch.nn.ELU` for more details.
    """
    if inplace:
        return torch._C._nn.elu_(input, alpha)
    return torch._C._nn.elu(input, alpha)
```
![Elu](https://upload-images.jianshu.io/upload_images/3802398-a29abd5e572480d2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
这个函数的定义是max(x, a*(exp(x)-1))，继承了Relu的所有优点，but贵一点，均值为0的输出、而且处处一阶可导，眼看着就顺滑啊，哈哈，负值很好的处理了，鲁棒性很好， nice！学完批标准化后，我们展示一个小示例，它居然在那个例子中干掉了批标准化。
于是其他变种应运而生
# SELU
```
def selu(input, inplace=False):
    r"""selu(input, inplace=False) -> Tensor

    Applies element-wise,
    :math:`\text{SELU}(x) = scale * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))`,
    with :math:`\alpha=1.6732632423543772848170429916717` and
    :math:`scale=1.0507009873554804934193349852946`.

    See :class:`~torch.nn.SELU` for more details.
    """
    if inplace:
        return torch.selu_(input)
    return torch.selu(input)
```
![SELU](https://upload-images.jianshu.io/upload_images/3802398-f125b41a8dc31c7a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
还有其他变种relu6、celu等等

这些激活函数我们来个经验参考：
* 首先使用Relu，然后慢慢调整学习率
* 可以尝试Lecky Relu/Elu
* 试一下tanh，不要期望太多
* 不要尝试sigmoid
