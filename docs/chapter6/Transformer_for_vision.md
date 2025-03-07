# 视觉领域的Transformer

通过前五章的学习，我们大致了解了Transformer的结构、应用及其代码实现。
Attention is all you need 这篇论文告诉我们，Transformer最开始是用在文本处理方面的机器翻译领域的。后来的ChatGPT、Claude、DeepSeek这些生成式大模型也是基于Transformer进行开发的。但是不止在自然语言处理领域，在视觉领域和多模态领域，Transformer及其变种也有发挥。下面本章将简单介绍下视觉领域中的Transformer，供学习者们进一步去探索研究。


## ViT (Vision Transformer)
    
ViT (Vision Transformer) 是一种基于Transformer架构的视觉模型。它首次将纯Transfomer结构成功用于“图像分类”任务，打破了传统卷积神经网络（CNN）在视觉任务中的主导地位。其核心思想是将图像分割为小块（Patch），并像处理文本序列一样处理这些图像块。<p>

ViT的核心结构包括：
- 图像分块嵌入（Patch Embedding）
- 位置编码/嵌入（Position Embedding）
- 线性投影/映射（Liner Projection）
- Transformer编码器
- MLP分类头
- 额外可学习的分类嵌入

![ViT结构图](./image/ViT结构图zh.jpg)

### 图像分块嵌入
    
### 位置编码/嵌入

### 线性投影/映射

### Transformer编码器

### MLP分类头







参考链接：<p>
[ViT论文精读](https://www.bilibili.com/video/BV15P4y137jb)<p>
[ViT论文原文](https://arxiv.org/abs/2010.11929)<p>
[Vision Transformer详解](https://blog.csdn.net/qq_37541097/article/details/118242600)



## Swin Transformer (SwinT)
目标检测领域



[Swin Transofrmer详解](https://blog.csdn.net/qq_39478403/article/details/120042232)

## CvT




## DETR (Detected Transformer)


## YOLOs (You Only Look at One Sequence)