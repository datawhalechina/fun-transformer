# 1. 编码器(Encoder)
## 1.1 Encoder 工作流程
### 1.1.1 输入阶段
**(1)初始输入**

整个 Encoder 部分由 6 个相同的子模块按顺序连接构成。第一个 Encoder 子模块接收来自**嵌入（Input Embedding）和位置编码（Position Embedding）组合后的输入（inputs）**。这里的输入嵌入通常是将输入的原始数据（比如文本中的单词等）转化为向量表示，而位置编码则是为了让模型能够捕捉到输入序列中元素的位置信息，因为标准的向量表示本身没有位置概念。

**(2)后续 Encoder 输入**

除了第一个 Encoder 之外的其他 Encoder 子模块，它们从**前一个** Encoder 接收相应的输入（inputs），这样就形成了一个顺序传递信息的链路。

![图片描述](./images/C3image1.PNG)

### 1.1.2 核心处理阶段
**(1)多头自注意力层处理**

每个 Encoder 子模块在接收到输入后，首先会将其传递到**多头自注意力层（Multi-Head Self-Attention layer）**。在这一层中，通过多头自注意力机制（如前面所述，查询、键、值都来自同一个输入序列自身）去计算输入序列不同位置之间的关联关系，生成相应的自注意力输出。

**(2)前馈层处理**

自注意力层的输出紧接着被传递到**前馈层（Feedforward layer）**。前馈层一般是**由全连接网络等构成**，对自注意力层输出的特征做进一步的**非线性变换**，提取更复杂、高层次的特征，然后将其输出向上发送到下一个编码器（如果不是最后一个 Encoder 的话），以便后续 Encoder 子模块继续进行处理。

### 1.1.3 残差与归一化阶段
**(1)残差连接（Residual Connection）**

自注意力层和前馈子层（Feedforward sublayer）均配备了残差快捷链路。从网络拓扑结构来看，这种连接方式构建了一种并行通路，使得**输入信号能够以残差的形式**参与到每一层的输出计算中。

对于自注意力层，其**输入会与自注意力层的输出进行相加操作**（ 假设自注意力层输入为 x，输出为 y，经过残差连接后变为 x + y ）

同样，前馈层的输入也会和前馈层的输出进行相加。**残差连接**（Residual Connection）有助于缓解深度网络训练过程中的梯度消失或梯度爆炸问题，使得网络能够更容易地训练深层模型，并且能够让信息更顺畅地在网络中传递。


**(2)层归一化（Layer Norm）**

在残差连接之后，紧跟着会进行层归一化操作。**层归一化是对每一层的神经元的输入进行归一化处理**，它可以加速网络的收敛速度、提高模型的泛化能力等，使得模型训练更加稳定、高效。经过层归一化后的结果就是当前 Encoder 子模块最终的输出，然后传递给下一个 Encoder 子模块或者后续的其他模块（比如在 Encoder-Decoder 架构中传递给 Decoder 部分等情况）。


## 1.2 Encoder 组成成分
每个子模块就是下面图中右侧那个方块了。包含几个子部分：

![图片描述](./images/C3image2.PNG)

- Multi-Head Attention
- Residual connection
- Normalisation
- Position-wise Feed-Forward Networks

![图片描述](./images/C3image3.PNG)

在 Transformer 模型中，Encoder 部分由多个相同的 Encoder Layer 堆叠而成，每个 Encoder Layer 包含两个主要子层，分别是 **Multi-Head Self-Attention 和 Position-wise Feed-Forward Network。**

Multi-Head Self-Attention**由 Scaled Dot-product Attention 和 Multi-Head Attention 以及 Self Attention 和 Add & Norm 组成。**

# 2. 多头自注意力(Multi-Head Self-Attention)
Transformer模型中，多头注意力机制允许模型在不同的子空间中学习到不同的关系。每个头都有自己的Q、K和V，最后将所有头的输出通过一个线性层拼接起来。

> Q，K，V 叫法的起源
Q、K 和 V 分别代表 Query（查询）、Key（键）和 Value（值）。这些术语的来源和它们在注意力机制中的角色都与数据库的查询概念有相似之处。

在注意力机制中：<p>
1. **Query (Q):** 它代表了正在询问的信息或关心的上下文。在自注意力机制中，每个序列元素都有一个对应的查询，它试图从其他部分找到相关信息。<p>
2. **Key (K):** 这些是可以查询的条目或“索引”。在自注意力机制中，每个序列元素都有一个对应的键。<p>
3. **Value (V):** 对于每一个“键”，都有一个与之关联的“值”，它代表实际的信息内容。当查询匹配到一个特定的键时，其对应的值就会被选中并返回。<p>
<p>这种思路与数据库查询非常相似，可以将 Query 看作是搜索查询，Key 看作是数据库索引，而 Value 则是实际的数据库条目。
<p>在 "Attention Is All You Need" 这篇 Transformer 论文中，这些术语首次被广泛地采纳和使用。注意力机制的核心思想是：

> **对于给定的 Query，计算其与所有 Keys 的相似度，然后用这些相似度对 Values 进行加权求和，得到最终的输出。**

<p>尽管这些术语在 Transformer 和许多现代 NLP 模型中被普遍接受，但它们的选择并没有特定的历史背景或来源；它们只是直观的命名，用来描述注意力计算的各个部分。
    
## 2.1 缩放点积注意力(Scaled Dot-Product Attention)
self-attention 的输入是序列词向量，此处记为 x。而 x 经过一个线性变换得到 query(Q), x经过第二个线性变换得到 key(K),  x经过第三个线性变换得到 value(V)。
    
![图片描述](./images/C3image4.png)
    
也就是说，Q、K、V 都是对输入 x 的线性映射：
- query = linear_q(x)
- key = linear_k(x)
- value = linear_v(x)
    
![图片描述](./images/C3image5.png)
    
> “查询、键和值的概念来自检索系统。例如，当您键入查询以在 Youtube 上搜索某些视频时，搜索引擎会将您的查询与数据库中与候选视频相关的一组键（视频标题、描述等）进行映射，然后向您显示最匹配的视频（值）。
    
![图片描述](./images/C3image6.png)
    
注意：这里的 linear_q(x)，linear_k(x)，linear_v(x) 相互独立，通过 softmax 函数对 Query（Q）和 Key（K）向量缩放点积的分数进行归一化，得到权重系数（attention weights），值都介于0到1之间。按照公式：
$$Attention(Q, K, V)=Softmax(\frac{QK^T}{\sqrt{d_k}})V$$
使用权重系数对Value（V）向量进行加权求和，得到最终的注意力输出 Attention(Q, K, V)。 

### 2.1.1 如何得到缩放因子
![图片描述](./images/C3image7.png)
在多头注意力机制中，参数 $d_k$ （每个头的维度）通常是由总的模型维度 $d_{model}$ 和多头注意力的头数 (h) 决定的。具体来说，$d_k$ 通常是这样计算的：$d_k=\frac{d_{\mathrm{model}}}h$

这样做有几个原因：
1. 参数平衡：通过这种方式可以确保每个头的参数数量相同，并且总的参数数量与单头注意力相当。这有助于模型的扩展性和可管理性。
2. 计算效率：因为 $d_k$ 是 $d_{model}$ 的一个因子，所以在进行矩阵运算时，这能更有效地利用硬件加速。
3. 多样性：当使用多个头时，每个头都在不同的子空间中进行操作，这有助于模型捕获输入之间多样性更丰富的关系。
4. 可解释性和调试：选择一个合适的 $d_k$ 可以使每个注意力头更易于解释和调试，因为每个头的维度都相对较小。在某些特殊应用或研究场景下，**可以手动设置** $d_k$ 。
    
### 2.1.2 缩放因子的作用
这个公式相比于正常的点积注意力多了一个缩放因子 ${\sqrt{d_k}}$，这个缩放因子可以防止内积过大，防止它经过 softmax 后落入饱和区间，因为饱和区的梯度几乎为0，容易发生梯度消失。<p>
如果忽略激活函数softmax的话，那么事实上它就是Q，K，V三个矩阵相乘，最后的结果是一个维度为 $（n·d_v）$ 的矩阵。于是我们可以认为：这是一个Attention层，将序列Q编码成了一个新的 $（n·d_v）$ 的序列。<p>
在缩放点积注意力(scaled dot-product attention) 中，还有mask部分，在训练时它将被关闭，在测试或者推理时，**它将被打开去遮蔽当前预测词后面的序列**。
    
### 2.1.3 计算 attention 时为何选择点乘
(1) 计算效率更高，可以通过矩阵乘法进行并行优化，尤其适合大规模的模型训练和推理。<p>
(2) 在计算复杂度上，虽然理论上点乘和加法注意力的复杂度都是 $O(d)$，但点乘在实际硬件中通过并行化能够显著提升计算速度。<p>
(3) 在效果上，点乘注意力能够有效衡量向量的相似性，尤其在高维度向量时，通过缩放避免数值不稳定问题，而加法注意力由于非线性操作的引入，效果上并无显著提升，且计算更为复杂。
    
## 2.2 多头注意力机制(Multi-Head Attention)
    
![图片描述](./images/C3image8.png)
    
多头(Multi-Head) 的方式是将**多个 head 的输出 z**，进行**拼接**（**Concat**）后，通过线性变换得到**最后的输出 z**。
    
![图片描述](./images/C3image9.png)
    
Multi-Head Attention把Q,K,V通过参数矩阵映射，然后再做Attention，把这个过程重复做 h 次，结果拼接起来。<p>
具体用公式表达：
$head_i=Attention(QW_i^Q,KW_i^K,VW_i^V)$
其中
$W_i^Q\text{、}W_i^K\text{、}W_i^V$ 
的权重矩阵的维度分别为<p>
$ (d_{k}\times\tilde{d}_{k})\text{、}(d_{k}\times\tilde{d}_{k})\text{、}(d_{v}\times\tilde{d}_{v}) $ 
<p>
然后通过

$MultiHead(Q,K,V)=Concat(head_1,\ldots,head_h)$

>Concat（Concatenate，拼接）将所有头的输出沿着**特征维度**拼接，形成一个大矩阵。

最后得到一个 

$n\times(h\tilde{d}_v)\text{}$ 

维度的序列，所谓Multi-Head(多头)，就是只多做几次同样的事情（**参数不共享**），然后把结果拼接。
    
**其中需要注意的是**<p>
(1) 不同的 head 的矩阵是不同的
    
(2) multi-head-Attention可以并行计算，Google论文里
$h=8\text{, }d_k=d_v=d_{model}/4=64$

一般来说一个多头 attention 的效果要优于单个 attention。按照Google官方的说法，这么做形成多个子空间，可以让模型关注不同方面的信息，体现出差异性。<p>
但有实验表明，头之间的差距随着所在层数变大而减小，因此这种差异性是否是模型追求的还不一定。
至于头数 h 的设置，并不是越大越好，到了某一个数就没效果了。
    
### 2.2.1 多头注意力机制的优点：
- **并行化**：通过同时关注输入序列的不同部分，多头注意力显著加快了计算速度，使其比传统的注意力机制更加高效。
- **增强表示**：每个注意力头都关注输入序列的不同方面，使模型能够捕捉各种模式和关系。这导致输入的表示更丰富、更强大，增强了模型理解和生成文本的能力。
- **改进泛化性**：多头注意力使模型能够关注序列内的局部和全局依赖关系，从而提高了跨不同任务和领域的泛化性。
### 2.2.2 多头注意力的计算：
- **线性变换**：输入序列经历可学习的线性变换，将其投影到多个较低维度的表示，称为“头”。每个头关注输入的不同方面，使模型能够捕捉各种模式。
- **缩放点积注意力**：每个头独立地计算输入序列的查询、键和值表示之间的注意力分数。这一步涉及计算令牌及其上下文之间的相似度，除以模型深度的平方根进行缩放。得到的注意力权重突出了每个令牌相对于其他令牌的重要性。
- **拼接和线性投影**：来自所有头的注意力输出被pinjie并线性投影回原始维度。这个过程将来自多个头的见解结合起来，增强了模型理解序列内复杂关系的能力。
    
## 2.3 自注意力机制(Self Attention)
首先我们来定义一下什么是“self-Attention” 。Cheng 等人在论文《Long Short-Term Memory-Networks for Machine Reading》中将self-Attention 定义为将单个序列或句子的不同位置关联起来以获得更有效表示的机制。<p>
在自注意力中，Query、Key和Value都来自于同一个输入序列。它允许模型在处理序列数据时，同时考虑序列中所有元素之间的关系。
1. Encoder Self-Attention：Encoder 阶段捕获当前 word 和其他输入词的关联；
2. Masked Decoder Self-Attention ：Decoder 阶段捕获当前 word 与已经看到的解码词之间的关联，从矩阵上直观来看就是一个带有 mask 的三角矩阵；
3. Encoder-Decoder Attention：就是将 Decoder 和 Encoder 输入建立联系，和之前那些普通 Attention 一样；
    
![图片描述](./images/C3image10.png)
    
在 Google 的论文中，所采用的大部分 Attention 机制为 Self Attention，也就是 “自注意力”，亦称为内部注意力。

Self Attention 是在序列内部进行 Attention 操作，旨在寻找序列内部的联系。其原理可进一步解释为：Attention 的输入是 Q（Query）、K（Key）、V（Value），对于自注意力而言，是对同一个输入序列 X，分别进行三种独立的线性变换得到 Q_x、K_x、V_x 后，将其输入 Attention，体现在公式上即 Attention (Q_x, K_x, V_x)。

在机器翻译乃至一般的 Seq2Seq 任务中，内部注意力在序列编码方面相当关键。以往关于 Seq2Seq 的研究大多仅将注意力机制应用于解码端，而 Google 的创新之处在于使用 Self Multi-Head Attention 进行序列编码，其计算公式为
$Y=MultiHead(X,X,X)$
即把同一个序列同时当作查询（Query）、键（Key）以及值（Value）来参与注意力机制的运算，以此挖掘序列内部诸如句子中单词之间的语义关联、语法结构依存关系等各种联系。

Multi-Head-Attention 的具体操作是将经过 embedding 后的 X 按照维度 d_model = 512 切割成 h = 8 个部分，分别进行 self-attention 计算，之后再将结果合并在一起。
### 2.3.1自注意力的工作原理
考虑一句话：“The cat sat on the mat.”<p>
**(1)嵌入**<p>
首先，模型将输入序列中的每个单词嵌入到一个高维向量表示中。这个嵌入过程允许模型捕捉单词之间的语义相似性。<p>
**(2)查询、键和值向量**<p>
接下来，模型为序列中的每个单词计算三个向量：查询向量、键向量和值向量。在训练过程中，模型学习这些向量，每个向量都有不同的作用。查询向量表示单词的查询，即模型在序列中寻找的内容。键向量表示单词的键，即序列中其他单词应该注意的内容。值向量表示单词的值，即单词对输出所贡献的信息。<p>
**(3)注意力分数**<p>
一旦模型计算了每个单词的查询、键和值向量，它就会为序列中的每一对单词计算注意力分数。这通常通过取查询向量和键向量的点积来实现，以评估单词之间的相似性。<p>
**(4)SoftMax 归一化**<p>
然后，使用 softmax 函数对注意力分数进行归一化，以获得注意力权重。这些权重表示每个单词应该关注序列中其他单词的程度。注意力权重较高的单词被认为对正在执行的任务更为关键。<p>
**(5)加权求和**<p>
最后，使用注意力权重计算值向量的加权和。这产生了每个序列中单词的自注意力机制输出，捕获了来自其他单词的上下文信息。
    
### 2.3.2 Self-Attention优点
1. **参数少：**
self-attention的参数为 $O(n^2d)$，而循环网络的参数为 $O(nd^2)$，卷积的参数为 $O(knd^2)$，当 n 远小于 d 时，self-attention更快
2. **可并行化：**
RNN 需要一步步递推才能捕捉到，而 CNN 则需要通过层叠来扩大感受野。Attention机制每一步计算不依赖于上一步的计算结果，因此可以和 CNN 一样并行处理。
3. **捕捉全局信息：**
更好的解决了长时依赖问题，同时只需一步计算即可获取全局信息，在 Attention 机制引入之前，有一个问题大家一直很苦恼：长距离的信息会被弱化，就好像记忆能力弱的人，记不住过去的事情是一样的。Attention 是挑重点，就算文本比较长，也能从中间抓住重点，不丢失重要的信息。<p>
<p>从计算一个序列长度为n的信息要经过的路径长度来看, CNN 需要增加卷积层数来扩大视野，RNN需要从 1 到 n 逐个进行计算，而 Self-attention 只需要一步矩阵计算就可以。Self-Attention 可以比 RNN更好地解决长时依赖问题。当然如果计算量太大，比如序列长度 N 大于序列维度 D 这种情况，也可以用窗口限制 Self-Attention 的计算数量。
    
### 2.3.3 Self-Attention缺点
**计算量**：<p>
可以看到，事实上 Attention 的计算量并不低。比如 Self Attention 中，首先要对 X 做三次线性映射，这计算量已经相当于卷积核大小为 3 的一维卷积了，不过这部分计算量还只是 $O(n)$的；然后还包含了两次序列自身的矩阵乘法，这两次矩阵乘法的计算量都是 $O(n^2d)$ 的，要是序列足够长，这个计算量其实是很难接受的。<p>
**没法捕捉位置信息**：<p>
即没法学习序列中的顺序关系。这点可以通过加入位置信息，如通过位置向量来改善，具体可以参考BERT模型。<p>
> 在Google原文中没有提到缺点，后来在论文Universal Transformers中指出的，主要是两点：<p>
（1）实践上：存在一些任务，RNN能够轻松应对，而Transformer则未能有效解决。例如，在复制字符串的任务中，或者当推理过程中遇到的序列长度超出训练时的最大长度时（由于遇到了未曾见过的位置嵌入），Transformer的表现不如RNN。<p>
（2）理论上：与RNN不同，Transformer模型并不具备计算上的通用性（即非图灵完备）。这类非RNN架构的模型，包括基于Transformer的BERT等，在处理NLP领域中的推理和决策等计算密集型问题时，存在固有的局限性。即无法独立完成某些复杂的计算任务。
    
### 2.3.4 Add & Norm　
Add & Norm : 是指将**残差连接（Addition）和归一化（Normalization）结合在一起**的操作，用于提高训练过程中的稳定性和性能<p>
**1. 残差连接（Add）**
  - 残差连接是一种常见的深度学习技巧，它通过将子层的输出添加到其输入上来实现。这样可以形成一个短路，允许梯度直接流过网络层，有助于缓解深层网络中的梯度消失问题。
  - 数学上，残差连接可以表示为：
    $\mathrm{Residual}=x+\mathrm{SubLayer}(x)$
    其中 $x$ 是子层的输入
    ${SubLayer}(x)$
    是子层的输出。<p>
      
**2. 层归一化（Norm）**<p>
  **(1)独立归一化**<p>
  与**批量归一化**（**Batch Normalization**）不同，层归一化不依赖于批次中的其他样本。这意味着即使在处理小批量数据或者在线学习场景时，层归一化也能保持稳定和有效。<p>
  **(2)稳定训练过程**<p>
  层归一化通过将每个特征的均值变为 0，标准差变为 1，有助于减少**内部协变量偏移**（**Internal Covariate Shift**）。这种偏移是指神经网络在训练过程中，由于参数更新导致的每层输入分布的变化。通过归一化，可以使得每一层的输入分布更加稳定，从而加速训练过程。<p>
**(3)提高模型稳定性**<p>
  由于层归一化减少了特征之间的尺度差异，这有助于避免某些特征在学习过程中占据主导地位，从而提高了模型的泛化能力和稳定性。<p>
 **(4)适应不同类型的网络**<p>
层归一化特别适用于循环神经网络（RNN）和Transformer模型，因为这些网络结构在处理序列数据时，每个时间步或位置的状态是相互依赖的，而批量归一化在这些情况下可能不太适用。<p>
**(5)减少梯度消失和爆炸**<p>
  通过归一化处理，可以减少梯度在传播过程中的消失或爆炸问题，尤其是在深层网络中。这有助于更有效地进行反向传播，从而提高训练效率。<p>
 **(6)不受批量大小限制**<p>
  层归一化不依赖于批次大小，因此在处理不同大小的批次时，不需要调整超参数，这使得层归一化更加灵活。
     
**将这两个操作结合在一起，“Add & Norm” 的步骤如下：**
1. 计算子层的输出：
   ${SubLayer}(x)$
2. 执行残差连接：
   $\mathrm{Residual}=x+\mathrm{SubLayer}(x)$
3. 应用层归一化：
   $\mathrm{Output}=\text{Layer}\mathrm{Norm}(\text{Residual})$
<p>所以，“Add & Norm” 的整个过程可以表示为：
    
$\mathrm{Output}=\text{LayerNorm}(x+\mathrm{SubLayer}(x))$

输入的 x 序列经过 “Multi-Head Self-Attention” 之后实际经过一个“Add & Norm”层，再进入“feed-forward network”(后面简称FFN)，在FFN之后又经过一个norm再输入下一个encoder layer。
注意：几乎每个子层都会经过一个归一化操作，然后再将其加在原来的输入上，这个过程被称为**残余连接**（**Residual Connection**）
    
## 2.4  前馈全连接网络(Position-wise Feed-Forward Networks)
**前馈全连接网络(FFN )层实际上就是一个线性变换层**，用来完成输入数据到输出数据的维度变换。FFN层是一个顺序结构：包括一个**全连接层(FC) + ReLU 激活层 + 第二个全连接层(FC)**，通过在两个 FC 中间添加非线性变换，增加模型的表达能力，使模型能够捕捉到复杂的特征和模式。<p>
$FFN(x)=max(0,xW_1+b_1)W_2+b_2$
上式中，$xW_1+b_1$ 为第一个全连接层的计算公式
$max(0,xW_1+b_1)W_2$ 
为 relu 的计算公式
$max(0,xW_1+b_1)W_2+b_2$
则为第二个全连接层的计算公式。
全连接层线性变换的主要作用为数据的升维和降维
$W_1$ 
的维度是(2048，512)
$W_2$ 
是 (512，2048)。 即**先升维，后降维**，这是为了扩充中间层的表示能力，从而抵抗 ReLU 带来的模型表达能力的下降。


## 2.5 Multi-Head Attention vs Multi-Head Self-Attention
**(1)基础架构相同**

Multi - Head Self - Attention 和 Multi - Head Attention 都基于多头注意力（Multi - Head Attention）机制的基本架构。它们都包含多个并行的注意力头（Attention Head），每个头都有自己的线性变换矩阵用于计算查询（Query）、键（Key）和值（Value）。计算过程都涉及到缩放点积注意力（Scaled Dot - Product Attention）

**(2)计算流程相似**

在整体的计算流程上，两者都需要先对输入进行线性变换以得到 Q、K、V，然后通过注意力机制计算每个头的输出，最后将各个头的输出进行拼接和线性变换得到最终的输出。

**(3)查询、键、值的来源不同**

**Multi - Head Attention**

查询（Query）、键（Key）和值（Value）可以来自**不同**的输入源。例如 Decoder部分，查询（Query）来自解码器当前的输入，而键（Key）和值（Value）通常来自编码器的输出。这种机制使得模型能够将解码器当前的信息与编码器已经处理好的信息进行关联，从而更好地生成输出序列。

**Multi - Head Self - Attention**

查询（Query）、键（Key）和值（Value）都来自**同一个**输入序列。这意味着模型关注的是输入序列**自身**不同位置之间的关系。它可以让模型自己发现句子中不同单词之间的相互关联，比如在句子 “The dog chased the cat” 中，单词 “dog” 与 “chased”、“chased” 与 “cat” 之间的关系可以通过 Multi - Head Self - Attention 来挖掘。

**(4)功能重点有所差异**

Multi - Head Attention

**主要用于融合不同来源的信息**。比如在机器翻译的任务中，它用于将元文本经过 Encoder编码后的信息（作为 K 和 V ）与解码器当前生成的部分目标语言句子（作为 Q ）相结合，帮助解码器在生成目标语言句子时更好地参考源语言句子的语义和结构，从而生成更准确的翻译。

Multi - Head Self - Attention

**更侧重于挖掘输入序列自身的内在结构和关系**。在文本生成任务中，它可以帮助模型理解当前正在生成的文本自身的语义连贯和语法结构。例如，在续写一个故事时，通过 Multi - Head Self - Attention 可以让模型把握已经生成的部分文本的主题、情节发展等内部关系，以便更好地续写。

**(5)输出信息性质不同**

**Multi - Head Attention**

由于其融合了不同来源的信息，输出的结果往往**包含了两个或多个不同输入序列之间相互作用后的特征**。例如，在跨模态任务（如将文本和图像信息相结合）中，输出会包含文本和图像相互关联后的综合特征，用于后续的分类或生成等任务。

**Multi - Head Self - Attention**

输出的是**输入序列自身内部关系的一种特征表示**。例如，在对一个文本序列进行词性标注任务时，输出的特征能够反映出句子内部单词之间的语法和语义关联，用于确定每个单词的词性。


# 3. Cross Attention
## 3.1 Cross attention简述
**概念**<p>
    交叉注意力是一种注意力机制，它允许一个序列（称为“查询”序列）中的元素关注另一个序列（称为“键-值”序列）中的元素，从而在两个序列之间建立联系。<p>
**序列维度要求**<p>
    为了进行交叉注意力计算，两个序列必须具有相同的维度。这是因为注意力机制的计算涉及到查询（Q）、键（K）和值（V）向量的点积操作，这些操作要求参与计算的向量具有相同的维度。<p>
**序列的多样性**<p>
两个序列可以是不同模态的数据，例如：
- 文本序列：一系列单词或子词的嵌入表示。
- 声音序列：音频信号的时序特征表示。
- 图像序列：图像的像素或特征图的嵌入表示。
<p>这种灵活性使得交叉注意力成为多模态学习任务中的有力工具，因为它能够桥接不同数据类型之间的信息。

## 3.2 交叉注意力的操作：
- 查询（Q）序列：这个序列定义了输出的序列长度，即查询序列中的每个元素都会生成一个对应的输出元素。
- 键（K）和值（V）序列：这两个序列提供输入信息，其中键序列用于计算与查询序列中元素的相似度，值序列则用于生成最终的输出表示<p>
<p>计算过程：<p>
1. 对于查询序列中的每个元素，计算其与键序列中所有元素的相似度。<p>
2. 使用相似度分数对值序列中的元素进行加权求和，得到每个查询元素的上下文表示。<p>
3. 将这些上下文表示组合起来，形成最终的输出序列。
    
# 4. Cross Attention 和 Self Attention 主要的区别
Cross Attention（交叉注意力）和 Self Attention（自注意力）是注意力机制中的两种不同类型，它们在信息交互的方式上有显著的区别。具体来说，**主要区别**体现在输入的来源和信息交互的对象上。
## 4.1 Self Attention 的定义与特点
自注意力（Self Attention）是Transformer架构中的核心组成部分，其主要功能是捕捉序列内部各元素之间的依赖关系。<p>
(1)定义与工作原理：<p>
自注意力机制允许序列中的每个元素（如单词或特征向量）与序列中的所有其他元素进行交互。这一过程涉及为每个元素生成查询（Query）、键（Key）和值（Value）三个向量。通过计算查询向量与所有键向量之间的点积，得到注意力权重，这些权重随后用于对值向量进行加权求和，从而为每个元素生成一个包含上下文信息的向量。<p>
(2)作用：<p>
  - **并行处理**：Self Attention允许模型并行处理序列中的所有元素，因为它不依赖于序列元素之间的顺序。<p>
  - **长距离依赖**：它能够有效地捕捉长距离依赖，即使序列中的元素相隔很远，也能通过注意力机制建立联系。<p>
  - **参数效率**：由于所有元素共享相同的权重，Self Attention在参数数量上比传统的循环神经网络更为高效。

(3)主要特点：
  - 信息来源：自注意力的查询、键和值均来源于同一输入序列，这使得它能够从序列内部挖掘信息。
  - 应用场景：自注意力适用于需要对序列内部元素之间的相互依赖进行建模的场景，如NLP中的句子编码、图像处理中的自相关特征提取等。
      
## 4.2 Cross Attention 的定义与特点
交叉注意力（Cross Attention）是一种机制，它促进了两个不同序列之间的信息交互，广泛应用于多模态任务以及需要跨数据源交互的场景，例如在序列到序列模型（Seq2Seq）中，或者在图像与文本信息对齐的任务中。<p>
(1)定义与工作原理：<p>
交叉注意力机制涉及两个序列的交互，其中一个序列提供查询（Query）向量，而另一个序列提供键（Key）和值（Value）向量。这种配置允许模型在两个序列之间建立联系，通过匹配和关联不同序列中的信息，从而在它们之间建立相关性。<p>
(2)作用：<p>
  - **跨模态学习**：Cross Attention在多模态学习中尤为重要，因为它能够将不同模态（如文本、图像、声音）的信息进行有效融合。
  - **交互式解码**：在序列到序列的任务中，Cross Attention使得解码器能够利用编码器提供的上下文信息，从而更准确地生成目标序列。
  - **灵活性**：Cross Attention提供了灵活性，因为它允许模型动态地关注另一个序列中与当前任务最相关的部分。<p>
<p>(3)主要特点：<p>
  - 信息来源：交叉注意力的信息交互发生在两个不同的序列之间，其中查询向量来自一个序列，而键和值向量来自另一个序列。<p>
  - 应用场景：交叉注意力广泛应用于那些需要建立跨序列依赖关系的任务。例如，在Transformer模型的解码器部分，交叉注意力机制使得解码器的查询向量能够与编码器生成的隐藏状态（作为键和值）进行交互，进而生成序列的下一个元素。<p>
<p>通过这种方式，交叉注意力为模型提供了在不同数据源之间建立复杂关联的能力，增强了模型在处理多模态数据和序列转换任务时的表现。
    
    
## 4.3 Self Attention 和 Cross Attention 的对比


| -------- | Self Attention| Cross Attention |
| -------- | -------- | -------- |
| 输入来源     | 在Self Attention中，Query、Key和Value均来源于同一个序列。这意味着模型是在内部进行信息的自我比较和关联。     | 在Cross Attention中，Query来自于一个序列，而Key和Value则来自于另一个不同的序列。这种配置允许模型在不同的数据源之间建立联系。     |
| 信息交互对象     | Self Attention使得序列中的每个元素都能够关注序列中的所有其他元素，并基于这种关注来更新自己的表示。     | Cross Attention则允许来自一个序列的元素（通过Query）关注另一个序列中的所有元素（通过Key和Value），从而实现跨序列的信息融合。     |
| 应用场景     | Self Attention广泛应用于需要理解序列内部复杂依赖关系的场景，例如在自然语言处理中，用于捕捉句子中单词之间的相互作用。     | Cross Attention适用于那些需要在不同序列之间建立联系的场合，如机器翻译中的编码器和解码器之间的交互，或者在多模态学习中，将文本信息与图像特征对齐。     |
| 特征捕捉     | Self Attention能够捕捉并编码序列内部的全局依赖关系，使得每个位置的表示都融入了序列中其他位置的信息。    | Cross Attention则专注于捕捉并编码不同序列之间的全局依赖关系，使得一个序列的表示能够反映另一个序列中的相关信息。     |
    
# 5. 参考链接
1、https://www.cnblogs.com/zhubolin/p/17156347.html
<p>2、https://github.com/xmu-xiaoma666/External-Attention-pytorch
<p>3、交叉注意力机制CrossAttention：https://blog.csdn.net/m0_63097763/article/details/132293568
<p>4、论文解读——交叉注意力融合2024经典论文（配套模块和代码）：https://maxin.blog.csdn.net/article/details/138181060
<p>5、归一化：https://hwcoder.top/Manual-Coding-2
<p>6、CrossViT：https://arxiv.org/abs/2103.14899v2
