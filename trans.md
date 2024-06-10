# S-LORA: SERVING THOUSANDS OF CONCURRENT LORA ADAPTERS

### ABSTRACT

The “pretrain-then-finetune” paradigm is commonly adopted in the deployment of large language models. LowRank Adaptation (LoRA), a parameter-efficient fine-tuning method, is often employed to adapt a base model to a multitude of tasks, resulting in a substantial collection of LoRA adapters derived from one base model. We observe that this paradigm presents significant opportunities for batched inference during serving. To capitalize on these opportunities, we present S-LoRA, a system designed for the scalable serving of many LoRA adapters. S-LoRA stores all adapters in the main memory and fetches the adapters used by the currently running queries to the GPU memory. To efficiently use the GPU memory and reduce fragmentation, S-LoRA proposes Unified Paging. Unified Paging uses a unified memory pool to manage dynamic adapter weights with different ranks and KV cache tensors with varying sequence lengths. Additionally, S-LoRA employs a novel tensor parallelism strategy and highly optimized custom CUDA kernels for heterogeneous batching of LoRA computation. Collectively, these features enable S-LoRA to serve thousands of LoRA adapters on a single GPU or across multiple GPUs with a small overhead. Compared to state-of-the-art libraries such as HuggingFace PEFT and vLLM (with naive support of LoRA serving), S-LoRA can improve the throughput by up to 4 times and increase the number of served adapters by several orders of magnitude. As a result, S-LoRA enables scalable serving of many task-specific fine-tuned models and offers the potential for large-scale customized fine-tuning services. The code is available at https://github.com/S-LoRA/S-LoRA.

### 摘要

“预训练-微调”范式通常用于大语言模型的部署。低秩适应（LoRA）是一种参数高效的微调方法，常用于将一个基础模型适应于多种任务，从而产生大量基于同一基础模型的LoRA adapter。我们发现，这种范式在服务过程中为批量推理提供了显著的机会。为了抓住这些机会，我们提出了S-LoRA，一个旨在大规模服务众多LoRA adapter的系统。S-LoRA将所有adapter存储在主存中，并将当前运行查询所使用的adapter加载到GPU存储中。为了高效利用GPU存储并减少碎片化，S-LoRA提出了统一分页（Unified Paging）。统一分页使用统一内存池来管理具有不同秩的动态adapter权重和具有不同序列长度的KV缓存张量。此外，S-LoRA采用了一种新颖的张量并行策略和高度优化的定制CUDA内核，用于异构批处理的LoRA计算。总的来说，这些功能使S-LoRA能够在单个GPU或多个GPU上以小额开销服务数千个LoRA adapter。与诸如HuggingFace PEFT和vLLM（对LoRA服务提供原始支持）等最先进的库相比，S-LoRA可以将吞吐量提高至4倍，并将服务的adapter数量增加数个数量级。因此，S-LoRA实现了众多任务特定微调模型的可扩展服务，并为大规模定制微调服务提供了潜力。代码可在https://github.com/S-LoRA/S-LoRA获取。

### Original Text

**1 INTRODUCTION**

Large language models (LLMs) have become ubiquitous in modern applications, ranging from natural language processing to more general tasks (OpenAI, 2023; Touvron et al., 2023b; Alayrac et al., 2022). Within these domains, LLMs have consistently demonstrated superior performance, especially when fine-tuned for specific tasks (Kenton & Toutanova, 2019; Houlsby et al., 2019; Ouyang et al., 2022). This “pretrain-then-finetune” paradigm has led to the proliferation of numerous fine-tuned variants of a single base LLM, each tailored to a specific task or domain.

When scaling the fine-tuning of a base model for numerous tasks, such as personalized assistants, which could involve thousands or millions of users, the associated training and serving costs can become substantial. To address this, several parameter-efficient fine-tuning methods have been developed. A prime exemplar is Low-Rank Adaptation (LoRA) (Hu et al., 2021), which enables efficient fine-tuning by updating only low-rank additive matrices. These matrices consist of a small number of parameters, referred to as adapter weights. LoRA has shown that by fine-tuning just these adapter weights, it is possible to achieve performance on par with full-weight fine-tuning. However, despite considerable research into fine-tuning, the question of how to serve these fine-tuned variants at scale remains unexplored.

One of the key innovations in the LoRA paper was the elimination of adapter inference latency by directly merging the adapter with the model parameters. Additionally, to support multiple models on a single machine, the same paper proposes swapping adapters by adding and subtracting LoRA weights from the base model. While this approach enables low-latency inference for a single adapter and serial execution across adapters, it significantly reduces overall serving throughput and increases total latency when serving multiple adapters concurrently. Moreover, the paper does not consider the opportunity to leverage host memory to increase the number of adapters hosted by a single machine.

### 翻译内容

**1 引言**

大型语言模型（LLMs）已广泛应用于现代应用程序中，从自然语言处理到更广泛的任务（OpenAI, 2023; Touvron et al., 2023b; Alayrac et al., 2022）。在这些领域，LLMs始终表现出卓越的性能，尤其是在针对特定任务进行微调时（Kenton & Toutanova, 2019; Houlsby et al., 2019; Ouyang et al., 2022）。这种“预训练-微调”范式导致了单一基础LLM的众多微调变体的激增，每种变体都针对特定任务或领域进行了定制。

当对基础模型进行大规模微调以应对众多任务时，比如个性化助手，可能涉及成千上万或数百万用户，相关的训练和服务成本会变得非常高。为了解决这个问题，开发了几种参数高效的微调方法。一个主要的例子是低秩适应（LoRA）（Hu et al., 2021），它通过只更新低秩加性矩阵实现了高效微调。这些矩阵由少量参数组成，被称为adapter权重。LoRA已经表明，通过仅微调这些adapter权重，可以达到与全权重微调相当的性能。然而，尽管在微调方面进行了大量研究，但如何大规模服务这些微调变体的问题仍未得到探索。

LoRA论文中的一个关键创新是通过直接将adapter与模型参数合并来消除adapter推理延迟。此外，为了在一台机器上支持多个模型，同一篇论文提出了通过加减LoRA权重来交换adapter的方法。虽然这种方法能够实现单个adapter的低延迟推理和跨adapter的串行执行，但在同时服务多个adapter时，它显著降低了整体服务吞吐量并增加了总延迟。此外，论文没有考虑利用主存来增加单台机器托管的adapter数量的机会。

### Original Text

In this paper, we study how to scalably serve thousands of LoRA adapters on a single machine. We observe that the shared base model, which underpins numerous LoRA adapters, presents a substantial opportunity for batched inference. To achieve high-throughput multi-adapter serving, it is advantageous to separate the batchable base model computation from individual LoRA computations.

While leveraging batching in the base model is straightforward (as all queries share the base model), extending batching to the adapters is challenging. First, serving many LoRA adapters simultaneously requires efficient memory management. Since GPU memory is limited, we must store adapter weights outside the GPU and dynamically fetch them when needed. However, dynamically loading and unloading adapters of varying sizes, coupled with the dynamic allocation and deallocation of KV cache tensors for requests with different sequence lengths, can lead to significant memory fragmentation and I/O overhead. Second, apart from the easily batchable base model computation, the separated computation of many adapters with distinct ranks in noncontiguous memory is challenging to batch and demands the development of new computation kernels. Third, leveraging multiple GPUs on a single machine requires novel parallelism strategies to accommodate the added LoRA weights and computations. It is essential to carefully design this strategy to minimize communication and memory overheads.

To this end, we introduce S-LoRA, a scalable LoRA serving system. S-LoRA exploits batching opportunities, efficiently manages both host and GPU memory, and orchestrates parallelism across multiple GPUs. The primary contributions of S-LoRA are summarized as follows:
- **Unified Paging**: To reduce memory fragmentation and increase batch size, S-LoRA introduces a unified memory pool. This pool manages dynamic adapter weights and KV cache tensors by a unified paging mechanism.
- **Heterogeneous Batching**: To minimize the latency overhead when batching different adapters of varying ranks, S-LoRA employs highly optimized custom CUDA kernels. These kernels operate directly on non-contiguous memory and align with the memory pool design, facilitating efficient batched inference for LoRA.
- **S-LoRA TP**: To ensure effective parallelization across multiple GPUs, S-LoRA introduces a novel tensor parallelism strategy. This approach incurs minimal communication cost for the added LoRA computation compared to that of the base model. This is realized by scheduling communications on small intermediate tensors and fusing the large ones with the communications of the base model.

### 翻译内容

在本文中，我们研究了如何在一台机器上扩展地服务数千个LoRA adapter。我们观察到，共享的基础模型支持众多LoRA adapter，这为批量推理提供了显著的机会。为了实现高吞吐量的多adapter服务，将可批处理的基础模型计算与单个LoRA计算分离是有利的。

虽然在基础模型中利用批处理是直截了当的（因为所有查询共享基础模型），但将批处理扩展到adapter是具有挑战性的。首先，同时服务许多LoRA adapter需要高效的内存管理。由于GPU内存是有限的，我们必须将adapter权重存储在GPU外，并在需要时动态获取它们。然而，动态加载和卸载不同大小的adapter，再加上为具有不同序列长度的请求动态分配和释放KV缓存张量，可能导致显著的内存碎片和I/O开销。其次，除了易于批处理的基础模型计算外，在非连续内存中分离计算具有不同秩的许多adapter也是难以批处理的，并需要开发新的计算内核。第三，在一台机器上利用多个GPU需要新的并行策略，以容纳增加的LoRA权重和计算。必须仔细设计这一策略，以最小化通信和内存开销。

为此，我们引入了S-LoRA，一个可扩展的LoRA服务系统。S-LoRA利用批处理机会，高效管理主存和GPU内存，并协调多个GPU的并行性。S-LoRA的主要贡献总结如下：
- **统一分页**：为了减少内存碎片并增加批量大小，S-LoRA引入了一个统一的内存池。该内存池通过统一分页机制管理动态adapter权重和KV缓存张量。
- **异构批处理**：为了在批处理不同秩的adapter时最小化延迟开销，S-LoRA采用了高度优化的定制CUDA内核。这些内核直接在非连续内存上操作，并与内存池设计对齐，促进了LoRA的高效批量推理。
- **S-LoRA TP**：为了确保在多个GPU上有效并行化，S-LoRA引入了一种新颖的张量并行策略。与基础模型的计算相比，这种方法在增加的LoRA计算上产生的通信成本很小。这是通过在小的中间张量上安排通信并将大的张量与基础模型的通信融合来实现的。

### Original Text

We evaluate S-LoRA by serving Llama-7B/13B/30B/70B. Results show that S-LoRA can serve thousands of LoRA adapters on a single GPU or across multiple GPUs with a small overhead. When compared to the state-of-the-art parameter-efficient fine-tuning library, Huggingface PEFT, S-LoRA can enhance throughput by up to 30×. In comparison to the high-throughput serving system vLLM using a naive support of LoRA serving, S-LoRA can improve throughput by up to 4× and increase the number of served adapters by several orders of magnitude.

### 翻译内容

我们通过服务Llama-7B/13B/30B/70B来评估S-LoRA。结果表明，S-LoRA可以在单个GPU或多个GPU上以小额开销服务数千个LoRA adapter。与最先进的参数高效微调库Huggingface PEFT相比，S-LoRA可以将吞吐量提高至30倍。相比于使用LoRA服务的简单支持的高吞吐量服务系统vLLM，S-LoRA可以将吞吐量提高至4倍，并将服务的adapter数量增加数个数量级。

### Original Text

**2 BACKGROUND**

Low-Rank Adaptation (LoRA) (Hu et al., 2021) is a parameter-efficient fine-tuning method designed to adapt pre-trained large language models to new tasks. The motivation behind LoRA stems from the low intrinsic dimensionality of model updates during adaptation. In the training phase, LoRA freezes the weights of a pre-trained base model and adds trainable low-rank matrices to each layer. This approach significantly reduces the number of trainable parameters and memory consumption. When compared to full parameter fine-tuning, LoRA can often reduce the number of trainable parameters by orders of magnitude (e.g., 10000×) while retaining comparable accuracy. For the inference phase, the original paper suggests merging the low-rank matrices with the weights of the base model. As a result, there is no added overhead during inference, setting it apart from previous adapters like (Houlsby et al., 2019) or prompt tuning methods such as (Lester et al., 2021).

Formally, for a pre-trained weight matrix \( W \in \mathbb{R}^{h \times d} \), LoRA introduces the update as \( W' = W + AB \), where \( A \in \mathbb{R}^{h \times r} \), \( B \in \mathbb{R}^{r \times d} \), and the rank \( r \ll \min(h, d) \). If the forward pass of a base model is defined by \( h = xW \), then after applying LoRA, the forward pass becomes:
\[ h = xW' = x(W + AB) \]
\[ = xW + xAB. \]
Typically, this adjustment is only applied to the query, key, value, and output projection matrices in the self-attention module, excluding the feed-forward module.

Because LoRA greatly reduces the training and weight storage costs, it has been widely adopted by the community, and people have created hundreds of thousands of LoRA adapters for pre-trained large language models and diffusion models (Mangrulkar et al., 2022).

### 翻译内容

**2 背景**

低秩适应（LoRA）（Hu et al., 2021）是一种参数高效的微调方法，旨在将预训练的大型语言模型适应新任务。LoRA背后的动机源于模型更新过程中固有的低维特性。在训练阶段，LoRA冻结预训练基础模型的权重，并在每一层添加可训练的低秩矩阵。这种方法显著减少了可训练参数的数量和内存消耗。与全参数微调相比，LoRA通常可以将可训练参数的数量减少几个数量级（例如，10000倍），同时保持相当的准确性。在推理阶段，原论文建议将低秩矩阵与基础模型的权重合并。因此，在推理过程中没有额外的开销，这使其区别于之前的adapter（如Houlsby et al., 2019）或提示微调方法（如Lester et al., 2021）。

形式上，对于一个预训练的权重矩阵( $W \in \mathbb{R}^{h \times d}$ )，LoRA引入了更新 ( $W' = W + AB $)，其

$  A \in \mathbb{R}^{h \times r} $，$B \in \mathbb{R}^{r \times d} $，并且秩 $r \ll \min(h, d) $。如果基础模型的前向传播定义为$  h = xW $，那么应用LoRA后，前向传播变为：
$ h = xW' = x(W + AB) \\= xW + xAB. $
通常，这种调整仅应用于自注意模块中的查询、键、值和输出投影矩阵，不包括前馈模块。

由于LoRA极大地降低了训练和权重存储成本，它被广泛地采用，社区已经为预训练的大型语言模型和扩散模型创建了数十万个LoRA adapter（Mangrulkar et al., 2022）。



### Original Text

**2.1 Serving Large Language Models**

Most large language models (LLMs) are based on the transformer architecture (Vaswani et al., 2017). The number of parameters in an LLM ranges from several billion to several trillion (Brown et al., 2020; Chowdhery et al., 2022; Fedus et al., 2022), corresponding to disk sizes spanning several gigabytes to even terabytes. This scale results in LLM serving having significant computational and memory demands.

Additionally, the inference process for LLMs requires iterative autoregressive decoding. Initially, the model carries out a forward pass to encode the prompt. Following this, it decodes the output one token at a time. The sequential process makes decoding slow. Since each token attends to the hidden states of all its preceding tokens, it becomes essential to store the hidden states of all previous tokens. This storage is referred to as the “KV cache”. Such a mechanism adds to the memory overhead and causes the decoding process to be more memory-intensive than computation-intensive.

The challenges become even more pronounced in online settings, where requests of varying sequence lengths arrive dynamically. To accommodate such dynamic incoming requests, Orca (Yu et al., 2022) introduces a method of fine-grained, iteration-level scheduling. Instead of scheduling at the request level, Orca batches at the token level. This approach allows for the continuous addition of new requests to the currently running batch, resulting in substantially higher throughput. vLLM (Kwon et al., 2023) further optimizes Orca’s memory efficiency using PagedAttention. PagedAttention adopts concepts from virtual memory and paging in operating systems and manages the storage and access of dynamic KV cache tensors in a paged fashion. This method efficiently reduces fragmentation, facilitating larger batch sizes and higher throughput.

When serving very large models that exceed the memory capacity of a single GPU, or when there are stringent latency requirements, it is necessary to parallelize the model across multiple GPUs. Several model parallelism methods have been proposed, such as tensor parallelism (Shoeybi et al., 2019), sequence parallelism (Korthikanti et al., 2023), pipeline parallelism (Huang et al., 2019), and their combinations (Narayanan et al., 2021; Zheng et al., 2022).

### 翻译内容

**2.1 服务大型语言模型**

大多数大型语言模型（LLM）基于Transformer架构（Vaswani et al., 2017）。LLM中的参数数量从数十亿到数万亿不等（Brown et al., 2020；Chowdhery et al., 2022；Fedus et al., 2022），相应的磁盘大小从数GB到数TB不等。这种规模导致LLM服务具有显著的计算和内存需求。

此外，LLM的推理过程需要迭代自回归解码。最初，模型进行一次前向传递来编码提示。接下来，它一次解码一个token。这种顺序过程使解码变慢。由于每个token都要关注其所有前面token的隐藏状态，因此必须存储所有先前token的隐藏状态。这种存储称为“KV缓存”。这种机制增加了内存开销，使解码过程比计算过程更内存密集。

在在线环境中，挑战变得更加明显，在这种环境中，不同序列长度的请求动态到达。为了适应这种动态到达的请求，Orca（Yu et al., 2022）引入了一种细粒度的迭代级调度方法。Orca不是在请求级别进行调度，而是在token级别进行批处理。这种方法允许将新请求连续添加到当前运行的批处理中，从而显著提高吞吐量。vLLM（Kwon et al., 2023）通过使用PagedAttention进一步优化Orca的内存效率。PagedAttention借用了操作系统中的虚拟内存和分页概念，以分页方式管理动态KV缓存张量的存储和访问。该方法有效减少了碎片化，促进了更大的批处理大小和更高的吞吐量。

在服务超过单个GPU内存容量的非常大型模型时，或者在有严格延迟要求的情况下，有必要在多个GPU上并行化模型。已经提出了几种模型并行方法，如张量并行（Shoeybi et al., 2019）、序列并行（Korthikanti et al., 2023）、流水线并行（Huang et al., 2019）及其组合（Narayanan et al., 2021；Zheng et al., 2022）。

### Original Text

**3 OVERVIEW OF S-LORA**

S-LoRA encompasses three principal components of innovation. In Section 4, we introduce our batching strategy, which decomposes the computation between the base model and the LoRA adapters. Additionally, we discuss adapter clustering and admission control when scheduling the requests. The ability to batch across concurrent adapters introduces new challenges around memory management. In Section 5, we generalize PagedAttention (Kwon et al., 2023) to Unified Paging, which supports dynamically loading LoRA adapters. This approach uses a unified memory pool to store the KV caches and adapter weights in a paged fashion, which can reduce fragmentation and balance the dynamic changing size of the KV caches and adapter weights. In Section 6, we introduce our new tensor parallelism strategy that enables us to efficiently decouple the base model and LoRA adapters.

### 翻译内容

**3 S-LoRA概述**

S-LoRA包含三个主要创新组件。在第4节中，我们介绍了分批策略，该策略将计算分解为基础模型和LoRA adapter。此外，我们讨论了调度请求时的adapter聚类和准入控制。能够跨并发adapter进行批处理，带来了内存管理方面的新挑战。在第5节中，我们将PagedAttention（Kwon et al., 2023）推广为统一分页，支持动态加载LoRA adapter。这种方法使用统一的内存池以分页方式存储KV缓存和adapter权重，这可以减少碎片化并平衡KV缓存和adapter权重动态变化的大小。在第6节中，我们介绍了新的张量并行策略，使我们能够有效地解耦基础模型和LoRA adapter。

![image-20240610211941069](/Users/miniast/Library/Application Support/typora-user-images/image-20240610211941069.png)

![image-20240610212000793](/Users/miniast/Library/Application Support/typora-user-images/image-20240610212000793.png)

### Original Text

**4 BATCHING AND SCHEDULING**

**4.1 Batching**

Our batching strategy aims to support online and high-throughput serving of many LoRA adapters simultaneously. For a single adapter, the method recommended by (Hu et al., 2021) is to merge the adapter weights into the base model weights, resulting in a new model (see Eq. 1). This has the advantage that there is no additional adapter overhead during inference, since the new model has the same number of parameters as the base model. In fact, this was a prominent feature of the original LoRA work.

However, when there are multiple adapters, merging the weights into the base model leads to multiple weight copies and missed batching opportunities. Directly merging the models requires maintaining many copies of the full language model. In the original LoRA paper, the authors proposed adding and subtracting LoRA weights on the fly to enable serving multiple models without increasing the memory overhead. However, this approach doesn’t support concurrent inference on separate LoRA adapters and therefore limits batching opportunities.

In this paper, we show that merging LoRA adapters into the base model is inefficient for the multi-LoRA high-throughput serving setting. Instead, we propose computing the LoRA computation xAB on-the-fly as shown in Eq. 2. This avoids weight duplication and enables batching of the more costly xW operation. But this approach also increases the computation overhead. However, because the cost of xAB is substantially lower than xW and there is a considerable savings from batching xW across different adapters, we show that the savings far exceed the additional overhead.

Unfortunately, directly implementing the factored computation of the base model and individual LoRA adapters using the batch GEMM kernel from the existing BLAS libraries would require significant padding and result in poor hardware utilization. This is because of the heterogeneity of sequence lengths and adapter ranks. In S-LoRA, we batch the computation of the base model and then employ custom CUDA kernels to execute the additional xAB for all adapters separately. This process is illustrated by Figure 1. Instead of naively using padding and using the batch GEMM kernel from the BLAS library for the LoRA computation, we implement custom CUDA kernels for more efficient computation without padding. In Subsection 5.3, we discuss the implementation details.

While the number of LoRA adapters can be large if we store them in main memory, the number of LoRA adapters needed for the currently running batch is manageable, because the batch size is bounded by the GPU memory. To take advantage of this, we store all LoRA adapters in the main memory and fetch only the LoRA adapters needed for the currently running batch to the GPU RAM when running the inference for that batch. In this case, the maximum number of adapters that can be served is bounded by the main memory size. This process is illustrated by Figure 2. To achieve high-throughput serving, we adopt the iteration-level scheduling batching strategy from Orca (Yu et al., 2022). In this approach, requests are scheduled at the token level. We immediately incorporate a new request into the running batch if space is available. The request will exit the batch once it reaches the maximum number of generated tokens or fulfills other stopping criteria. This process reduces GPU memory usage but introduces new memory management challenges. In Section 5, we will discuss our techniques to manage memory efficiently.

### 翻译内容

**4 分批与调度**

**4.1 分批**

我们的分批策略旨在支持同时在线和高吞吐量地服务许多LoRA adapter。对于单个adapter，(Hu et al., 2021) 推荐的方法是将adapter权重合并到基础模型权重中，生成一个新模型（见方程1）。这种方法的优点是在推理过程中没有额外的adapter开销，因为新模型的参数数量与基础模型相同。实际上，这是原始LoRA工作的一个显著特点。

然而，当有多个adapter时，将权重合并到基础模型中会导致多个权重副本和错失的批处理机会。直接合并模型需要维护许多完整语言模型的副本。在原始LoRA论文中，作者提出了动态添加和减去LoRA权重，以便在不增加内存开销的情况下服务多个模型。然而，这种方法不支持对不同LoRA adapter的并行推理，因此限制了批处理的机会。

在本文中，我们展示了将LoRA adapter合并到基础模型中对于多LoRA高吞吐量服务设置是低效的。相反，我们提出按需计算LoRA计算xAB，如方程2所示。这避免了权重重复，并且可以对更昂贵的xW操作进行批处理。但这种方法也增加了计算开销。然而，由于xAB的成本远低于xW，并且通过对不同adapter的xW进行批处理可以节省大量成本，我们展示了节省的成本远超过额外的开销。

不幸的是，使用现有BLAS库中的批量GEMM内核直接实现基础模型和单个LoRA adapter的分解计算需要大量填充，并导致硬件利用率低下。这是因为序列长度和adapter等级的异质性。在S-LoRA中，我们首先对基础模型的计算进行批处理，然后使用自定义CUDA内核分别执行所有adapter的额外xAB。这一过程如图1所示。我们没有天真地使用填充和BLAS库中的批量GEMM内核进行LoRA计算，而是实现了自定义CUDA内核，以便在没有填充的情况下更有效地进行计算。在小节5.3中，我们讨论了实现细节。

虽然如果我们将LoRA adapter存储在主内存中，LoRA adapter的数量可能很大，但当前运行批次所需的LoRA adapter数量是可控的，因为批次大小受GPU内存的限制。为了利用这一点，我们将所有LoRA adapter存储在主内存中，并在运行该批次的推理时仅将当前运行批次所需的LoRA adapter提取到GPU内存中。在这种情况下，可以服务的最大adapter数量受主内存大小的限制。这一过程如图2所示。为了实现高吞吐量服务，我们采用了Orca (Yu et al., 2022) 的迭代级调度批处理策略。在这种方法中，请求在token级别进行调度。如果有空间，我们会立即将新请求并入正在运行的批处理中。一旦请求达到生成的最大token数量或满足其他停止标准，它将退出批处理。这一过程减少了GPU内存使用，但引入了新的内存管理挑战。在第5节中，我们将讨论管理内存的有效技术。

### Original Text

**4.2 adapter Clustering**

To enhance batching efficiency, one potential strategy is reducing the number of active adapters in a running batch. By using fewer adapters, there is an opportunity to allocate more memory to the KV cache, which in turn can facilitate larger batch sizes. Given the common memory capacities of GPUs, they are often underutilized while decoding. Consequently, increasing the batch size can lead to higher throughput. A direct approach to reducing the number of adapters in a running batch is to prioritize batching requests that use the same adapter, a strategy we term “adapter clustering”. However, adapter clustering comes with its own set of trade-offs. For example, it can hurt the average latency or fairness among adapters. We provide an ablation study in Appendix A to illustrate how throughput and latency change according to the cluster size.

### 翻译内容

**4.2 adapter聚类**

为了提高批处理效率，一个潜在的策略是减少运行批次中的活动adapter数量。通过使用更少的adapter，有机会为KV缓存分配更多内存，从而促进更大的批次规模。考虑到GPU的常见内存容量，它们在解码时通常未充分利用。因此，增加批次规模可以提高吞吐量。减少运行批次中adapter数量的直接方法是优先处理使用相同adapter的请求，这一策略我们称之为“adapter聚类”。然而，adapter聚类也带来一系列权衡。例如，它可能会影响平均延迟或adapter之间的公平性。我们在附录A中提供了一个消融研究，说明吞吐量和延迟如何根据聚类大小变化。

### Original Text

**4.3 Admission Control**

In S-LoRA, we also applied an admission control strategy to sustain good attainment when the traffic is higher than the serving system capacity. A serving system is typically characterized by a service level objective (SLO) which specifies the desired latency of processing requests. If the serving system has fixed capacity, it must implement an admission control mechanism, that drops a request, if the system cannot meet its SLO. Otherwise, if no request is dropped, and the number of incoming requests is larger than the system capacity for long enough, the serving system is bound to violate the SLO. We implemented an abort strategy to mimic admission control in S-LoRA, called early abort strategy. Intuitively, we estimate the set of latest requests that we can serve in SLO, and then serve them in the order of arrival time. More implementation details and mathematical justifications are deferred to Appendix B.

### 翻译内容

**4.3 访问控制**

在 S-LoRA 中，我们还应用了一种访问控制策略，以在流量超过服务系统容量时保持良好的性能。服务系统通常以服务水平目标 (SLO) 为特征，该目标指定了处理请求的期望延迟。如果服务系统的容量是固定的，则必须实施访问控制机制，当系统无法满足其 SLO 时，会丢弃请求。否则，如果没有请求被丢弃，而传入请求的数量在足够长的时间内超过系统容量，则服务系统必然会违反 SLO。我们在 S-LoRA 中实施了一种模拟访问控制的中止策略，称为早期中止策略。直观上，我们估计能够在 SLO 内服务的最新请求集，然后按到达时间顺序处理这些请求。更多的实现细节和数学依据请参见附录 B。

### Original Text

**5 Memory Management**

Compared to serving a single base model, serving multiple LoRA adapters simultaneously presents new memory management challenges. To support many adapters, S-LoRA stores them in the main memory and dynamically loads the adapter weights needed for the currently running batch into GPU RAM. During this process, there are two noticeable challenges. The first is memory fragmentation, resulting from the dynamic loading and offloading adapter weights of various sizes. The second is the latency overhead introduced by adapter loading and offloading. To tackle these challenges efficiently, we propose Unified Paging and overlap the I/O with computation by prefetching adapter weights.

### 翻译内容

**5 内存管理**

相比于服务单个基础模型，同时服务多个 LoRA adapter带来了新的内存管理挑战。为了支持多个adapter，S-LoRA 将它们存储在主内存中，并动态加载当前运行批次所需的adapter权重到 GPU 内存中。在此过程中，有两个明显的挑战。第一个是由于动态加载和卸载各种大小的adapter权重所导致的内存碎片。第二个是adapter加载和卸载引入的延迟开销。为了解决这些挑战，我们提出了统一分页（Unified Paging）方法，并通过预取adapter权重将 I/O 与计算重叠。

### Original Text

**5.1 Unified Paging**

Understanding the nature of adapter weights is essential for optimizing memory usage. Our primary observation is that these dynamic adapter weights are analogous to dynamic KV caches in several ways:
- **Variable sizes and operations:** Just as the size of the KV cache fluctuates with the sequence length, the ranks of the active adapters can also depend on the choice of adapter associated with each request. KV caches are allocated when requests arrive and deallocated once the requests are completed. Similarly, adapter weights are loaded and cleared with each request. If not managed properly, this variability can result in fragmentation.
- **Dimensionality:** A KV cache tensor for a request in a layer has a shape of (S, H), where S denotes the sequence length and H represents the hidden dimension. Meanwhile, the shape of a LoRA weight is (R, H), with R standing for the rank and H the hidden dimension. Both share a dimension size of H that can be leveraged to reduce fragmentation.

Motivated by these parallels, we extend the idea of PagedAttention (Kwon et al., 2023) to Unified Paging which manages adapter weights in addition to the KV cache. Unified Paging uses a unified memory pool to jointly manage both KV cache and adapter weights. To implement this, we first allocate a large buffer statically for the memory pool. This buffer uses all available space except for the space occupied by the base model weights and temporary activation tensors. Both KV caches and adapter weights are stored in this memory pool in a paged manner, with each page corresponding to a vector of H. Thus, a KV cache tensor with a sequence length of S uses up S pages, while a LoRA weight tensor of rank R takes up R pages. Figure 3 illustrates the layout of our memory pool, where KV caches and adapter weights are stored interleaved and non-contiguously. This approach significantly reduces fragmentation, ensuring that adapters weights of various ranks can coexist with dynamic KV caches in a structured and systematic manner.

### 翻译内容

**5.1 统一分页**

了解adapter权重的性质对于优化内存使用至关重要。我们的主要观察是，这些动态adapter权重在多个方面与动态KV缓存相似：
- **可变大小和操作：** 正如KV缓存的大小随着序列长度而波动，激活的adapter的rank也可以根据每个请求关联的adapter的选择而变化。KV缓存在请求到达时分配，在请求完成后释放。类似地，adapter权重会随着每个请求的到来加载并在完成后清除。如果管理不当，这种可变性可能导致内存碎片。
- **维度：** 一层中针对请求的KV缓存张量的形状为(S, H)，其中S表示序列长度，H表示隐藏维度。而LoRA权重的形状为(R, H)，其中R代表rank，H为隐藏维度。两者共享一个维度H，这可以用来减少碎片。

受这些相似性的启发，我们将Paged Attention (Kwon et al., 2023) 的理念扩展到统一分页（Unified Paging），不仅管理KV缓存，还管理adapter权重。统一分页使用统一内存池共同管理KV缓存和adapter权重。为实现这一点，我们首先为内存池静态分配一个大缓冲区。此缓冲区使用除基础模型权重和临时激活张量占用的空间外的所有可用空间。KV缓存和adapter权重都以分页方式存储在此内存池中，每页对应于一个H向量。因此，具有序列长度S的KV缓存张量使用S页，而具有等级R的LoRA权重张量使用R页。图3展示了我们内存池的布局，其中KV缓存和adapter权重交错且非连续地存储。此方法显著减少了碎片，确保各种等级的adapter权重可以与动态KV缓存以结构化和系统化的方式共存。

### Original Text

**5.2 Prefetching and Overlapping**

Although the unified memory pool mitigates fragmentation, the I/O overhead from loading and offloading remains a concern—especially when dealing with numerous or large adapters. The latency introduced by waiting to load these adapters can compromise the efficiency of the system. To proactively address this issue, we introduce a dynamic prediction mechanism. While running the current decoding batch, we predict the adapters required for the next batch based on the current waiting queue. This prediction allows us to prefetch and store them in available memory. Such a forward-looking strategy keeps most of the adapters needed for the next batch already in place before running it, which reduces I/O time for adapter swapping.

### 翻译内容

**5.2 预取和重叠**

尽管统一内存池缓解了碎片化问题，但加载和卸载的I/O开销仍然是一个值得关注的问题，尤其是在处理大量或大型adapter时。等待加载这些adapter所引入的延迟可能会影响系统的效率。为了主动解决这个问题，我们引入了一种动态预测机制。在运行当前解码批次时，我们根据当前等待队列预测下一个批次所需的adapter。此预测使我们能够预取并将它们存储在可用内存中。这样一种前瞻性的策略可以在运行下一个批次之前，提前准备好大多数所需的adapter，从而减少adapter交换的I/O时间。

### Original Text

**5.3 Custom Kernels for Heterogeneous LoRA Batching on Non-Contiguous Memory**

Due to the design of the unified memory pool, the adapter weights are stored in non-contiguous memory. To run computations efficiently under this design, we implement custom CUDA kernels that support batching LoRA computations with varying ranks and sequence lengths in a non-contiguous memory layout.

In the prefill stage, the kernel handles a sequence of tokens and gathers adapter weights with different ranks from the memory pool. We call this kernel Multi-size Batched Gather Matrix-Matrix Multiplication (MBGMM). It is implemented in Triton (Tillet et al., 2019) with tiling.

In the decode stage, the kernel handles a single token and gathers adapter weights with different ranks from the memory pool. We call this kernel Multi-size Batched Gather Matrix-Vector Multiplication (MBGMV). We implemented two versions of this kernel: one in Triton and another by modifying an earlier version of Punica kernels (Chen, 2023) to extend support for non-contiguous memory, multiple ranks in a batch, and more fine-grained memory gathering. We found the latter one was faster, so we used it in the experiments.

Punica (Chen et al., 2023) is concurrent work on serving multiple LoRA adapters, which will be discussed in Section 8. In addition to Triton and Punica kernels, NVIDIA CUTLASS also provides high-performance kernels for grouped GEMM (NVIDIA) that can be used for heterogeneous batching.

### 翻译内容

**5.3 用于非连续内存上异构LoRA批处理的自定义内核**

由于统一内存池的设计，adapter权重存储在非连续的内存中。为了在这种设计下高效地运行计算，我们实现了自定义的CUDA内核，这些内核支持在非连续内存布局中对具有不同秩和序列长度的LoRA计算进行批处理。

在预填充阶段，内核处理一系列token并从内存池中收集具有不同秩的adapter权重。我们称这个内核为多尺寸批处理收集矩阵-矩阵乘法（MBGMM）。它在Triton (Tillet et al., 2019) 中通过平铺实现。

在解码阶段，内核处理单个token并从内存池中收集具有不同秩的adapter权重。我们称这个内核为多尺寸批处理收集矩阵-向量乘法（MBGMV）。我们实现了这个内核的两个版本：一个在Triton中实现，另一个通过修改早期版本的Punica内核（Chen, 2023）来扩展对非连续内存、多秩批处理以及更细粒度内存收集的支持。我们发现后者更快，所以在实验中使用了它。

Punica（Chen et al., 2023）是同时进行的关于多LoRA adapter服务的工作，这将在第8节讨论。除了Triton和Punica内核，NVIDIA CUTLASS也提供了用于异构批处理的高性能分组GEMM内核（NVIDIA）。

### 6 Tensor Parallelism

We design novel tensor parallelism strategies for batched LoRA inference to support multi-GPU inference of large transformer models. Tensor parallelism is the most widely used parallelism method because its single-program multiple-data pattern simplifies its implementation and integration with existing systems. Tensor parallelism can reduce the per-GPU memory usage and latency when serving large models. In our setting, the additional LoRA adapters introduce new weight matrices and matrix multiplications, which calls for new partition strategies for these added items.

### 6.1 Partition Strategy

Since the base model uses the Megatron-LM tensor parallelism strategy (Shoeybi et al., 2019), our approach aims to align the partition strategies of inputs and outputs of the added LoRA computation with those of the base model. In this way, we can minimize the communication costs by avoiding unnecessary communications and fusing some communications.

We use the feed-forward module (2-layer MLP) to illustrate our partition strategy. We will explain later how this strategy can easily be adapted to the self-attention layer. As depicted in Figure 4, the upper box illustrates the base model’s Megatron-LM partition strategy: the first weight matrix (W1) is column-partitioned, and the second (W2) is row-partitioned. An all-reduce communication is required to accumulate the partial sum from distributed devices.

The lower box illustrates the partitioning strategy for the added LoRA computation. The matrices A1 and B1 for the adapter of the first weight matrix (W1) are column-partitioned. An all-gather operation is used to collect the intermediate results. The matrices A2 and B2 for the adapter of the second weight (W2) are row-partitioned and column-partitioned, respectively. An all-reduce operation is used to sum up the intermediate results. Finally, the result from the LoRA computation is added to that from the base model (add 2). A single all-reduce operation is sufficient to accumulate the final results. It is worth noting that we are essentially fusing an all-gather operation for matmul 4 with the final all-reduce. To our knowledge, this parallelization strategy has not been studied before.

Next, we discuss adapting the strategy from the 2-layer MLP to the self-attention layer. Similar to the Megatron-LM strategy, we partition the head dimension of the self-attention layer. The query-key-value projection weight matrix can be seen as W1 in our example and the output projection weight matrix can be seen as W2 in our example.

### 6.2 Communication and Memory Cost Analysis

Let N be the number of devices, B be the number of tokens, h be the hidden size, and r be the adapter rank. The communication cost of the base model is one all-reduce, or \( \frac{2(N-1)Bh}{N} \). The communication cost of the added LoRA computation is three all-gather for query, key, and value projections, and one all-reduce for the output projection. Formally, it is \( 3 \frac{(N-1)Br}{N} + 2 \frac{(N-1)Br}{N} = 5 \frac{(N-1)Br}{N} \).

Under our strategy, the additional communication cost introduced by LoRA is negligible when compared to the communication cost of the base model, because \( r \ll h \). Intuitively, this is achieved by carefully scheduling communications on the small intermediate tensors of LoRA computation and fusing communications with base models.

In terms of memory usage, our strategy is optimal because we partition all weight matrices among all devices and there is no replicated weight matrix.

---

### 6 张量并行

我们设计了新颖的张量并行策略用于批处理LoRA推理，以支持大型Transformer模型的多GPU推理。张量并行是最广泛使用的并行方法，因为其单程序多数据模式简化了实现和与现有系统的集成。当服务于大型模型时，张量并行可以减少每个GPU的内存使用和延迟。在我们的设置中，额外的LoRA adapter引入了新的权重矩阵和矩阵乘法，这需要为这些新增项设计新的分区策略。

### 6.1 分区策略

由于基础模型使用了Megatron-LM张量并行策略（Shoeybi等，2019），我们的方法旨在使新增的LoRA计算的输入和输出的分区策略与基础模型的分区策略对齐。这样，我们可以通过避免不必要的通信并融合一些通信来最大限度地减少通信成本。

我们使用前馈模块（2层MLP）来说明我们的分区策略。稍后我们将解释如何将这一策略轻松适配到自注意力层。如图4所示，上框说明了基础模型的Megatron-LM分区策略：第一个权重矩阵（W1）是列分区的，第二个（W2）是行分区的。需要进行一次全规约通信以累积来自分布式设备的部分和。

下框说明了新增的LoRA计算的分区策略。第一个权重矩阵（W1）的adapter的矩阵A1和B1是列分区的。使用全收集操作来收集中间结果。第二个权重（W2）的adapter的矩阵A2和B2分别是行分区和列分区的。使用全规约操作来累积中间结果。最后，将LoRA计算的结果添加到基础模型的结果中（add 2）。一次全规约操作足以累积最终结果。值得注意的是，我们实际上将matmul 4的全收集操作与最终的全规约操作融合了。据我们所知，这种并行化策略以前未曾被研究过。

接下来，我们讨论如何将该策略从2层MLP适配到自注意力层。类似于Megatron-LM策略，我们对自注意力层的头维度进行分区。查询-键-值投影权重矩阵可以看作我们例子中的W1，输出投影权重矩阵可以看作我们例子中的W2。

### 6.2 通信和内存成本分析

设N为设备数，B为token数，h为隐藏层大小，r为adapter等级。基础模型的通信成本是一次全规约，即 \( \frac{2(N-1)Bh}{N} \)。新增LoRA计算的通信成本是查询、键和值投影的三次全收集，以及输出投影的一次全规约。形式上，它是 \( 3 \frac{(N-1)Br}{N} + 2 \frac{(N-1)Br}{N} = 5 \frac{(N-1)Br}{N} \)。

在我们的策略下，LoRA引入的额外通信成本在与基础模型的通信成本相比时可以忽略不计，因为 \( r \ll h \)。直观上，这是通过在LoRA计算的小中间张量上仔细调度通信并将通信与基础模型融合来实现的。

在内存使用方面，我们的策略是最优的，因为我们将所有权重矩阵在所有设备之间进行分区，没有重复的权重矩阵。

![image-20240610213338813](/Users/miniast/Library/Application Support/typora-user-images/image-20240610213338813.png)

### 7 Evaluation

We evaluate the performance of S-LoRA on both synthetic and real production workloads. S-LoRA is built on top of LightLLM (ModelTC, 2023), a single-model LLM serving system based on PyTorch (Paszke et al., 2019) and Triton (Tillet et al., 2019). We evaluate the scalability of S-LoRA by serving up to two thousand LoRA adapters simultaneously and compare it with other strong baselines. We then perform ablation studies to verify the effectiveness of individual components.

### 7.1 Setup

**Model.** We test the Llama model series (Touvron et al., 2023a; b), one of the most popular open large language models. We consider five different model and adapter configurations, which are listed in Table 1. Our optimizations can be easily adapted to other transformer-based architectures as well, such as GPT-3 (Brown et al., 2020) and PaLM (Chowdhery et al., 2022; Anil et al., 2023).

**Hardware.** We conduct tests on various hardware settings, including a single NVIDIA A10G GPU (24GB), a single A100 GPU (40GB), a single A100 GPU (80GB), and multiple A100 GPUs (40GB/80GB). The host’s main memory varies based on the GPU setup, ranging from 64 GB to 670 GB. We will show that S-LoRA can efficiently scale the number of adapters, limited only by the available main memory.

**Baselines.** We benchmark several variants of S-LoRA, HuggingFace PEFT (Mangrulkar et al., 2022), and vLLM (Kwon et al., 2023).
- **"HuggingFace PEFT"** is a library for training and running parameter-efficient fine-tuning models. It lacks advanced batching and memory management. We build a server using it that batches single adapter requests and switches adapter weights between batches.
- **"vLLM m-packed"** is a simple multi-model serving solution based on vLLM, a high-throughput serving system. Because vLLM does not support LoRA, we merge the LoRA weights into the base model and serve the multiple versions of the merged weights separately. To serve m LoRA adapters, we run m vLLM workers on a single GPU, where multiple workers are separate processes managed by NVIDIA MPS. We statistically allocate the GPU memory proportionally to the average request rate for each process.
- **"S-LoRA"** is S-LoRA with all the optimizations and it is using the first-come-first-serve scheduling strategy.
- **"S-LoRA-no-unify-mem"** is S-LoRA without the unified memory management.
- **"S-LoRA-bmm"** is S-LoRA without unified memory management and customized kernels. It copies the adapter weights to continuous memory space and performs batched matrix multiplication with padding.

**Metrics.** There are several metrics to measure the performance of serving systems, including latency and throughput. Following common practice, we report the throughput, average request latency, average first token latency, and SLO attainment. SLO attainment is defined as the percentage of requests that return the first token in 6 seconds. Additionally, we introduce a new metric termed user satisfaction (see Appendix B), which offers a more fine-grained analysis of the first token latency. Intuitively, a shorter first token latency gives higher satisfaction. The satisfaction becomes 0 if the first token latency exceeds the SLO.

---

### 7 评估

我们在合成和真实生产工作负载上评估了S-LoRA的性能。S-LoRA构建在LightLLM（ModelTC, 2023）之上，这是一个基于PyTorch（Paszke et al., 2019）和Triton（Tillet et al., 2019）的单模型LLM服务系统。我们通过同时服务多达两千个LoRA adapter来评估S-LoRA的可扩展性，并将其与其他强大的基线进行比较。然后，我们进行消融研究，以验证各个组件的有效性。

### 7.1 设置

**模型.** 我们测试了Llama模型系列（Touvron et al., 2023a; b），这是最受欢迎的开放大型语言模型之一。我们考虑了五种不同的模型和adapter配置，列在表1中。我们的优化也可以轻松适配到其他基于Transformer的架构，例如GPT-3（Brown et al., 2020）和PaLM（Chowdhery et al., 2022；Anil et al., 2023）。

**硬件.** 我们在各种硬件设置上进行了测试，包括单个NVIDIA A10G GPU（24GB），单个A100 GPU（40GB），单个A100 GPU（80GB）以及多个A100 GPU（40GB/80GB）。主机的主内存根据GPU设置的不同，从64 GB到670 GB不等。我们将展示S-LoRA可以有效地扩展adapter数量，唯一的限制是可用的主内存。

**基线.** 我们基准测试了几种S-LoRA的变体，HuggingFace PEFT（Mangrulkar et al., 2022）和vLLM（Kwon et al., 2023）。
- **"HuggingFace PEFT"** 是一个用于训练和运行参数高效微调模型的库。它缺乏高级的批处理和内存管理。我们使用它构建了一个服务器，该服务器批处理单个adapter请求并在批次之间切换adapter权重。
- **"vLLM m-packed"** 是一个基于vLLM的简单多模型服务解决方案，vLLM是一个高吞吐量服务系统。由于vLLM不支持LoRA，我们将LoRA权重合并到基础模型中，并分别服务多个合并权重版本。为了服务m个LoRA adapter，我们在单个GPU上运行m个vLLM工作进程，这些进程由NVIDIA MPS管理。我们根据每个进程的平均请求率按比例分配GPU内存。
- **"S-LoRA"** 是包含所有优化的S-LoRA，并使用先到先服务调度策略。
- **"S-LoRA-no-unify-mem"** 是没有统一内存管理的S-LoRA。
- **"S-LoRA-bmm"** 是没有统一内存管理和自定义内核的S-LoRA。它将adapter权重复制到连续的内存空间，并使用填充执行批量矩阵乘法。

**指标.** 有几个指标可以衡量服务系统的性能，包括延迟和吞吐量。按照常规做法，我们报告吞吐量、平均请求延迟、平均第一个token延迟和SLO达成率。SLO达成率定义为在6秒内返回第一个token的请求的百分比。此外，我们引入了一个新的指标，称为用户满意度（见附录B），它提供了第一个token延迟的更细粒度分析。直观地说，较短的第一个token延迟会带来更高的满意度。如果第一个token延迟超过SLO，满意度将变为0。

![image-20240610213544899](/Users/miniast/Library/Application Support/typora-user-images/image-20240610213544899.png)

### 7.2 End-to-End Results on Synthetic Workloads

**Workload Trace.** We generate synthetic workload traces using the Gamma process, which is commonly used in machine learning serving literature (Crankshaw et al., 2020; Li et al., 2023). Given \( n \) adapters, the requests for adapter \( i \) are modeled using a Gamma arrival process with a mean rate of \( \lambda_i \) and a coefficient of variance (CV) of \( cv \). The mean rate, \( \lambda_i \), adheres to a power-law distribution with an exponent \( \alpha \). The total request rate for all adapters is \( R \) requests per second. For the \( n \) adapters, we set their ranks based on the list provided in Table 1 with a round-robin method. Our tests cover various combinations of \( n \), \( \alpha \), \( R \), and \( cv \). For every request, the input and output lengths are sampled from uniform distributions \( U[I_l, I_u] \) and \( U[O_l, O_u] \) respectively. The default duration of a trace is 5 minutes. To conduct comprehensive experiments, we first pick a set of default parameters for generating workloads, as shown in Table 2. We then vary one of the \( n \), \( \alpha \), \( R \), and \( cv \) to see how each factor affects the performance.

**Comparison with Other Systems.** We compare S-LoRA with both vLLM-packed and HuggingFace PEFT for serving many LoRA adapters. The results are shown in Table 3. Remarkably, S-LoRA can serve 2,000 adapters simultaneously, maintaining minimal overhead for the added LoRA computation. In contrast, vLLM-packed needs to maintain multiple weight copies and can only serve fewer than 5 adapters due to the GPU memory constraint. The throughput of vLLM-packed is also much lower due to the missed batching opportunity. Although PEFT can swap adapters between batches, enabling it to handle a large number of adapters, its lack of advanced batching methods and memory management results in significantly worse performance. Overall, S-LoRA achieves a throughput up to 4x higher than vLLM-packed when serving a small number of adapters, and up to 30x higher than PEFT, while supporting a significantly larger number of adapters.

**Comparing with Own Variants.** Since no baseline system can efficiently scale to a large number of adapters, we now focus on comparing S-LoRA with its own variants. Figure 5 illustrates how they scale with the number of adapters. S-LoRA achieves noticeably higher throughput and lower latency compared to S-LoRA-bmm and S-LoRA-no-unify-mem. This implies that our memory pool and custom kernels are effective. When the number of adapters increases, the throughput of S-LoRA initially experiences a slight decline due to the overhead introduced by LoRA. However, once the number of adapters reaches a certain threshold (e.g., 100 in most experiments), the throughput of S-LoRA no longer decreases. This stability can be attributed to the fact that as the number of adapters grows, the number of activated adapters for the currently running batch remains unchanged, maintaining a constant overhead. Consequently, S-LoRA can scale to a much larger number of adapters without incurring additional overhead, constrained only by the available main memory.

Figure 6 demonstrates the variation in throughput, first token latency, and SLO attainment relative to the total request rate, revealing a pattern consistent with the aforementioned observations and underscoring the efficacy of our design.

---

### 7.2 合成工作负载的端到端结果

**工作负载跟踪.** 我们使用Gamma过程生成合成工作负载跟踪，这在机器学习服务文献中常用（Crankshaw et al., 2020；Li et al., 2023）。给定 \( n \) 个adapter，adapter \( i \) 的请求使用均值速率为 \( \lambda_i \) 和方差系数（CV）为 \( cv \) 的Gamma到达过程进行建模。均值速率 \( \lambda_i \) 遵循指数为 \( \alpha \) 的幂律分布。所有adapter的总请求速率为 \( R \) 请求每秒。对于 \( n \) 个adapter，我们根据表1中提供的列表采用循环法设置它们的等级。我们的测试涵盖了 \( n \)、 \( \alpha \)、 \( R \) 和 \( cv \) 的各种组合。对于每个请求，输入和输出长度分别从均匀分布 \( U[I_l, I_u] \) 和 \( U[O_l, O_u] \) 中采样。默认的跟踪持续时间为5分钟。为了进行全面的实验，我们首先选择一组默认参数来生成工作负载，如表2所示。然后，我们分别变化 \( n \)、 \( \alpha \)、 \( R \) 和 \( cv \) 中的一个，看看每个因素如何影响性能。

**与其他系统的比较.** 我们将S-LoRA与vLLM-packed和HuggingFace PEFT进行了比较，以服务于许多LoRA adapter。结果如表3所示。值得注意的是，S-LoRA可以同时服务2,000个adapter，并保持增加的LoRA计算的最小开销。相比之下，vLLM-packed需要维护多个权重副本，并且由于GPU内存限制只能服务少于5个adapter。由于错失了批处理机会，vLLM-packed的吞吐量也低得多。尽管PEFT可以在批次之间交换adapter，使其能够处理大量adapter，但由于缺乏高级批处理方法和内存管理，其性能显著较差。总体而言，S-LoRA在服务少量adapter时实现了比vLLM-packed高4倍的吞吐量，比PEFT高30倍，同时支持显著更多的adapter。

**与自身变体的比较.** 由于没有基线系统能够有效地扩展到大量adapter，我们现在专注于将S-LoRA与其自身的变体进行比较。图5说明了它们如何随着adapter数量的增加而扩展。与S-LoRA-bmm和S-LoRA-no-unify-mem相比，S-LoRA实现了明显更高的吞吐量和更低的延迟。这表明我们的内存池和自定义内核是有效的。当adapter数量增加时，S-LoRA的吞吐量最初会因LoRA引入的开销而略有下降。然而，一旦adapter数量达到一定阈值（例如，大多数实验中的100），S-LoRA的吞吐量不再下降。这种稳定性可归因于adapter数量增加时当前运行批次的激活adapter数量保持不变，从而保持恒定的开销。因此，S-LoRA可以扩展到更多的adapter，而不会引入额外的开销，仅受限于可用的主内存。

图6展示了吞吐量、第一个token延迟和SLO达成率相对于总请求速率的变化，揭示了与前述观察结果一致的模式，并强调了我们设计的有效性。

![image-20240610213738555](/Users/miniast/Library/Application Support/typora-user-images/image-20240610213738555.png)

![image-20240610213752292](/Users/miniast/Library/Application Support/typora-user-images/image-20240610213752292.png)

### 7.3 End-to-End Results on Real Workloads

**Real Workload Trace.** We construct real-world serving traces by downsampling from the traces of LMSYS Chatbot Arena (Zheng et al., 2023b;a), a website that serves multiple LLMs. The raw log from Arena does not concern LoRA adapters; it focuses on different base models. Nonetheless, we treat the distribution of different base models as if they were the distribution of different adapters of a single base model. The raw log can be sampled into traces that exhibit varying request rates, denoted as \( R \), and durations, represented by \( D \). To achieve this, we sample \( R \cdot D \) requests from the raw log and rescale the timestamps to fit within the range of \([0, D]\). The number of models \( n \) corresponds to the number of adapters. Furthermore, we set the adapter ranks based on Table 1 with a round-robin method.

Since we are using a real workload trace, there are no parameters such as \( \alpha \), \( \lambda_i \), or \( cv \). For consistency, we set the duration to 5 minutes. We adjust the request rate \( R \) to study its impact on performance metrics. In the sampled trace, the average input length is 85 tokens, the average output length is 165 tokens, and the number of adapters is around 26.

**Results.** Figure 7 shows the throughput and attainment results, which exhibit a similar pattern to the synthetic workloads. This consistency indicates that S-LoRA performs strongly under real-world workloads.

### 7.4 Multi-GPU Tensor Parallelism

We test the scalability of our tensor parallelism strategy by running: 
1. Llama-30B on two A100 (40GB) and four A100 (40GB) GPUs with 10 to 100 adapters.
2. Llama-70B on two A100 (80GB) and four A100 (80GB) GPUs with 10 adapters.

We then report the serving throughputs.

As depicted in Figure 8, the disparity between S-LoRA with and without LoRA communication is small. This suggests that the added LoRA communication in our strategy has a very small overhead. The figure further reveals that the communication overhead due to LoRA is less than the computational overhead it introduces. Furthermore, when transitioning from 2 GPUs to 4 GPUs, the serving throughput increases by more than 2 times. This significant increase can be attributed to the fact that the system is predominantly memory-bound in this context. Adding more GPUs alleviates memory constraints, leading to superlinear scaling.

**Conclusion.** The results verify both the minimal overhead and the scalability of our tensor parallelism strategy.

---

### 7.3 实际工作负载的端到端结果

**实际工作负载跟踪.** 我们通过从LMSYS Chatbot Arena (Zheng et al., 2023b;a) 的跟踪数据中抽样，构建了实际的服务跟踪。Arena的原始日志不涉及LoRA adapter，它侧重于不同的基础模型。然而，我们将不同基础模型的分布视为单一基础模型的不同adapter的分布。原始日志可以被抽样成显示不同请求率（用 \( R \) 表示）和持续时间（用 \( D \) 表示）的跟踪数据。为了实现这一点，我们从原始日志中抽取 \( R \cdot D \) 个请求，并重新调整时间戳以适应 \([0, D]\) 范围。模型数量 \( n \) 对应adapter数量。此外，我们根据表1中的轮转法设置adapter等级。

由于我们使用的是实际工作负载跟踪，因此没有参数如 \( \alpha \)、 \( \lambda_i \) 或 \( cv \)。为了保持一致性，我们将持续时间设置为5分钟。我们调整请求率 \( R \) 以研究其对性能指标的影响。在抽样的跟踪数据中，平均输入长度为85个token，平均输出长度为165个token，adapter数量约为26。

**结果.** 图7显示了吞吐量和达成率结果，表现出与合成工作负载类似的模式。这种一致性表明S-LoRA在实际工作负载下表现强劲。

### 7.4 多GPU张量并行

我们通过运行以下测试我们的张量并行策略的可扩展性：
1. 在两个A100 (40GB) 和四个A100 (40GB) GPUs上运行Llama-30B，adapter数量从10到100。
2. 在两个A100 (80GB) 和四个A100 (80GB) GPUs上运行Llama-70B，adapter数量为10。

然后我们报告服务的吞吐量。

如图8所示，S-LoRA在有无LoRA通信情况下的差异很小。这表明我们策略中增加的LoRA通信开销非常小。图中进一步显示，由于LoRA引入的通信开销小于其引入的计算开销。此外，从2个GPU过渡到4个GPU时，服务吞吐量增加超过2倍。这一显著增加可以归因于系统在这种情况下主要受内存限制。增加更多的GPU减轻了内存限制，导致超线性扩展。

**结论.** 结果验证了我们张量并行策略的最小开销和可扩展性。

### 7.5 Ablation Study

**Merging adapter Weights Versus Computing On-the-Fly.** While S-LoRA does not merge adapter weights and computes LoRA matrices on-the-fly each time, we compare it with an alternative design that merges an adapter with the base model, denoted as \( x(W + AB) \), as proposed in the LoRA paper. This approach involves: 
1. Updating the base model with the current adapter weights before each new batch.
2. Switching to a new adapter if there are too many waiting requests.

This method is efficient for a small number of adapters due to the reduced LoRA computation overhead.

Results in Figure 9 demonstrate that with just one adapter, the merging approach outperforms the on-the-fly computation owing to a one-time merging cost. However, its performance declines with more than 2 adapters, primarily because of the time-consuming switch between adapters. Such switching results in periods of GPU under-utilization. Furthermore, a smaller value of \( \alpha \) causes requests to be distributed unevenly across adapters, which in turn reduces batch sizes and overall performance.

**Early Abort Strategy Experiments.** We compared S-LoRA’s early abort strategy to First Come First Serve (FCFS) and Last Come First Serve (LCFS) for optimizing user satisfaction and SLO attainment. As shown in Figure 10, S-LoRA-Abort outperforms both, especially as \( cv \) scales. FCFS is least effective, often processing requests that have already missed the SLO. LCFS, similar to a greedy algorithm that only prioritizes the newest requests, works well for small \( cv \), but its performance drops with larger \( cv \). S-LoRA-Abort excels as it avoids prioritizing only the newest requests, as detailed in Appendix B.

---

### 7.5 消融研究

**合并adapter权重与实时计算.** 尽管S-LoRA每次都不合并adapter权重并实时计算LoRA矩阵，但我们将其与另一种将adapter与基础模型合并的设计进行了比较，该设计在LoRA论文中被表示为 \( x(W + AB) \)。这种方法包括：
1. 在每个新批次之前使用当前的adapter权重更新基础模型。
2. 如果有太多等待请求，则切换到新的adapter。

由于减少了LoRA计算开销，该方法对于少量adapter是有效的。

图9中的结果表明，对于仅一个adapter，合并方法因一次性合并成本而优于实时计算。然而，当adapter数量超过2时，其性能下降，主要是因为adapter之间切换耗时。这种切换导致GPU利用率低。此外，较小的 \( \alpha \) 值会导致请求在adapter之间分布不均，从而减少批次大小并降低整体性能。

**提前中止策略实验.** 我们将S-LoRA的提前中止策略与先来先服务（FCFS）和后到先服务（LCFS）进行了比较，以优化用户满意度和SLO达成率。如图10所示，S-LoRA-Abort在 \( cv \) 扩展时表现优于两者。FCFS最不有效，通常处理已经错过SLO的请求。LCFS类似于仅优先处理最新请求的贪心算法，对于较小的 \( cv \) 表现良好，但在较大的 \( cv \) 时性能下降。S-LoRA-Abort表现出色，因为它避免了仅优先处理最新请求，详见附录B。

### 8 Related Work

**Optimizing LLM Serving with System Techniques.** The prominence of the transformer architecture has led to the development of numerous specialized serving systems. These systems leverage advanced batching mechanisms  , memory optimizations  , GPU kernel optimizations    , model parallelism  , parameter sharing , and speculative execution   to enhance serving efficiency. Among these, PetS  is the most relevant to our work. However, PetS focuses only on serving small encoder-only BERT models and does not address generative inference, handling a large number of adapters, or scaling large models beyond a single GPU, which are central to our settings. In concurrent work, Punica  explored decomposed computation for the base model and adapters. Some of our CUDA kernels were inspired by Punica’s implementation, with additional support for batching different ranks and non-contiguous memory. While kernel performance analysis is not the main focus of this paper, it is discussed in Punica. Our work distinguishes itself with novel memory management and tensor parallelism techniques, which have not been addressed in previous research.

**Optimizing LLM Serving with Algorithm Techniques.** Beyond system-level improvements, inference efficiency can be enhanced through algorithmic techniques such as quantization     , sparsification  , and model architecture improvements . These approaches can reduce memory consumption and speed up computation with minimal compromise in model quality, complementing the techniques presented in this paper.

**Parameter-Efficient Fine-Tuning.** Recent research has developed methods for parameter-efficient fine-tuning of large pre-trained language models. These methods demonstrate that fine-tuning is feasible with only a small fraction of the parameters. State-of-the-art methods include LoRA , Prefix-tuning , P-Tuning , Prompt tuning  , AdaLoRA , and (IA)³ . While our paper focuses on LoRA due to its widespread adoption, most techniques can be easily applied to other parameter-efficient fine-tuning methods as well.

**General Purpose Model Serving Systems.** Over the years, the domain of general model serving has seen significant advancements. Notable systems from earlier research include Clipper , TensorFlow Serving , Nexus , InferLine , and Clockwork . These systems address topics such as batching, caching, and model placement for both single and multiple model deployments. More recent developments like DVABatch , REEF , Shepherd , and AlpaServe  have explored concepts such as multi-entry multi-exit batching, preemption, and statistical multiplexing with model parallelism. Despite their significant contributions, these systems do not address the auto-regressive characteristics and parameter-efficient adapters in LLM serving, leading to potential optimization gaps.

---

### 8 相关工作

**通过系统技术优化LLM服务.** Transformer架构的重要性导致了许多专门服务系统的发展。这些系统利用先进的批处理机制  、内存优化  、GPU内核优化    、模型并行  、参数共享 和推测执行  来提高服务效率。在这些系统中，PetS 与我们的工作最相关。然而，PetS仅考虑了小型编码器仅BERT模型的服务，未涉及生成推理、处理大量adapter或将大模型扩展到单个GPU之外，这些都是我们设置中的问题。在并行工作中，Punica 探索了基于基础模型和adapter的分解计算。我们的一些CUDA内核基于Punica的实现进行了开发，增加了对不同秩和非连续内存的批处理支持。虽然内核性能分析不是本文的重点，但在Punica中有所讨论。我们的工作与Punica不同，我们提出了新的内存管理和张量并行技术，这些技术在之前的研究中没有被涵盖。

**通过算法技术优化LLM服务.** 除了系统级改进外，推理效率还可以通过量化     、稀疏化  和模型架构改进 等算法技术来提高。这些方法可以在模型质量略有妥协的情况下减少内存消耗并加速计算，与本文中提出的技术相辅相成。

**参数高效微调.** 最近的研究开发了大规模预训练语言模型的参数高效微调方法。这些方法表明，只需调整一小部分参数即可进行微调。最先进的方法包括LoRA 、Prefix-tuning 、P-Tuning 、Prompt tuning  、AdaLoRA 和(IA)³ 。尽管本文主要关注LoRA，因为其广泛采用，但大多数技术也可以轻松应用于其他参数高效微调方法。

**通用模型服务系统.** 多年来，通用模型服务领域取得了显著进展。早期研究中的著名系统包括Clipper 、TensorFlow Serving 、Nexus 、InferLine 和Clockwork 。这些系统探讨了批处理、缓存和模型放置等主题，适用于单模型和多模型部署。在最近的进展中，DVABatch 、REEF 、Shepherd 和AlpaServe 探索了多入口多出口批处理、抢占和使用模型并行进行统计复用的概念。尽管这些系统做出了重大贡献，但它们没有考虑LLM服务中的自回归特性和参数高效adapter，从而导致潜在的优化空白。

### 9 Conclusion

We present S-LoRA, a system capable of serving thousands of LoRA adapters from a single machine with much higher throughput compared to existing systems. S-LoRA is made possible by our innovative design of the unified memory pool, tensor parallelism strategy, adapter batching, and CUDA kernels. S-LoRA enables large-scale, customized fine-tuning services essential for deploying models tailored to diverse requirements. Future extensions of S-LoRA will encompass support for additional adapter methods, enhanced fused kernels, and the use of multiple CUDA streams to parallelize base model and LoRA computations.

### Acknowledgment

This research was supported by gifts from Anyscale, Astronomer, Google, IBM, Intel, Lacework, Microsoft, Mohamed Bin Zayed University of Artificial Intelligence, Samsung SDS, Uber, and VMware. Ying is partly supported by the Stanford Center for Automated Reasoning. We thank Clark Barrett for academic advising and funding support. We also thank Yonghao Zhuang and Lisa Dunlap for their helpful discussions and feedback.

---

### 9 结论

我们介绍了S-LoRA，一个能够从单台机器上服务数千个LoRA adapter的系统，与现有系统相比，其吞吐量显著提高。S-LoRA的实现得益于我们创新的统一内存池设计、张量并行策略、adapter批处理和CUDA内核。S-LoRA使得大规模的定制化微调服务成为可能，这对于部署满足多样化需求的模型至关重要。S-LoRA未来的扩展将包括支持更多的adapter方法、增强的融合内核，以及使用多CUDA流并行化基础模型和LoRA计算。

### 致谢

本研究得到了Anyscale, Astronomer, Google, IBM, Intel, Lacework, Microsoft, Mohamed Bin Zayed University of Artificial Intelligence, Samsung SDS, Uber, 和 VMware的资助。Ying部分得到了斯坦福自动推理中心的支持。我们感谢Clark Barrett的学术指导和资金支持，也感谢Yonghao Zhuang和Lisa Dunlap的有益讨论和反馈。