# Awesome Efficient Diffusion/LLM/Visual Autoregressive Model

This is a collection of research papers for **Efficient Diffusion/LLM**, including accelerated sampling, fine-tuning, architecture and so on.


## Selected Work (I like!!)

### Visual Autogressive Model & Diffusion-related Recent work

> In recent times, I've immersed myself in Visual Autoregressive Model, which I'll list over here.

- [Muse: Text-To-Image Generation via Masked Generative Transformers](https://arxiv.org/abs/2301.00704) [ICML 2023] [Mask-Token Prediction]
  - Huiwen Chang, Han Zhang,..., Dilip Krishnan
  - Code: [Community](https://github.com/lucidrains/muse-maskgit-pytorch) [Community](https://github.com/baaivision/muse-pytorch) [Community](https://github.com/huggingface/amused) [Community](https://github.com/Qiyuan-Ge/PaintMind)

- [Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation](https://arxiv.org/abs/2406.06525) [Arxiv 2024] [Next-Token Prediction]
  - Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, Zehuan Yuan
  - Code: [Official](https://github.com/foundationvision/llamagen)

- [MaskGIT: Masked Generative Image Transformer](https://arxiv.org/abs/2202.04200) [CVPR 2022] [Mask-Token Prediction]
  - Huiwen Chang and Han Zhang and Lu Jiang and Ce Liu and William T. Freeman
  - Code: [Official](https://github.com/google-research/maskgit)

- [Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://arxiv.org/pdf/2405.09818) [Arxiv 2024] [Next-Token Prediction] [MLLM]
  - Chameleon Team
  - Code: Waiting...

- [Computational Tradeoffs in Image Synthesis: Diffusion, Masked-Token, and Next-Token Prediction](https://www.arxiv.org/abs/2405.13218) [Arxiv 2024] [Next-Token Prediction] [Mask-Token Prediction]
  - Maciej Kilian, Varun Jampani, Luke Zettlemoyer
  - Code: Waiting...

- [SHOW-O: One Single Transformer to Unify Multimodel Understanding and Generation](https://arxiv.org/pdf/2408.12528) [Arxiv 2024] [Next-Token Prediction] [Mask-Token Prediction] [MLLM]
  - Jinheng Xie, Weijia Mao,..., Mike Zheng Shou
  - Code: [Official-Jax](https://github.com/showlab/show-o)

- [Lumina-mGPT: Illuminate Flexible Photorealistic Text-to-Image Generation with Multimodal Generative Pretraining](https://arxiv.org/abs/2408.02657) [Arxiv 2024] [Next-Token Prediction] [MLLM]
  - Dongyang Liu, Shitian Zhao, Le Zhuo, Weifeng Lin, Yu Qiao, Hongsheng Li, Peng Gao
  - Code: [Official](https://github.com/alpha-vllm/lumina-mgpt)

- [VAR-CLIP: Text-to-Image Generator with Visual Auto-Regressive Modeling](https://arxiv.org/pdf/2408.01181) [Arxiv 2024] [Next-Scale Prediction]
  - Qian Zhang, Xiangzi Dai, Ninghua Yang, Xiang An, Ziyong Feng, Xingyu Ren
  - Code: [Official](https://github.com/daixiangzi/var-clip)

- [OmniTokenizer: A Joint Image-Video Tokenizer for Visual Generation](https://arxiv.org/pdf/2406.09399) [Arxiv 2024] [Next-Token Prediction] [Diffusion] [Tokenizer]
  - Junke Wang, Yi Jiang, Zehuan Yuan, Binyue Peng, Zuxuan Wu, Yu-Gang Jiang
  - Code: [Official](https://github.com/foundationvision/omnitokenizer)

- [Open-MAGVIT2: An Open-Source Project Toward Democratizing Auto-regressive Visual Generation](https://arxiv.org/pdf/2409.04410) [Arxiv 2024] [Next-Token Prediction]
  - Zhuoyan Luo, Fengyuan Shi, Yixiao Ge, Yujiu Yang, Limin Wang, Ying Shan
  - Code: Waiting...

- [Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation](https://arxiv.org/abs/2310.05737) [ICLR 2024] [Next-Token Prediction] [Mask-Token Prediction] [Tokenizer]
  - Lijun Yu, Jose Lezama,..., Lu Jiang
  - Code: [Official](https://github.com/bornfly-detachment/asymmetric_magvitv2)

- [Emu3: Next-Token Prediction is All You Need](https://arxiv.org/abs/2409.18869) [Arxiv 2024] [Next-Token Prediction]
  - Xinlong Wang, Xiaosong Zhang,..., Zhongyuan Wang
  - Code: Waiting...

- [Monoformer: One Transformer for Both Diffusion And Autoregression](https://arxiv.org/pdf/2409.16280) [Arxiv 2024] [Next-Token Prediction] [Diffusion]
  - Chuyang Zhao1, Yuxing Song1,..., Jingdong Wang
  - Code: Waiting...

- [MaskBit: Embedding-free Image Generation via Bit Tokens](https://arxiv.org/pdf/2409.16211) [Arxiv 2024] [Mask-Token Prediction] [Tokenizer]
  - Mark Weber, Lijun Yu, Qihang Yu, Xueqing Deng, Xiaohui Shen, Daniel Cremers, Liang-Chieh Chen
  - Code: Waiting...

- [Scaling Diffusion Transformers to 16 Billion Parameters](https://arxiv.org/pdf/2407.11633) [Arxiv 2024] [Diffusion] [MoE]
  - Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, Junshi Huang
  - Code: [Official](https://github.com/feizc/dit-moe)

- [Efficient Autoregressive Audio Modeling via Next-Scale Prediction](https://arxiv.org/abs/2408.09027) [Arxiv 2024] [Next-Scale Prediction]
  - Kai Qiu, Xiang Li, Hao Chen, Jie Sun, Jinglu Wang, Zhe Lin, Marios Savvides, Bhiksha Raj
  - Code: [Official](https://github.com/qiuk2/aar)

- [StableRep: Synthetic Images from Text-to-Image Models Make Strong Visual Representation Learners](https://proceedings.neurips.cc/paper_files/paper/2023/hash/971f1e59cd956cc094da4e2f78c6ea7c-Abstract-Conference.html) [NeurIPS 2023] [Diffusion]
  - Yonglong Tian, Lijie Fan, Phillip Isola, Huiwen Chang, Dilip Krishnan
  - Code: [Official](https://github.com/google-research/syn-rep-learn)

- [Autoregressive Image Generation without Vector Quantization](https://arxiv.org/abs/2406.11838) [NeurIPS 2024] [Next-Token Prediction]
  - Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, Kaiming He
  - Code: [Official](https://github.com/lth14/mar)

- [AdaNAT: Exploring Adaptive Policy for Token-Based Image Generation](https://arxiv.org/pdf/2409.00342) [ECCV 2024] [Mask-Token Prediction] [Heavy-Inference Algorithm]
  - Zanlin Ni, Yulin Wang, Renping Zhou, Rui Lu, Jiayi Guo, Jinyi Hu, Zhiyuan Liu, Yuan Yao, Gao Huang
  - Code: [Official](https://github.com/leaplabthu/adanat)

- [Discrete Flow Matching](https://arxiv.org/pdf/2407.15595) [Arxiv 2024] [META] [Diffusion in LLM]
  - Meta AI FAIR, Weizmann Institute
  - Code: Waiting...


### Heavy-Inference Algorithm in T2V Synthesis (Inference Scaling Laws)

> This was my research direction, even though the vast majority of the work was REJECTED.

- [FreeNoise: Tuning-Free Longer Video Diffusion via Noise Rescheduling](https://arxiv.org/abs/2310.15169) [ICLR 2024]
  - Haonan Qiu, Menghan Xia, Yong Zhang, Yingqing He, Xintao Wang, Ying Shan, Ziwei Liu
  - Code: [Official](https://github.com/AILab-CVC/FreeNoise)

- [FreeInit : Bridging Initialization Gap in Video Diffusion Models](https://arxiv.org/abs/2312.07537) [ECCV 2024]
  - Tianxing Wu, Chenyang Si, Yuming Jiang, Ziqi Huang, Ziwei Liu
  - Code: [Official](https://github.com/TianxingWu/FreeInit)

- [UniCtrl: Improving the Spatiotemporal Consistency of Text-to-Video Diffusion Models via Training-Free Unified Attention Control](https://arxiv.org/abs/2403.02332) [Arxiv 2024]
  - Xuweiyi Chen, Tian Xia, Sihan Xu
  - Code: [Official](https://github.com/XuweiyiChen/UniCtrl)

- [I4VGen: Image as Free Stepping Stone for Text-to-Video Generation](https://arxiv.org/abs/2406.02230) [Arxiv 2024]
  - Xiefan Guo, Jinlin Liu, Miaomiao Cui, Liefeng Bo, Di Huang
  - Code: [Official](https://github.com/xiefan-guo/i4vgen)

- [ViCo: Plug-and-play Visual Condition for Personalized Text-to-image Generation](https://arxiv.org/abs/2306.00971) [Arxiv 2023]
  - Shaozhe Hao, Kai Han, Shihao Zhao, Kwan-Yee K. Wong
  - Code: [Official](https://github.com/haoosz/ViCo)

- [VideoBooth: Diffusion-based Video Generation with Image Prompts](https://openaccess.thecvf.com/content/CVPR2024/html/Jiang_VideoBooth_Diffusion-based_Video_Generation_with_Image_Prompts_CVPR_2024_paper.html) [CVPR 2024]
  - Yuming Jiang, Tianxing Wu, Shuai Yang, Chenyang Si, Dahua Lin, Yu Qiao, Chen Change Loy, Ziwei Liu
  - Code: [Official](https://github.com/Vchitect/VideoBooth)

- [VideoElevator: Elevating Video Generation Quality with Versatile Text-to-Image Diffusion Models](https://arxiv.org/abs/2403.05438) [Arxiv 2024]
  - Yabo Zhang, Yuxiang Wei, Xianhui Lin, Zheng Hui, Peiran Ren, Xuansong Xie, Xiangyang Ji, Wangmeng Zuo
  - Code: [Official](https://github.com/YBYBZhang/VideoElevator)

- [FreeLong: Training-Free Long Video Generation with SpectralBlend Temporal Attention](https://arxiv.org/abs/2407.19918) [NeurIPS 2024]
  - Yu Lu, Yuanzhi Liang, Linchao Zhu and Yi Yang
  - Code: Waiting

- [GPT4Motion: Scripting Physical Motions in Text-to-Video Generation via Blender-Oriented GPT Planning](https://openaccess.thecvf.com/content/CVPR2024W/PBDL/papers/Lv_GPT4Motion_Scripting_Physical_Motions_in_Text-to-Video_Generation_via_Blender-Oriented_GPT_CVPRW_2024_paper.pdf) [CVPR 2024 Workshop]
  - Jiaxi Lv, Yi Huang,..., Shifeng Chen
  - Code: [Official](https://github.com/jiaxilv/GPT4Motion)

<!-- - [IV-Mixed Sampler: Leveraging Image Diffusion Models for Enhanced Video Synthesis]()
  - Shitong Shao, Zikai Zhou, Lichen Bai, Haoyi Xiong, Zeke Xie
 -->

### Data-Centric Optimization in LLM


## ICML 2024

### Oral

- [Improving Transformers with Dynamically Composable Multi-Head Attention](https://openreview.net/pdf?id=RbiBKPtuHp)
  - Shentao Da Xiao, Qingye Meng, Shengping Li, xingyuan yuan
  - Code: [Official](https://github.com/caiyun-ai/dcformer)
  - Idea: 提出了Dynamically Composable Multi-Head Attention（DCMHA），这是一种参数和计算高效的注意力架构，旨在解决传统Multi-Head Attention（MHA）中存在的缺点，如注意力得分矩阵的低秩瓶颈和头部冗余问题。DCMHA通过动态组合注意力头来增加模型的表达能力。其核心在于一个Compose函数，该函数以输入依赖的方式转换注意力得分和权重矩阵。通过这种方式，DCMHA可以在任何Transformer架构中作为MHA的替代品使用，以获得对应的DCFormer模型。DCFormer在各种不同的架构和模型规模上，在语言建模任务上显著优于原始Transformer，达到了约1.7×–2.0×的计算性能模型的性能。

- [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://openreview.net/pdf?id=hYHsrKDiX7)
  - Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, Yuandong Tian
  - Code: [Official](https://github.com/jiaweizzhao/GaLore)
  - Idea: GaLore，一种通过梯度低秩投影实现大型语言模型（LLM）内存高效训练的方法。GaLore不仅实现了内存高效的预训练，还适用于内存高效的微调。实验表明，GaLore在多个GLUE任务上相较于LoRA等基线方法，以更小的内存占用取得了更好的性能。GaLore通过将梯度投影到低秩子空间（采用SVD分解后优化部分基），减少了内存需求，同时保持了模型训练的准确性。尽管GaLore在LLM训练中展现出优势，但其可能受到数据稀疏性和模型复杂性的影响。未来研究可探索如何进一步优化GaLore以适应更广泛的场景，并探讨其在大规模自然语言处理任务中的潜在应用。【注意，该方法我在diffusion上试过，由于diffusion梯度变化大最终这个范式不行】

- [Compressible Dynamics in Deep Overparameterized Low-Rank Learning & Adaptation](https://openreview.net/pdf?id=uDkXoZMzBv)
  - Can Yaras, Peng Wang, Laura Balzano, Qing Qu
  - Code: [Official](https://github.com/cjyaras/deep-lora-transformers)
  - Idea: 通过利用数据和模型参数内部固有的低维结构以及可压缩的动力学，能够在不增加计算负担的情况下享受过参数化带来的优化和泛化优势。具体来说，论文提出了在深度学习中，尽管过参数化可以带来优化和泛化的好处，但随着模型规模的增大，计算需求也随之增加。然而，通过发现每个权重矩阵的学习动态都被限制在一个不变的低维子空间中，可以构建和训练紧凑、高度压缩的分解形式，这些分解形式具有与过参数化模型相同的优点。对于深度矩阵补全，这种方法显著提高了训练效率，同时保留了过参数化的优势。对于语言模型的微调，论文提出了“Deep LoRA”方法，该方法改进了现有的低秩适应（LoRA）技术，减少了过拟合，简化了超参数设置，同时保持了相当的效率。这种方法在自然语言任务上进行了验证，特别是在有限数据的微调场景下。

- [Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision](https://openreview.net/pdf?id=ghNRg2mEgN)
  - Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, Ilya Sutskever, Jeffrey Wu
  - Code: Waiting
  - Idea: 是提出了一个研究超越人类模型的核心挑战的简单类比，并展示了在这个问题上取得显著进展的可行性。作者首先指出了深度学习中对过参数化（overparameterization）的兴趣日益增加，但随后转向了一个更具挑战性的问题：如何对齐超越人类的模型。为了实现这一目标，文章通过类比和假设，探索了可能的研究途径，并讨论了如何通过现有的系统和方法来应对未来的模型能力可能带来的挑战。尽管作者没有声称拥有完整的答案，但他们概述了一些最佳实践，以最大化在解决对齐超越人类模型这一核心难题上取得真实进展的机会。此外，文章还强调了在研究中明确和枚举关键假设的重要性，以便更好地理解何时研究结果可能失效。其中文章提到的超越人类模型的“对齐”指的是确保超级智能模型（即那些在某些任务上性能超越人类的模型）的行为和决策与人类的价值观、道德准则和预期目标保持一致。
  - Citation: 85 Now

- [LoRA Training in the NTK Regime has No Spurious Local Minima](https://openreview.net/pdf?id=s1sdx6vNsU)
  - Uijeong Jang, Jason D. Lee, Ernest K. Ryu
  - Code: [Official](https://github.com/uijeongjang/lora-ntk)
  - Idea: 这篇论文主要解决的问题是：证明了LoRA（Low-Rank Adaptation）训练方法在无噪声或低噪声设置下不存在虚假的局部最小值，因此（随机）梯度下降算法能够找到低秩的全局最小值。这进一步表明LoRA方法能够找到有效的低秩解，这些解在泛化能力上也表现良好。利用LoRA训练方法，这种方法在训练预训练模型时，只调整部分权重矩阵的低秩分解因子，而不是整个权重矩阵。这种方法在保持模型性能的同时，显著减少了训练参数的数量，提高了训练效率。


- [ExCP: Extreme LLM Checkpoint Compression via Weight-Momentum Joint Shrinking](https://openreview.net/pdf?id=hlvKd7Vdxm)
  - Uijeong Jang, Jason D. Lee, Ernest K. Ryu
  - Code: [Official](https://github.com/gaffey/excp)
  - Idea: 这篇文章主要解决了在机器学习领域，特别是在处理大型预训练模型（如大型语言模型）时，模型检查点的压缩问题。随着模型规模的不断增大，存储和传输这些模型检查点所需的资源也在急剧增加，因此，如何有效地压缩这些检查点变得尤为迫切。核心idea是利用矩阵分解技术来压缩模型检查点。具体来说，作者通过证明存在低秩解来减少模型参数的维度，从而实现模型检查点的压缩。文章通过引入LoRA（Low-Rank Adaptation）技术，展示了如何在保持模型性能的同时，显著减少模型参数的存储需求。

- [PRISE: LLM-Style Sequence Compression for Learning Temporal Action Abstractions in Control](https://openreview.net/pdf?id=p225Od0aYt)
  - Ruijie Zheng, Ching-An Cheng, Hal Daumé III, Furong Huang, Andrey Kolobov
  - Code: [Official](https://github.com/frankzheng2022/prise)
  - Idea: 该问题的背景是连续控制领域的序贯决策问题，特别是模拟机器人操作中的技能学习。作者为了解决在连续控制域中学习具有可变时间跨度的技能的问题，提出了一种新的观点，即将诱导时间动作抽象视为序列压缩问题。为了解决这一问题，作者提出了一个名为Primitive Sequence Encoding（PRISE）的算法。具体来说，作者提出的PRISE算法结合了连续动作量化和字节对编码（Byte Pair Encoding，BPE）来学习强大的动作抽象。首先，使用向量量化模块将连续动作转换为离散代码，然后应用BPE来学习时间扩展的技能令牌。这种方法允许PRISE捕获跨预训练任务的多种运动模式，从而实现了有效的多任务策略学习和对未见问题的少次学习适应。简而言之，该问题的背景是连续控制下的序贯决策技能学习，作者为了解决在连续控制域中学习具有可变时间跨度的技能的问题，提出了PRISE算法。

- [APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference](https://openreview.net/pdf?id=sb81Xl50JG)
  - Bowen Zhao, Hannaneh Hajishirzi, Qingqing Cao
  - Code: [Official](https://github.com/roim1998/apt)
  - Idea: 提出了APT方法，该方法结合了自适应裁剪和自适应调优，以在保持性能的同时提高模型的训练和推理效率。APT确定调优参数位置的方式主要基于以下步骤：计算APT适配器的重要性：APT算法首先计算APT适配器的重要性得分I(Hapt)。这一步骤帮助算法识别哪些适配器对于恢复任务性能更为关键。根据重要性排序调优层：APT算法接下来会根据APT适配器的重要性得分对所有调优层进行排序。这一步的目的是为了识别哪些调优层对于模型性能的影响更大。动态增加显著调优层的秩：在确定了显著调优层之后，APT算法会按照预算∆t逐步增加这些调优层的秩rapt。这通过增加动态秩在APT适配器中来实现，从而添加调优参数。更具体地说，当从∆t增加到∆t′时，显著层的秩rapt会按照apt = ⌊rapt · ∆t′/∆t⌋进行调整，其中⌊·⌋表示向下取整操作。保持训练稳定性：在添加参数时，APT算法会采用与LoRA初始化类似的方法，通过在WA中添加随机初始化的高斯参数N(0, σ2)，同时在WB中添加零值，以确保在添加新参数前后层的输出保持不变，从而保持训练的稳定性。

- [Fast Timing-Conditioned Latent Audio Diffusion](https://openreview.net/pdf?id=jOlO8t1xdx)
  - Zach Evans, CJ Carr, Josiah Taylor, Scott H. Hawley, Jordi Pons
  - Code: [Official](https://github.com/stability-ai/stable-audio-metrics)
  - Idea: 本文介绍了Stable Audio，一种基于文本提示和时序嵌入的潜变量音频扩散模型，旨在高效生成长达95秒的44.1kHz立体声音乐和声音。Stable Audio采用全卷积变分自编码器定义潜变量，通过文本和时序条件控制生成内容和长度。模型在GPU上能够在8秒内渲染长达95秒的音频，并在两个公开基准测试中表现优异。文章提到Moûsai的模型基于谱图编码器和需要100步解码的扩散解码器，而作者的方法则采用了全卷积的端到端VAE。这种架构对于实现快速推理时间至关重要。Stable Audio被设计为在指定窗口长度（如95秒）内生成内容，这允许它生成可变长度的长形式音乐和声音。

- [Any-Precision LLM: Low-Cost Deployment of Multiple, Different-Sized LLMs](https://openreview.net/pdf?id=u09gadH3BU)
  - Yeonhong Park, Jake Hyun, SangLyul Cho, Bonggeun Sim, Jae W. Lee
  - Code: [Official](https://github.com/SNU-ARC/any-precision-llm)
  - Idea: 本文探讨了低成本部署多种不同大小的大规模语言模型（LLMs）的方法，即Any-Precision LLM。研究分析了通过量化技术降低LLM部署成本的有效性，并比较了不同精度下的模型性能。权重位平面布局优化（Weight Bitplane Layout Optimization, WLO）：这种技术优化了缓存访问模式，使得在较低位宽下，缓存的利用更加高效。改进的位转置算法（Improved Bit-Transpose Algorithm, IBT）：相对于现有的位转置算法（如Warren, 2012年提出的算法），改进后的算法在计算效率上有了显著提升。表查找合并（Table Lookup Merging, TLM）：对于3位的情况，通过表查找合并技术，进一步提高了计算效率。

- [Accurate LoRA-Finetuning Quantization of LLMs via Information Retention](https://openreview.net/pdf?id=jQ92egz5Ym)
  - Haotong Qin, Xudong Ma, Xingyu Zheng, Xiaoyang Li, Yang Zhang, Shouda Liu, Jie Luo, Xianglong Liu, Michele Magno
  - Code: [Official](https://openreview.net/pdf?id=jQ92egz5Ym)
  - Idea: 本文提出了IR-QLoRA方法，旨在通过信息保留的方式对大型语言模型（LLMs）进行精确的LoRA微调量化。该方法结合了信息校准量化（ICQ）和信息弹性连接（IEC），以在量化过程中保持模型的性能。（1）IR-QLoRA方法通过信息保留技术实现了对大型语言模型LoRA微调过程的精确量化，旨在减少模型大小同时保持性能。（2）IR-QLoRA包括信息校准量化（ICQ）用于量化LLMs，以及信息弹性连接（IEC）用于增强LoRA微调过程中的信息流动。

- [DiJiang: Efficient Large Language Models through Compact Kernelization](https://openreview.net/pdf?id=0uUHfhXdnH)
  - Hanting Chen, Liuzhicheng, Xutao Wang, Yuchuan Tian, Yunhe Wang
  - Code: [Official](https://openreview.net/pdf?id=jQ92egz5Ym)
  - Idea: 本文探讨了通过紧凑核化（Compact Kernelization）提高大型语言模型（Large Language Models, LLMs）的效率。针对Transformer架构在自然语言处理（NLP）领域的广泛应用，文章分析了现有LLMs在计算注意力机制时存在的效率问题，并提出了一种新的基于核方法的注意力近似计算方法，以改善模型的性能。通过引入一种新颖的Frequency Domain Kernelization方法，DiJiang能够基于预训练的Transformer模型，将其转化为具有线性复杂度的模型，而几乎不需要额外的训练成本。这解决了传统Transformer模型在适应线性注意力机制时通常需要大量重新训练的问题，特别适用于大型语言模型，其中训练和重新训练的成本和时间都是巨大的障碍。

- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://openreview.net/pdf?id=3d5CIRG1n2)
  - Shih-yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Min-Hung Chen
  - Code: [Official](https://github.com/NVlabs/DoRA)
  - Idea: 本论文提出了一种名为DoRA（Decomposed Rank Adaptation）的方法，该方法旨在通过分解权重矩阵的低秩适应来改进深度学习模型的适应性和泛化能力。论文详细描述了DoRA的实现细节，并通过实验验证了其在多个基准数据集上的有效性。DoRA 将预训练的权重分解为两个部分，即幅度和方向，以便进行微调，特别是利用 LoRA 进行方向更新，从而有效地减少可训练参数的数量。通过使用 DoRA，我们增强了 LoRA 的学习能力和训练稳定性，同时避免了任何额外的推理开销。在各种下游任务（如常识推理、视觉指令调整和图像/视频文本理解）中，DoRA 在微调 LLaMA、LLaVA 和 VL-BART 方面的表现始终优于 LoRA。

### Spotlight


### Poster

- [COLLAGE: Light-Weight Low-Precision Strategy for LLM Training](https://arxiv.org/pdf/2405.03637)
- Tao Yu, Gaurav Gupta, Karthick Gopalswamy, Amith Mamidala, Hao Zhou, Jeffrey Huynh, Youngsuk Park, Ron Diamant, Anoop Deoras, Luke Huan
- Code: Waiting
- Idea: 该论文引入了一种已经存在的低比特张良组织形式，名为MCF，它采用这个替代传统的float32进行优化器的参数更新。具体来说，它其实没有改变任何的计算逻辑，也没有做出一些算法上的创新，更多体现在工程上。实现优化其中二阶动量更新和梯度更新采用MCF来计算，并给予实现，大规模降低了计算存储和消耗。


## NeurIPS 2024