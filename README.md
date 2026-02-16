# Learned Optimizers Literature

Categorized list of 35 papers on learned optimizers.

Use `python manage_papers.py` to update abstracts and links.

## 2025

### [Celo: Training Versatile Learned Optimizers on a Compute Diet](https://www.semanticscholar.org/paper/3fd5e6f90f34109f8737cb307d29d08a3638bd24)
**Authors:** Moudgil, Abhinav and Knyazev, Boris and Lajoie, Guillaume and Belilovsky, Eugene

<details>
<summary>Abstract</summary>

> Learned optimization has emerged as a promising alternative to hand-crafted optimizers, with the potential to discover stronger learned update rules that enable faster, hyperparameter-free training of neural networks. A critical element for practically useful learned optimizers, that can be used off-the-shelf after meta-training, is strong meta-generalization: the ability to apply the optimizers to new tasks. Recent state-of-the-art work in learned optimizers, VeLO (Metz et al., 2022), requires a large number of highly diverse meta-training tasks along with massive computational resources, 4000 TPU months, to achieve meta-generalization. This makes further improvements to such learned optimizers impractical. In this work, we identify several key elements in learned optimizer architectures and meta-training procedures that can lead to strong meta-generalization. We also propose evaluation metrics to reliably assess quantitative performance of an optimizer at scale on a set of evaluation tasks. Our proposed approach, Celo, makes a significant leap in improving the meta-generalization performance of learned optimizers and also outperforms tuned state-of-the-art optimizers on a diverse set of out-of-distribution tasks, despite being meta-trained for just 24 GPU hours.
</details>

---

### [PyLO: Towards Accessible Learned Optimizers in PyTorch](https://www.semanticscholar.org/paper/9e537b053ee3b3b8fa3ada451944ea611c347efe)
**Authors:** Janson, Paul and Therien, Benjamin and Anthony, Quentin and Huang, Xiaolong and Moudgil, Abhinav and Belilovsky, Eugene

<details>
<summary>Abstract</summary>

> Learned optimizers have been an active research topic over the past decade, with increasing progress toward practical, general-purpose optimizers that can serve as drop-in replacements for widely used methods like Adam. However, recent advances -- such as VeLO, which was meta-trained for 4000 TPU-months -- remain largely inaccessible to the broader community, in part due to their reliance on JAX and the absence of user-friendly packages for applying the optimizers after meta-training. To address this gap, we introduce PyLO, a PyTorch-based library that brings learned optimizers to the broader machine learning community through familiar, widely adopted workflows. Unlike prior work focused on synthetic or convex tasks, our emphasis is on applying learned optimization to real-world large-scale pre-training tasks. Our release includes a CUDA-accelerated version of the small_fc_lopt learned optimizer architecture from (Metz et al., 2022a), delivering substantial speedups -- from 39.36 to 205.59 samples/sec throughput for training ViT B/16 with batch size 32. PyLO also allows us to easily combine learned optimizers with existing optimization tools such as learning rate schedules and weight decay. When doing so, we find that learned optimizers can substantially benefit. Our code is available at https://github.com/Belilovsky-Lab/pylo
</details>

---

## 2024

### [μLO: Compute-Efficient Meta-Generalization of Learned Optimizers](https://www.semanticscholar.org/paper/7cb61bf31a57366343b323f554e71be35ee83b91)
**Authors:** Thérien, Benjamin and Joseph, C. and Knyazev, B. A. and Oyallon, Edouard and Rish, Irina and Belilovsky, Eugene

<details>
<summary>Abstract</summary>

> Learned optimizers (LOs) have the potential to significantly reduce the wall-clock training time of neural networks. However, they can struggle to optimize unseen tasks (\emph{meta-generalize}), especially when training networks wider than those seen during meta-training. To address this, we derive the Maximal Update Parametrization ($\mu$P) for two state-of-the-art learned optimizer architectures and propose a simple meta-training recipe for $\mu$-parameterized LOs ($\mu$LOs). Our empirical evaluation demonstrates that LOs meta-trained with our recipe substantially improve meta-generalization to wider unseen tasks when compared to LOs trained under standard parametrization (SP) using the same compute budget. We also empirically observe that $\mu$LOs exhibit unexpectedly improved meta-generalization to deeper networks ($5\times$ meta-training) and surprising generalization to much longer training horizons ($25\times$ meta-training) when compared to SP LOs.
</details>

---

### [MADA: Meta-Adaptive Optimizers through hyper-gradient Descent](https://www.semanticscholar.org/paper/bef33d15c3e8d433261f97f7001cc41a5ae0ec32)
**Authors:** Ozkara, Kaan and Karakus, Can and Raman, Parameswaran and Hong, Mingyi and Sabach, Shoham and Kveton, Branislav and Cevher, Volkan

<details>
<summary>Abstract</summary>

> Following the introduction of Adam, several novel adaptive optimizers for deep learning have been proposed. These optimizers typically excel in some tasks but may not outperform Adam uniformly across all tasks. In this work, we introduce Meta-Adaptive Optimizers (MADA), a unified optimizer framework that can generalize several known optimizers and dynamically learn the most suitable one during training. The key idea in MADA is to parameterize the space of optimizers and dynamically search through it using hyper-gradient descent during training. We empirically compare MADA to other popular optimizers on vision and language tasks, and find that MADA consistently outperforms Adam and other popular optimizers, and is robust against sub-optimally tuned hyper-parameters. MADA achieves a greater validation performance improvement over Adam compared to other popular optimizers during GPT-2 training and fine-tuning. We also propose AVGrad, a modification of AMSGrad that replaces the maximum operator with averaging, which is more suitable for hyper-gradient optimization. Finally, we provide a convergence analysis to show that parameterized interpolations of optimizers can improve their error bounds (up to constants), hinting at an advantage for meta-optimizers.
</details>

---

### [Graph Neural Networks for Learning Equivariant Representations of Neural Networks](https://www.semanticscholar.org/paper/fc580c211689663a64f42e2ba92c864cb134ba9b)
**Authors:** Kofinas, Miltiadis and Knyazev, Boris and Zhang, Yan and Chen, Yunlu and Burghouts, Gertjan J. and Gavves, Efstratios and Snoek, Cees G. M. and Zhang, David

<details>
<summary>Abstract</summary>

> Neural networks that process the parameters of other neural networks find applications in domains as diverse as classifying implicit neural representations, generating neural network weights, and predicting generalization errors. However, existing approaches either overlook the inherent permutation symmetry in the neural network or rely on intricate weight-sharing patterns to achieve equivariance, while ignoring the impact of the network architecture itself. In this work, we propose to represent neural networks as computational graphs of parameters, which allows us to harness powerful graph neural networks and transformers that preserve permutation symmetry. Consequently, our approach enables a single model to encode neural computational graphs with diverse architectures. We showcase the effectiveness of our method on a wide range of tasks, including classification and editing of implicit neural representations, predicting generalization performance, and learning to optimize, while consistently outperforming state-of-the-art methods. The source code is open-sourced at https://github.com/mkofinas/neural-graphs.
</details>

---

### [Scale Equivariant Graph Metanetworks](https://www.semanticscholar.org/paper/d584110aad0ba7492823d041b18af4ca77239c95)
**Authors:** Kalogeropoulos, Ioannis and Bouritsas, Giorgos and Panagakis, Yannis

<details>
<summary>Abstract</summary>

> This paper pertains to an emerging machine learning paradigm: learning higher-order functions, i.e. functions whose inputs are functions themselves, $\textit{particularly when these inputs are Neural Networks (NNs)}$. With the growing interest in architectures that process NNs, a recurring design principle has permeated the field: adhering to the permutation symmetries arising from the connectionist structure of NNs. $\textit{However, are these the sole symmetries present in NN parameterizations}$? Zooming into most practical activation functions (e.g. sine, ReLU, tanh) answers this question negatively and gives rise to intriguing new symmetries, which we collectively refer to as $\textit{scaling symmetries}$, that is, non-zero scalar multiplications and divisions of weights and biases. In this work, we propose $\textit{Scale Equivariant Graph MetaNetworks - ScaleGMNs}$, a framework that adapts the Graph Metanetwork (message-passing) paradigm by incorporating scaling symmetries and thus rendering neuron and edge representations equivariant to valid scalings. We introduce novel building blocks, of independent technical interest, that allow for equivariance or invariance with respect to individual scalar multipliers or their product and use them in all components of ScaleGMN. Furthermore, we prove that, under certain expressivity conditions, ScaleGMN can simulate the forward and backward pass of any input feedforward neural network. Experimental results demonstrate that our method advances the state-of-the-art performance for several datasets and activation functions, highlighting the power of scaling symmetries as an inductive bias for NN processing. The source code is publicly available at https://github.com/jkalogero/scalegmn.
</details>

---

### [Can Learned Optimization Make Reinforcement Learning Less Difficult?](https://www.semanticscholar.org/paper/5ecf53ab083f72f10421225a7dde25eb51cb6b22)
**Authors:** Goldie, Alexander David and Lu, Chris Xiaoxuan and Jackson, Matthew Thomas and Whiteson, Shimon and Foerster, Jakob

<details>
<summary>Abstract</summary>

> While reinforcement learning (RL) holds great potential for decision making in the real world, it suffers from a number of unique difficulties which often need specific consideration. In particular: it is highly non-stationary; suffers from high degrees of plasticity loss; and requires exploration to prevent premature convergence to local optima and maximize return. In this paper, we consider whether learned optimization can help overcome these problems. Our method, Learned Optimization for Plasticity, Exploration and Non-stationarity (OPEN), meta-learns an update rule whose input features and output structure are informed by previously proposed solutions to these difficulties. We show that our parameterization is flexible enough to enable meta-learning in diverse learning contexts, including the ability to use stochasticity for exploration. Our experiments demonstrate that when meta-trained on single and small sets of environments, OPEN outperforms or equals traditionally used optimizers. Furthermore, OPEN shows strong generalization characteristics across a range of environments and agent architectures.
</details>

---

## 2023

### [Graph Metanetworks for Processing Diverse Neural Architectures](https://www.semanticscholar.org/paper/af8df99efea4d4ed6f5cf6f6eaf3a5943f4d75db)
**Authors:** Lim, Derek and Maron, Haggai and Law, Marc T. and Lorraine, Jonathan and Lucas, James M.

<details>
<summary>Abstract</summary>

> Neural networks efficiently encode learned information within their parameters. Consequently, many tasks can be unified by treating neural networks themselves as input data. When doing so, recent studies demonstrated the importance of accounting for the symmetries and geometry of parameter spaces. However, those works developed architectures tailored to specific networks such as MLPs and CNNs without normalization layers, and generalizing such architectures to other types of networks can be challenging. In this work, we overcome these challenges by building new metanetworks - neural networks that take weights from other neural networks as input. Put simply, we carefully build graphs representing the input neural networks and process the graphs using graph neural networks. Our approach, Graph Metanetworks (GMNs), generalizes to neural architectures where competing methods struggle, such as multi-head attention layers, normalization layers, convolutional layers, ResNet blocks, and group-equivariant linear layers. We prove that GMNs are expressive and equivariant to parameter permutation symmetries that leave the input neural network functions unchanged. We validate the effectiveness of our method on several metanetwork tasks over diverse neural network architectures.
</details>

---

### [Meta-learning Optimizers for Communication-Efficient Learning](https://www.semanticscholar.org/paper/4ff9ae519bd9084af922479955b53674afcdb3fb)
**Authors:** Charles-'Etienne Joseph

<details>
<summary>Abstract</summary>

> Communication-efficient variants of SGD, specifically local SGD, have received a great deal of interest in recent years. These approaches compute multiple gradient steps locally on each worker, before averaging model parameters, helping relieve the critical communication bottleneck in distributed deep learning training. Although many variants of these approaches have been proposed, they can sometimes lag behind state-of-the-art adaptive optimizers for deep learning. In this work, we investigate if the recent progress in the emerging area of learned optimizers can potentially close this gap in homogeneous data and homogeneous device settings while remaining communication-efficient. Specifically, we meta-learn how to perform global updates given an update from local SGD iterations. Our results demonstrate that learned optimizers can substantially outperform local SGD and its sophisticated variants while maintaining their communication efficiency. Our learned optimizers can even generalize to unseen and much larger datasets and architectures, including ImageNet and ViTs, and to unseen modalities such as language modeling. We therefore show the potential of learned optimizers for improving communication-efficient distributed learning.
</details>

---

### [Improving physics-informed neural networks with meta-learned optimization](https://www.semanticscholar.org/paper/4f9f96679e943f7447ac431dffea506767d3cf3f)
**Authors:** Bihlo, Alex

<details>
<summary>Abstract</summary>

> We show that the error achievable using physics-informed neural networks for solving systems of differential equations can be substantially reduced when these networks are trained using meta-learned optimization methods rather than to using fixed, hand-crafted optimizers as traditionally done. We choose a learnable optimization method based on a shallow multi-layer perceptron that is meta-trained for specific classes of differential equations. We illustrate meta-trained optimizers for several equations of practical relevance in mathematical physics, including the linear advection equation, Poisson's equation, the Korteweg--de Vries equation and Burgers' equation. We also illustrate that meta-learned optimizers exhibit transfer learning abilities, in that a meta-trained optimizer on one differential equation can also be successfully deployed on another differential equation.
</details>

---

### [Is Scaling Learned Optimizers Worth It? Evaluating The Value of VeLO's 4000 TPU Months](https://www.semanticscholar.org/paper/d92fe2b8861e8ef74e030d674bedcd505a98def9)
**Authors:** Antoniou, Antreas and Gouk, Henry and Hospedales, Timothy M.

<details>
<summary>Abstract</summary>

> We analyze VeLO (versatile learned optimizer), the largest scale attempt to train a general purpose"foundational"optimizer to date. VeLO was trained on thousands of machine learning tasks using over 4000 TPU months with the goal of producing an optimizer capable of generalizing to new problems while being hyperparameter free, and outperforming industry standards such as Adam. We independently evaluate VeLO on the MLCommons optimizer benchmark suite. We find that, contrary to initial claims: (1) VeLO has a critical hyperparameter that needs problem-specific tuning, (2) VeLO does not necessarily outperform competitors in quality of solution found, and (3) VeLO is not faster than competing optimizers at reducing the training loss. These observations call into question VeLO's generality and the value of the investment in training it.
</details>

---

## 2022

### [Learned Learning Rate Schedules for Deep Neural Network Training Using Reinforcement Learning](https://www.semanticscholar.org/paper/e8b22993efec83105daae948878a2c2ec0358c85)
**Authors:** Subramanian, Shreyas Vathul and Ganapathiraman, Vignesh and Gamal, Aly El

*No abstract available.*

---

### [VeLO: Training Versatile Learned Optimizers by Scaling Up](https://www.semanticscholar.org/paper/c088b46519c036149a9f6da4ec36383b800a0d2a)
**Authors:** Metz, Luke and Harrison, J. and Freeman, C. Daniel and Merchant, Amil and Beyer, Lucas and Bradbury, James T. and Agrawal, Naman and Poole, Ben and Mordatch, Igor and Roberts, Adam and Sohl‐Dickstein, Jascha

<details>
<summary>Abstract</summary>

> While deep learning models have replaced hand-designed features across many domains, these models are still trained with hand-designed optimizers. In this work, we leverage the same scaling approach behind the success of deep learning to learn versatile optimizers. We train an optimizer for deep learning which is itself a small neural network that ingests gradients and outputs parameter updates. Meta-trained with approximately four thousand TPU-months of compute on a wide variety of optimization tasks, our optimizer not only exhibits compelling performance, but optimizes in interesting and unexpected ways. It requires no hyperparameter tuning, instead automatically adapting to the specifics of the problem being optimized. We open source our learned optimizer, meta-training code, the associated train and test data, and an extensive optimizer benchmark suite with baselines at velo-code.github.io.
</details>

---

### [A Closer Look at Learned Optimization: Stability, Robustness, and Inductive Biases](https://www.semanticscholar.org/paper/2d59b386a6037a895edf72c4420b76f64d921ee4)
**Authors:** Harrison, J. and Metz, Luke and Sohl‐Dickstein, Jascha

<details>
<summary>Abstract</summary>

> Learned optimizers -- neural networks that are trained to act as optimizers -- have the potential to dramatically accelerate training of machine learning models. However, even when meta-trained across thousands of tasks at huge computational expense, blackbox learned optimizers often struggle with stability and generalization when applied to tasks unlike those in their meta-training set. In this paper, we use tools from dynamical systems to investigate the inductive biases and stability properties of optimization algorithms, and apply the resulting insights to designing inductive biases for blackbox optimizers. Our investigation begins with a noisy quadratic model, where we characterize conditions in which optimization is stable, in terms of eigenvalues of the training dynamics. We then introduce simple modifications to a learned optimizer's architecture and meta-training procedure which lead to improved stability, and improve the optimizer's inductive bias. We apply the resulting learned optimizer to a variety of neural network training tasks, where it outperforms the current state of the art learned optimizer -- at matched optimizer computational overhead -- with regard to optimization performance and meta-training speed, and is capable of generalization to tasks far different from those it was meta-trained on.
</details>

---

### [Tutorial on amortized optimization for learning to optimize over continuous domains](https://www.semanticscholar.org/paper/8198ec90c9e97f583f921451e7112d943da75cb4)
**Authors:** Amos, Brandon

*No abstract available.*

---

## 2021

### [Learning to Optimize: A Primer and A Benchmark.](https://www.semanticscholar.org/paper/f401a919db41eca28aa1cff062d98cc03b7ab66b)
**Authors:** Chen, Tianlong and Chen, Xiaohan and Chen, Wuyang and Heaton, Howard and Liu, Jialin and Wang, Zhangyang and Yin, Wotao

<details>
<summary>Abstract</summary>

> Learning to optimize (L2O) is an emerging approach that leverages machine learning to develop optimization methods, aiming at reducing the laborious iterations of hand engineering. It automates the design of an optimization method based on its performance on a set of training problems. This data-driven procedure generates methods that can efficiently solve problems similar to those in the training. In sharp contrast, the typical and traditional designs of optimization methods are theory-driven, so they obtain performance guarantees over the classes of problems specified by the theory. The difference makes L2O suitable for repeatedly solving a certain type of optimization problems over a specific distribution of data, while it typically fails on out-of-distribution problems. The practicality of L2O depends on the type of target optimization, the chosen architecture of the method to learn, and the training procedure. This new paradigm has motivated a community of researchers to explore L2O and report their findings. This article is poised to be the first comprehensive survey and benchmark of L2O for continuous optimization. We set up taxonomies, categorize existing works and research directions, present insights, and identify open challenges. We also benchmarked many existing L2O approaches on a few but representative optimization problems. For reproducible research and fair benchmarking purposes, we released our software implementation and data in the package Open-L2O at https://github.com/VITA-Group/Open-L2O.
</details>

---

## 2020

### [Tasks, stability, architecture, and compute: Training more effective learned optimizers, and using them to train themselves](https://www.semanticscholar.org/paper/8a858fb857abc06817d245bcb774a3901676f144)
**Authors:** Metz, Luke and Maheswaranathan, Niru and Freeman, Chris and Poole, Benjamin and Sohl-Dickstein, Jascha Narain

<details>
<summary>Abstract</summary>

> Much as replacing hand-designed features with learned functions has revolutionized how we solve perceptual tasks, we believe learned algorithms will transform how we train models. In this work we focus on general-purpose learned optimizers capable of training a wide variety of problems with no user-specified hyperparameters. We introduce a new, neural network parameterized, hierarchical optimizer with access to additional features such as validation loss to enable automatic regularization. Most learned optimizers have been trained on only a single task, or a small number of tasks. We train our optimizers on thousands of tasks, making use of orders of magnitude more compute, resulting in optimizers that generalize better to unseen tasks. The learned optimizers not only perform well, but learn behaviors that are distinct from existing first order optimizers. For instance, they generate update steps that have implicit regularization and adapt as the problem hyperparameters (e.g. batch size) or architecture (e.g. neural network width) change. Finally, these learned optimizers show evidence of being useful for out of distribution tasks such as training themselves from scratch.
</details>

---

### [Tasks, stability, architecture, and compute: Training more effective learned optimizers, and using them to train themselves.](https://www.semanticscholar.org/paper/8a858fb857abc06817d245bcb774a3901676f144)
**Authors:** Metz, Luke and Maheswaranathan, Niru and Freeman, C. Daniel and Poole, Ben and Sohl‐Dickstein, Jascha

<details>
<summary>Abstract</summary>

> Much as replacing hand-designed features with learned functions has revolutionized how we solve perceptual tasks, we believe learned algorithms will transform how we train models. In this work we focus on general-purpose learned optimizers capable of training a wide variety of problems with no user-specified hyperparameters. We introduce a new, neural network parameterized, hierarchical optimizer with access to additional features such as validation loss to enable automatic regularization. Most learned optimizers have been trained on only a single task, or a small number of tasks. We train our optimizers on thousands of tasks, making use of orders of magnitude more compute, resulting in optimizers that generalize better to unseen tasks. The learned optimizers not only perform well, but learn behaviors that are distinct from existing first order optimizers. For instance, they generate update steps that have implicit regularization and adapt as the problem hyperparameters (e.g. batch size) or architecture (e.g. neural network width) change. Finally, these learned optimizers show evidence of being useful for out of distribution tasks such as training themselves from scratch.
</details>

---

### [Training Stronger Baselines for Learning to Optimize](https://www.semanticscholar.org/paper/3b5bd23386ded50640d285bc95d748272b23ca5c)
**Authors:** Chen, Tianlong and Zhang, Weiyi and Zhou, Jingyang and Chang, Shiyu and Liu, Sijia and Amini, Lisa and Yan, Shuicheng and Wang, Zhangyang

<details>
<summary>Abstract</summary>

> Learning to optimize (L2O) has gained increasing attention since classical optimizers require laborious problem-specific design and hyperparameter tuning. However, there is a gap between the practical demand and the achievable performance of existing L2O models. Specifically, those learned optimizers are applicable to only a limited class of problems, and often exhibit instability. With many efforts devoted to designing more sophisticated L2O models, we argue for another orthogonal, under-explored theme: the training techniques for those L2O models. We show that even the simplest L2O model could have been trained much better. We first present a progressive training scheme to gradually increase the optimizer unroll length, to mitigate a well-known L2O dilemma of truncation bias (shorter unrolling) versus gradient explosion (longer unrolling). We further leverage off-policy imitation learning to guide the L2O learning, by taking reference to the behavior of analytical optimizers. Our improved training techniques are plugged into a variety of state-of-the-art L2O models, and immediately boost their performance, without making any change to their model structures. Especially, by our proposed techniques, an earliest and simplest L2O model can be trained to outperform the latest complicated L2O models on a number of tasks. Our results demonstrate a greater potential of L2O yet to be unleashed, and urge to rethink the recent progress. Our codes are publicly available at: this https URL.
</details>

---

## 2019

### [Understanding and correcting pathologies in the training of learned optimizers](https://www.semanticscholar.org/paper/cb4147fbd0704398c692667078efff935a36bb6d)
**Authors:** Metz, Luke and Maheswaranathan, Niru and Nixon, Jeremy and Freeman, C. Daniel and Sohl‐Dickstein, Jascha

<details>
<summary>Abstract</summary>

> Deep learning has shown that learned functions can dramatically outperform hand-designed functions on perceptual tasks. Analogously, this suggests that learned optimizers may similarly outperform current hand-designed optimizers, especially for specific problems. However, learned optimizers are notoriously difficult to train and have yet to demonstrate wall-clock speedups over hand-designed optimizers, and thus are rarely used in practice. Typically, learned optimizers are trained by truncated backpropagation through an unrolled optimization process resulting in gradients that are either strongly biased (for short truncations) or have exploding norm (for long truncations). In this work we propose a training scheme which overcomes both of these difficulties, by dynamically weighting two unbiased gradient estimators for a variational loss on optimizer performance, allowing us to train neural networks to perform optimization of a specific task faster than tuned first-order methods. We demonstrate these results on problems where our learned optimizer trains convolutional networks faster in wall-clock time compared to tuned first-order methods and with an improvement in test loss.
</details>

---

## 2018

### [Understanding and correcting pathologies in the training of learned optimizers](https://www.semanticscholar.org/paper/cb4147fbd0704398c692667078efff935a36bb6d)
**Authors:** Metz, Luke and Maheswaranathan, Niru and Nixon, Jeremy and Freeman, Chris and Sohl-Dickstein, Jascha Narain

<details>
<summary>Abstract</summary>

> Deep learning has shown that learned functions can dramatically outperform hand-designed functions on perceptual tasks. Analogously, this suggests that learned optimizers may similarly outperform current hand-designed optimizers, especially for specific problems. However, learned optimizers are notoriously difficult to train and have yet to demonstrate wall-clock speedups over hand-designed optimizers, and thus are rarely used in practice. Typically, learned optimizers are trained by truncated backpropagation through an unrolled optimization process resulting in gradients that are either strongly biased (for short truncations) or have exploding norm (for long truncations). In this work we propose a training scheme which overcomes both of these difficulties, by dynamically weighting two unbiased gradient estimators for a variational loss on optimizer performance, allowing us to train neural networks to perform optimization of a specific task faster than tuned first-order methods. We demonstrate these results on problems where our learned optimizer trains convolutional networks faster in wall-clock time compared to tuned first-order methods and with an improvement in test loss.
</details>

---

## 2017

### [Learned optimizers that scale and generalize](https://www.semanticscholar.org/paper/b8ff7e02ffa1577d125acd3e998e8ce76a9059dc)
**Authors:** Wichrowska, Olga and Maheswaranathan, Niru and Hoffman, Matthew W. and Colmenarejo, Sergio Gómez and Denil, Misha and Freitas, Nando de and Sohl‐Dickstein, Jascha

<details>
<summary>Abstract</summary>

> Learning to learn has emerged as an important direction for achieving artificial intelligence. Two of the primary barriers to its adoption are an inability to scale to larger problems and a limited ability to generalize to new tasks. We introduce a learned gradient descent optimizer that generalizes well to new tasks, and which has significantly reduced memory and computation overhead. We achieve this by introducing a novel hierarchical RNN architecture, with minimal per-parameter overhead, augmented with additional architectural features that mirror the known structure of optimization tasks. We also develop a meta-training ensemble of small, diverse, optimization tasks capturing common properties of loss landscapes. The optimizer learns to outperform RMSProp/ADAM on problems in this corpus. More importantly, it performs comparably or better when applied to small convolutional neural networks, despite seeing no neural networks in its meta-training set. Finally, it generalizes to train Inception V3 and ResNet V2 architectures on the ImageNet dataset for thousands of steps, optimization problems that are of a vastly different scale than those it was trained on.
</details>

---

### [Learned Optimizers that Scale and Generalize](https://www.semanticscholar.org/paper/b8ff7e02ffa1577d125acd3e998e8ce76a9059dc)
**Authors:** Wichrowska, Olga N and Maheswaranathan, Niru and Hoffman, Matt and Colmenarejo, Sergio Gómez and Denil, Misha and Freitas, Nando de and Sohl-Dickstein, Jascha Narain

<details>
<summary>Abstract</summary>

> Learning to learn has emerged as an important direction for achieving artificial intelligence. Two of the primary barriers to its adoption are an inability to scale to larger problems and a limited ability to generalize to new tasks. We introduce a learned gradient descent optimizer that generalizes well to new tasks, and which has significantly reduced memory and computation overhead. We achieve this by introducing a novel hierarchical RNN architecture, with minimal per-parameter overhead, augmented with additional architectural features that mirror the known structure of optimization tasks. We also develop a meta-training ensemble of small, diverse, optimization tasks capturing common properties of loss landscapes. The optimizer learns to outperform RMSProp/ADAM on problems in this corpus. More importantly, it performs comparably or better when applied to small convolutional neural networks, despite seeing no neural networks in its meta-training set. Finally, it generalizes to train Inception V3 and ResNet V2 architectures on the ImageNet dataset for thousands of steps, optimization problems that are of a vastly different scale than those it was trained on.
</details>

---

### [Optimization as a Model for Few-Shot Learning](https://www.semanticscholar.org/paper/29c887794eed2ca9462638ff853e6fe1ab91d5d8)
**Authors:** Ravi, Sachin and Larochelle, Hugo

*No abstract available.*

---

### [Learning Gradient Descent: Better Generalization and Longer Horizons](https://www.semanticscholar.org/paper/822e7515152e74626265c26b7aaccd2c654b5eba)
**Authors:** Lv, Kaifeng and Jiang, Shunhua and Li, Jian

<details>
<summary>Abstract</summary>

> Training deep neural networks is a highly nontrivial task, involving carefully selecting appropriate training algorithms, scheduling step sizes and tuning other hyperparameters. Trying different combinations can be quite labor-intensive and time consuming. Recently, researchers have tried to use deep learning algorithms to exploit the landscape of the loss function of the training problem of interest, and learn how to optimize over it in an automatic way. In this paper, we propose a new learning-to-learn model and some useful and practical tricks. Our optimizer outperforms generic, hand-crafted optimization algorithms and state-of-the-art learning-to-learn optimizers by DeepMind in many tasks. We demonstrate the effectiveness of our algorithms on a number of tasks, including deep MLPs, CNNs, and simple LSTMs.
</details>

---

### [Learning to Optimize Neural Nets](https://www.semanticscholar.org/paper/5b8e5804c3adeb4a4e60f8f7d8d76aab0e02cfbe)
**Authors:** Li, Ke and Malik, Jitendra

<details>
<summary>Abstract</summary>

> Learning to Optimize is a recently proposed framework for learning optimization algorithms using reinforcement learning. In this paper, we explore learning an optimization algorithm for training shallow neural nets. Such high-dimensional stochastic optimization problems present interesting challenges for existing reinforcement learning algorithms. We develop an extension that is suited to learning optimization algorithms in this setting and demonstrate that the learned optimization algorithm consistently outperforms other known optimization algorithms even on unseen tasks and is robust to changes in stochasticity of gradients and the neural net architecture. More specifically, we show that an optimization algorithm trained with the proposed method on the problem of training a neural net on MNIST generalizes to the problems of training neural nets on the Toronto Faces Dataset, CIFAR-10 and CIFAR-100.
</details>

---

### [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://www.semanticscholar.org/paper/c889d6f98e6d79b89c3a6adf8a921f88fa6ba518)
**Authors:** Finn, Chelsea and Abbeel, Pieter and Levine, Sergey

<details>
<summary>Abstract</summary>

> We propose an algorithm for meta-learning that is model-agnostic, in the sense that it is compatible with any model trained with gradient descent and applicable to a variety of different learning problems, including classification, regression, and reinforcement learning. The goal of meta-learning is to train a model on a variety of learning tasks, such that it can solve new learning tasks using only a small number of training samples. In our approach, the parameters of the model are explicitly trained such that a small number of gradient steps with a small amount of training data from a new task will produce good generalization performance on that task. In effect, our method trains the model to be easy to fine-tune. We demonstrate that this approach leads to state-of-the-art performance on two few-shot image classification benchmarks, produces good results on few-shot regression, and accelerates fine-tuning for policy gradient reinforcement learning with neural network policies.
</details>

---

### [Learning to Learn without Gradient Descent by Gradient Descent](https://www.semanticscholar.org/paper/b5fdbacc37f1d5e1a72c292ac2107c06c7bd6d4f)
**Authors:** Chen, Yutian and Hoffman, Matthew W. and Colmenarejo, Sergio Gómez and Denil, Misha and Lillicrap, Timothy and Botvinick, Matt and Freitas, Nando de

*No abstract available.*

---

### [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://www.semanticscholar.org/paper/c889d6f98e6d79b89c3a6adf8a921f88fa6ba518)
**Authors:** Chelsea Finn

<details>
<summary>Abstract</summary>

> We propose an algorithm for meta-learning that is model-agnostic, in the sense that it is compatible with any model trained with gradient descent and applicable to a variety of different learning problems, including classification, regression, and reinforcement learning. The goal of meta-learning is to train a model on a variety of learning tasks, such that it can solve new learning tasks using only a small number of training samples. In our approach, the parameters of the model are explicitly trained such that a small number of gradient steps with a small amount of training data from a new task will produce good generalization performance on that task. In effect, our method trains the model to be easy to fine-tune. We demonstrate that this approach leads to state-of-the-art performance on two few-shot image classification benchmarks, produces good results on few-shot regression, and accelerates fine-tuning for policy gradient reinforcement learning with neural network policies.
</details>

---

### [Neural Optimizer Search with Reinforcement Learning](https://www.semanticscholar.org/paper/168b7d0ab57a331a228ce21ffd1becbb93066f79)
**Authors:** Bello, Irwan and Zoph, Barret and Vasudevan, Vijay and Le, Quoc V.

<details>
<summary>Abstract</summary>

> We present an approach to automate the process of discovering optimization methods, with a focus on deep learning architectures. We train a Recurrent Neural Network controller to generate a string in a domain specific language that describes a mathematical update equation based on a list of primitive functions, such as the gradient, running average of the gradient, etc. The controller is trained with Reinforcement Learning to maximize the performance of a model after a few epochs. On CIFAR-10, our method discovers several update rules that are better than many commonly used optimizers, such as Adam, RMSProp, or SGD with and without Momentum on a ConvNet model. We introduce two new optimizers, named PowerSign and AddSign, which we show transfer well and improve training on a variety of different tasks and architectures, including ImageNet classification and Google's neural machine translation system.
</details>

---

## 2016

### [Learning to reinforcement learn](https://www.semanticscholar.org/paper/282a380fb5ac26d99667224cef8c630f6882704f)
**Authors:** Wang, Jane X. and Kurth‐Nelson, Zeb and Tirumala, Dhruva and Soyer, Hubert and Leibo, Joel Z. and Munos, Rémi and Blundell, Charles and Kumaran, Dharshan and Botvinick, Matt

<details>
<summary>Abstract</summary>

> In recent years deep reinforcement learning (RL) systems have attained superhuman performance in a number of challenging task domains. However, a major limitation of such applications is their demand for massive amounts of training data. A critical present objective is thus to develop deep RL methods that can adapt rapidly to new tasks. In the present work we introduce a novel approach to this challenge, which we refer to as deep meta-reinforcement learning. Previous work has shown that recurrent networks can support meta-learning in a fully supervised context. We extend this approach to the RL setting. What emerges is a system that is trained using one RL algorithm, but whose recurrent dynamics implement a second, quite separate RL procedure. This second, learned RL algorithm can differ from the original one in arbitrary ways. Importantly, because it is learned, it is configured to exploit structure in the training domain. We unpack these points in a series of seven proof-of-concept experiments, each of which examines a key aspect of deep meta-RL. We consider prospects for extending and scaling up the approach, and also point out some potentially important implications for neuroscience.
</details>

---

### [Learning to Optimize](https://www.semanticscholar.org/paper/43c00597f8d59659af97dc35b24a836e8bd9b82a)
**Authors:** Li, Ke and Malik, Jitendra

*No abstract available.*

---

### [Learning to learn by gradient descent by gradient descent](https://www.semanticscholar.org/paper/71683e224ab91617950956b5005ed0439a733a71)
**Authors:** Andrychowicz, Marcin and Denil, Misha and Gómez, Sergio Luis Suárez and Hoffman, Matthew W. and Pfau, David and Schaul, Tom and Shillingford, Brendan and Freitas, Nando de

<details>
<summary>Abstract</summary>

> The move from hand-designed features to learned features in machine learning has been wildly successful. In spite of this, optimization algorithms are still designed by hand. In this paper we show how the design of an optimization algorithm can be cast as a learning problem, allowing the algorithm to learn to exploit structure in the problems of interest in an automatic way. Our learned algorithms, implemented by LSTMs, outperform generic, hand-designed competitors on the tasks for which they are trained, and also generalize well to new tasks with similar structure. We demonstrate this on a number of tasks, including simple convex problems, training neural networks, and styling images with neural art.
</details>

---

## 2007

### [Learning to Optimize](https://www.semanticscholar.org/paper/92c8217f50062673ab108f7ea64b20333f2008df)
**Authors:** Ören, Tuncer and Zeigler, Bernard P.

<details>
<summary>Abstract</summary>

> Learning to optimize (L2O) has gained increasing popularity, which automates the design of optimizers by data-driven approaches. However, current L2O methods often suffer from poor generalization performance in at least two folds: (i) applying the L2O-learned optimizer to unseen optimizees, in terms of lowering their loss function values (optimizer generalization, or ``generalizable learning of optimizers"); and (ii) the test performance of an optimizee (itself as a machine learning model), trained by the optimizer, in terms of the accuracy over unseen data (optimizee generalization, or ``learning to generalize"). While the optimizer generalization has been recently studied, the optimizee generalization (or learning to generalize) has not been rigorously studied in the L2O context, which is the aim of this paper. We first theoretically establish an implicit connection between the local entropy and the Hessian, and hence unify their roles in the handcrafted design of generalizable optimizers as equivalent metrics of the landscape flatness of loss functions. We then propose to incorporate these two metrics as flatness-aware regularizers into the L2O framework in order to meta-train optimizers to learn to generalize, and theoretically show that such generalization ability can be learned during the L2O meta-training process and then transformed to the optimizee loss function. Extensive experiments consistently validate the effectiveness of our proposals with substantially improved generalization on multiple sophisticated L2O models and diverse optimizees. Our code is available at: https://github.com/VITA-Group/Open-L2O/tree/main/Model_Free_L2O/L2O-Entropy.
</details>

---

## 2001

### [Learning to Learn Using Gradient Descent](https://www.semanticscholar.org/paper/044b2c29e0a54dc689786bd4d029b9ba6e355d58)
**Authors:** Hochreiter, Sepp and Younger, A. Steven and Conwell, Peter R.

*No abstract available.*

---

