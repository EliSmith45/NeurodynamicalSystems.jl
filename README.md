# NeurodynamicalSystems

This package implements a variety of biologically-inspired (though not necessarily biologically plausible) neural networks, i.e., dynamical systems that model neural circuits, used for typical machine learning tasks.

# Introduction
## Background

Neurodynamical systems were first popularized in neuroscience circles in the 1980s, where they were used primarily to explain the behavior of real neurons in brains. With recent advances in computing, dynamical systems, and deep learning, they have seen increasing application to general machine learning tasks, such as computer vision/listening, sparse coding, segmentation, object tracking etc. They are suitable for any deep learning task and are typically far more parameter efficient, making them surprisingly easier/faster to train than equally expressive feedforward networks.

This package implements a variety of popular neurodynamical systems as well as a few novel ones of my own. They are designed to give a Flux.jl-like experience, using stateful functions similar to flux RNN cells. This package is still early in development, so you may run into issues with training, but I will soon ensure that they are all compatible with Zygote for automatic differentiation. I'll also write CUDA GPU kernels for easy parallelization. 

## Why use neurodynamical systems instead of other deep-learning architectures?

Generally speaking, a single-layer neurodynamical system (NDS) is just like a typical feedforward network (FNN) layer applied iteratively, or equivalently, a deep feeddforward network where all layers are constrained to have equal weights. Surprisingly, these so-called weight-tied networks are almost always equally expressive as their traditional feedforward counterparts. [1] Most NDSs converge to a fixed point, meaning that given a static input, each iteration brings the neuron activations closer and closer to some fixed point. Once reached, activations remain unchanged with successive applications of the layer. This means that a single-layer NDS can represent arbitrarily deep FNNs, leading to massive reductions in unique parameters.

## Intuition 
To see the intution here, it is important to first understand the role that depth plays in deep neural networks (DNNs). Most often, the given explanation is that they can find hierarchical patterns. While true, depth plays another role, which is arguably far more important: they can nonlinearly "sharpen" a blurred input to create better sparse codes. 

In all neural networks, artificial or otherwise, neurons represent latent variables or patterns in an input signal. Each neuron *i* in layer *L* stores its corresponding pattern in its receptive field (row *i* of the *L<sup>th</sup>* weight matrix), and their activation is found by taking the inner product of their receptive field with their input (activations of layer *L-1* and applying some nonlinear activation function. The activation of a neuron should signal the presence of the pattern in the input. In practice, it rarely does, because the patterns/receptive fields are almost always non-orthogonal. This means that receptive fields will be correlated. If pattern *i* appears in the input, and neuron *j* is correlated with neuron *i,* its activation will likely be nonzero even when *j* doesn't appear in the input. 

Ideally, neuron *i* will supress neuron *j* in this scenario, thus creating a "sharper" and more sparse code. This can only be done iteratively. DNNs learn lots of unique layers that sequentially sharpen the input a little bit with each iteration in addition to finding hierarchical patterns. Using depth for this purpose is *incredibly* wasteful. NDSs solve this same problem by iteratively applying the same function, which works equally well (if not better) despite using dramatically fewer parameters.

## Online inference
NDSs are excellent for sequence modeling and online real-time inference. Using them for time-varying input signals provides natural time smoothing just like what the human brain does, and tuning of each layers' time constant gives the user control over the degree of smoothing. 

## Hierarchical models and feedback signals
Hierarchical NDSs can be exceptionally powerful due to their ability to incorporate top-down feedback. Most DNNs only use feedforward bottom-up excitation/inhibition where each layer's activation only depends on the activation of the previous layer. This prevents the use of higher level information to more precisely analyze lower level signals. 

For example, take an object detection CNN. Say it has a neuron to detect ping-pong balls, and another to detect golf balls (as well as tons of other object filters). Because golf balls and ping-pong balls look nearly identical in an image, such a network will struggle to distinguish between the two. Yet our brains almost never mix the two up due to our ability to incorporate high-level feedback. How does this work? Well, say the image also contains lush grass and a bag of golf clubs. It would be clear that the image was taken at a golf course, which would correspond to some other high level neuron. Because we know golf balls are common at golf courses while ping-pong balls are not, the activation of the high-level golf course neuron should send a positive feedback signal to the golf ball neuron, indicating that this neuron expects to see golf balls. 

Thus, NDSs can take a blurry input (both of the ball neurons partially active), form a rough high level scene (I am probably at a golf course), then sharpen the inputs (since I am at a golf course, that is definitely a golf ball instead of a ping-pong ball), and finally sharpen the high level scene (I am definitely at a golf course). This is done iteratively until converging on a fixed point.

Such an approach can solve tons of problems with today's deep learning architectures. For example, hierarchical dynamical transformers could lead to massively improved efficiency on long-form text. Rather than using each individual word token to update the embedding of every other token, an NDS could map tokens to sentence ideas, sentences to paragraph ideas, paragraphs to chapters etc. Then, the attention mechanism could have paragraphs attend to other paragraphs, instead of having the words in each paragraph attend to each other, the latter of which comes with absurd memory requirements. This general concept of forming hierarchical summaries can be naturally extended to other input data types, like images and audio.


# Layer types

Each unique neural circuit will be called a "layer." I'll update this README with more details soon when I refine the contents of this package, but for now I will just mention the layers and provide accompanying papers that describe them better. If you understand recurrent functions in Flux, you should be able to follow the code fairly easily. 

1. Lca: implements the basic Locally Competitive Algorithm. [2]
2. LcaRms: Novel algorithm that implements the LCA with a modified update rule based on the RMSProp optimizer for gradient descent.
3. LcAdam: Novel algorithm that implements the LCA with a modified update rule based on the Adam optimizer for gradient descent. Use this instead of the others whenever using LCA, since it has much nicer convergence and gives essentially the same results (up to a scaling factor).
4. Wta: implements the winner-takes-all neural circuit, which can produce very complicated dynamics when arranged into networks of WTA networks.[3]
5. WtaConv: Convolutional WTA network where local windows of neurons are assigned to (possibly overlapping) WTA circuits. This simple example of a network of WTA networks is quite useful for spectrum deconvolution or image sharpening. For example, say you apply an IIR filterbank to an audio signal. The resulting time-frequency distribution will be "smudged" by the frequency response of each filter, meaning that, given a, say, 400Hz signal, the resulting spectrum will have positive spectral energy for, e.g., the 398, 399, 400, 401, 402Hz frequency filter. Each filter corresponds to a neuron, and we want the 400Hz neuron to inhibit its neighbors. The simplest approach is peak-picking, where each neuron is set to zero unless it is a peak. This only works when the frequencies present in the signal are sufficiently spread out, which almost never occurs in real audio. 

The LCA algorithm will eventually converge to a sparse code representing the true deconvolved spectrum if each receptive field is equivalent to the frequency response of the corresponding filter. However, convergence can be incredibly slow, because adjacent neurons have highly correlated receptive fields. So, the 399Hz and 401Hz filters will inhibit the true 400Hz neuron as much as it inhibits them, so it can take hundreds of thousands of iterations to converge.

But, we can assume that 400Hz and 401Hz signals are perceptually indistinguishable, meaning we can treat them as just one 400.5Hz signal. By grouping windows of nearby filters into WTA circuits in addition to using LCA, the network encodes this assumption. Convergence becomes much faster, and perceptually distinct frequencies are still properly separated. This allows for incredibly fast and accurate time-frequency superresolution, which is useful for audio source separation. 

*This list will grow over time. I will soon add several Projection Neural Networks, Zeroing Neural Networks, and probably more advanced WTA circuits.* 

# More ideas
These layers can be arranged into more complex networks. I'll be exploring the idea of hierarchical feedback and will soon introduce tools/examples for dealing with such networks. Paiton et. al. gives a good example of a hierarchcial LCA model that I'm working on now. I'll also soon implement more useful examples, as I originally intended to use these networks for audio source separation. I also will eventually add more biologically-inspired update rules such as Hebbian plasticity, which may work better than automatic differentiation for online learning. This could be useful for forever-learning machines.



# Citations

[1] Shaojie Bai, J. Zico Kolter, Vladlen Koltun. Deep Equilibrium Models. https://arxiv.org/abs/1909.01377
[2] C. Rozell, D. Johnson, R. Baraniuk and B. Olshausen, "Locally Competitive Algorithms for Sparse Approximation," 2007 IEEE International Conference on Image Processing, San Antonio, TX, USA, 2007, pp. IV - 169-IV - 172, doi: 10.1109/ICIP.2007.4379981.
[3] Rutishauser U, Douglas RJ, Slotine JJ. Collective stability of networks of winner-take-all circuits. Neural Comput. 2011 Mar;23(3):735-73. doi: 10.1162/NECO_a_00091. Epub 2010 Dec 16. PMID: 21162667.
[4] Dylan Paiton, Sheng Lundquist, William Shainin, Xinhua Zhang, Peter Schultz, and Garrett Kenyon. 2016. A Deconvolutional Competitive Algorithm for Building Sparse Hierarchical Representations. In Proceedings of the 9th EAI International Conference on Bio-inspired Information and Communications Technologies (formerly BIONETICS) (BICT'15). ICST (Institute for Computer Sciences, Social-Informatics and Telecommunications Engineering), Brussels, BEL, 535–542. https://doi.org/10.4108/eai.3-12-2015.2262428

[![Build Status](https://github.com/EliSmith45/NeurodynamicalSystems.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/EliSmith45/NeurodynamicalSystems.jl/actions/workflows/CI.yml?query=branch%3Amaster)
