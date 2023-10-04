# NeurodynamicalSystems

NOTE: THIS PACKAGE IS STILL IN DEVELOPMENT

This is a high-performance machine learning library built around the predictive coding algorithm used by the human brain and other neurodynamical systems. It implements several standard predictive coding with hierarchical Gaussian models as well as other neural circuits observed in the brain. It can be used for anything in computer vision/listening, natural language processing, sparse coding, agent behavior control etc., or anything else you'd use deep learning for. 

It is fully GPU compatible and can be used for either supervised or unsupervised learning. Also, it's really fast!

# Introduction

## Relevant papers
None of these papers are affiliated with this library, but I've implemented the algorithms they describe. My paper on this library is coming soon! For now, read the papers below to learn about what this package does.

[1] Millidge, Beren, Anil Seth, and Christopher L. Buckley. “Predictive Coding: A Theoretical and Experimental Review.” arXiv, July 12, 2022. https://doi.org/10.48550/arXiv.2107.12979.

[2] Tscshantz, Alexander, Beren Millidge, Anil K. Seth, and Christopher L. Buckley. “Hybrid Predictive Coding: Inferring, Fast and Slow.” PLOS Computational Biology 19, no. 8 (August 2, 2023): e1011280. https://doi.org/10.1371/journal.pcbi.1011280.

[3] Salvatori, Tommaso, Yuhang Song, Thomas Lukasiewicz, Rafal Bogacz, and Zhenghua Xu. “Predictive Coding Can Do Exact Backpropagation on Convolutional and Recurrent Neural Networks.” arXiv, March 5, 2021. https://doi.org/10.48550/arXiv.2103.03725.

[4] Millidge, Beren, Tommaso Salvatori, Yuhang Song, Rafal Bogacz, and Thomas Lukasiewicz. “Predictive Coding: Towards a Future of Deep Learning beyond Backpropagation?” arXiv, February 18, 2022. http://arxiv.org/abs/2202.09467.

[5] Shaojie Bai, J. Zico Kolter, Vladlen Koltun. Deep Equilibrium Models. https://arxiv.org/abs/1909.01377

[6] C. Rozell, D. Johnson, R. Baraniuk and B. Olshausen, "Locally Competitive Algorithms for Sparse Approximation," 2007 IEEE International Conference on Image Processing, San Antonio, TX, USA, 2007, pp. IV - 169-IV - 172, doi: 10.1109/ICIP.2007.4379981.

[7] Rutishauser U, Douglas RJ, Slotine JJ. Collective stability of networks of winner-take-all circuits. Neural Comput. 2011 Mar;23(3):735-73. doi: 10.1162/NECO_a_00091. Epub 2010 Dec 16. PMID: 21162667.

[8] Dylan Paiton, Sheng Lundquist, William Shainin, Xinhua Zhang, Peter Schultz, and Garrett Kenyon. 2016. A Deconvolutional Competitive Algorithm for Building Sparse Hierarchical Representations. In Proceedings of the 9th EAI International Conference on Bio-inspired Information and Communications Technologies (formerly BIONETICS) (BICT'15). ICST (Institute for Computer Sciences, Social-Informatics and Telecommunications Engineering), Brussels, BEL, 535–542. https://doi.org/10.4108/eai.3-12-2015.2262428

## Background

Neurodynamical systems were first popularized in neuroscience circles in the 1980s, where they were used primarily to explain the behavior of real neurons in brains. With recent advances in computing, dynamical systems, and deep learning, they have seen increasing application to general machine perception and signal processing tasks. The general idea is to form some hierarchical generative model of the data where each "layer" predicts the layer below based on its activations and parameters. The forward pass uses an ODE system or nonlinear solver to find the activations of each layer which minimize prediction error given the parameters, while the backward pass updates the parameters to minimize prediction error given the activations. This is done in a layer-by-layer manner with analytically derived update rules, so no automatic differentiation is ever used at any point.

This package implements a variety of popular neurodynamical systems with a focus on predictive coding. It is designed to give a Flux.jl-like experience, but under the hood is nothing like it or other mainstream deep learning libraries.

## Why use neurodynamical systems instead of other deep-learning architectures?

Generally speaking, a single-layer neurodynamical system (NDS) is just like a typical feedforward network (FNN) layer applied iteratively, or equivalently, a deep feedforward network where all layers are constrained to have equal weights. These so-called weight-tied networks are almost always equally expressive as their traditional explicit feedforward counterparts.[1] All NDSs converge to a fixed point for a static input, meaning that given a static input, each iteration brings the neuron activations closer and closer to some fixed point. Once reached, activations remain unchanged with successive applications of the layer. This means that a single-layer NDS can represent arbitrarily deep FNNs, leading to massive reductions in memory and computing requirements.[1]

## Online inference
NDSs are excellent for sequence modeling and online real-time inference. When used for time-varying input signals, they provide natural smoothing just as the human brain does. The level of smoothing for each hierarchical layer can be controlled independently with tuning parameters, allowing these models to capture richer and more realistic dynamics in a machine perception context (e.g. little details can change quickly, but more abstract features like the setting or overall environment can change very slowly).

## Hierarchical models and feedback signals
Hierarchical NDSs can be exceptionally powerful due to their ability to incorporate top-down feedback. Most DNNs only use feedforward bottom-up excitation/inhibition where each layer's activation only depends on the activation of the previous layer. This prevents the use of higher level information to more precisely analyze the simpler lower level signals. 

For example, take an object detection CNN. Say it has a neuron to detect ping-pong balls, and another to detect golf balls (as well as lots of other object filters). Because golf balls and ping-pong balls look nearly identical in an image, such a network will struggle to distinguish between the two. Yet our brains almost never mix the two up, due to our ability to incorporate high-level feedback. How does this work? Well, say the image also contains lush grass and a bag of golf clubs. It would be clear that the image was taken at a golf course, which would correspond to some other high level neuron. Because we know golf balls are common at golf courses while ping-pong balls are not, the activation of the high-level golf course neuron should send a positive feedback signal to the golf ball neuron, indicating that this neuron expects to see golf balls. 

Thus, NDSs can take a blurry input (both of the ball neurons partially active), form a rough high level scene (I am probably at a golf course), then sharpen the inputs (since I am probably at a golf course, that is likely a golf ball instead of a ping-pong ball), and finally sharpen the high level scene (since that was a golf ball, I am definitely at a golf course). This is done iteratively until converging on a fixed point. Incorporating feedback like this makes it much easier to distinguish between very similar objects.

An approach like this can solve many problems with today's deep learning architectures. For example, hierarchical dynamical transformers could lead to massively improved efficiency on long-form text. Rather than using each individual word token to update the encoding of every other token, an NDS could dynamically map tokens to sentence ideas, sentences to paragraph ideas, paragraphs to chapters etc. Then, the attention mechanism could have, e.g., paragraphs attend to other paragraphs, and additional top-down feedback, instead of just having all the words in the entire text attend to each other (the latter of which comes with absurd memory requirements). This general concept of forming hierarchical summaries can be naturally extended to other input data types, like images and audio, and there are countless ways to impement it with NDSs.

# Workflow

*See Examples.jl for a better guide on usage until I update the documentation.* 

The general workflow is this:
1. Define layers: You must have one input layer appropriately sized for the data, and at least two more "hidden layers." Layers are not callable as in Flux, only serving to help the user build complex architectures.

2. Define a predictive coding module. Modules are hierarchical collections of layers or other modules. They store the parameters and other state variables like the predictions and errors. These are callable with arguments structured to be compatible with DifferentialEquations.jl. Calling the module calculates the update values in the ODE solver, so it is passed to ODEProblem.

3. Define a predictive coding network. This callable structure stores the predictive coding module, ODEProblem, and ODE solution. Calling it on the input data will initialize the lowest level's states to the input, and initializes all other levels with a trainable feedforward network. Then, it runs the ODE system for some time frame to compute the final activations.

4. Train the predictive coding network. Calling train!(pcnetwork, ...) will run the dynamical system then interrupt it at specified times via a callback function to calculate the weight updates. So, training doesn't happen with each time step of the ODE solver, but only the later ones.

# Layer/module types
This gives a brief overview of the current functionality of this package.

1. DenseModule: adjacent layers are connected via dense matrix multiplications.
   Input structure: each sample is a column in a matrix.
2. ConvModule: adjacent layers are connected via convolutions.
   Input structure: like with convolutions in Flux, data is stored in an array such that     iterating through the last dimension will iterate through the samples, while iterating    through the second to last iterates through channels. For N-dimensional convolutions      (N = 2 for images), the input must have N + 2 dimensions
3. SparseModule: adjacent layers are connected via sparse matrix multiplications.
   Input structure: same as DenseModule.
4. Dense1DSequenceModule: used for 1D sequences like text where each element of the          sequence is a column of a matrix. The input can only hold one sequence at a time, so      training is fastest when the sequences are very long.
5. Sparse1DSequenceModule: same as Dense1DSequenceModule but the weight matrices are         sparse. Useful when the sequence has a very large number of channels.

*This list will grow over time. I will soon add several Projection Neural Networks, Zeroing Neural Networks, several Hopfield network variants, and probably more advanced WTA circuits.* 



[![Build Status](https://github.com/EliSmith45/NeurodynamicalSystems.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/EliSmith45/NeurodynamicalSystems.jl/actions/workflows/CI.yml?query=branch%3Amaster)
