# NeurodynamicalSystems

NOTE: THIS PACKAGE IS STILL IN DEVELOPMENT

This is a high performance machine learning library built around the predictive coding algorithm (PC) used by the human brain. It implements several standard predictive coding with hierarchical Gaussian models as well as other observed neural circuits. The primary goal of this library is to compare the performance of PC to more common deep learning frameworks (i.e. feedforward networks trained with backpropogation) by facilitating the implementation and evaluation of PC models for popular deep learning tasks. 

It can be used for computer vision/listening, natural language processing, sparse coding, multimodal perception, or anything else you'd use deep learning for, and eventually will cover agent behavior control via active inference (AI). It is fully GPU compatible and designed to feel like Flux.jl, though the internals are nothing alike. Also, it's really fast!

# Introduction

## Relevant papers
None of these papers are affiliated with this library, but I've implemented the algorithms (and/or variants of those) they describe. My paper on this library is coming soon! For now, read the papers below to learn about what this package does.

[1] Millidge, Beren, Anil Seth, and Christopher L. Buckley. “Predictive Coding: A Theoretical and Experimental Review.” arXiv, July 12, 2022. https://doi.org/10.48550/arXiv.2107.12979.

[2] Tscshantz, Alexander, Beren Millidge, Anil K. Seth, and Christopher L. Buckley. “Hybrid Predictive Coding: Inferring, Fast and Slow.” PLOS Computational Biology 19, no. 8 (August 2, 2023): e1011280. https://doi.org/10.1371/journal.pcbi.1011280.

[3] Salvatori, Tommaso, Yuhang Song, Thomas Lukasiewicz, Rafal Bogacz, and Zhenghua Xu. “Predictive Coding Can Do Exact Backpropagation on Convolutional and Recurrent Neural Networks.” arXiv, March 5, 2021. https://doi.org/10.48550/arXiv.2103.03725.

[4] Millidge, Beren, Tommaso Salvatori, Yuhang Song, Rafal Bogacz, and Thomas Lukasiewicz. “Predictive Coding: Towards a Future of Deep Learning beyond Backpropagation?” arXiv, February 18, 2022. http://arxiv.org/abs/2202.09467.

[5] Shaojie Bai, J. Zico Kolter, Vladlen Koltun. Deep Equilibrium Models. https://arxiv.org/abs/1909.01377

[6] C. Rozell, D. Johnson, R. Baraniuk and B. Olshausen, "Locally Competitive Algorithms for Sparse Approximation," 2007 IEEE International Conference on Image Processing, San Antonio, TX, USA, 2007, pp. IV - 169-IV - 172, doi: 10.1109/ICIP.2007.4379981.

[7] Rutishauser U, Douglas RJ, Slotine JJ. Collective stability of networks of winner-take-all circuits. Neural Comput. 2011 Mar;23(3):735-73. doi: 10.1162/NECO_a_00091. Epub 2010 Dec 16. PMID: 21162667.

[8] Dylan Paiton, Sheng Lundquist, William Shainin, Xinhua Zhang, Peter Schultz, and Garrett Kenyon. 2016. A Deconvolutional Competitive Algorithm for Building Sparse Hierarchical Representations. In Proceedings of the 9th EAI International Conference on Bio-inspired Information and Communications Technologies (formerly BIONETICS) (BICT'15). ICST (Institute for Computer Sciences, Social-Informatics and Telecommunications Engineering), Brussels, BEL, 535–542. https://doi.org/10.4108/eai.3-12-2015.2262428

## Background

Neurodynamical systems were first popularized in neuroscience circles in the 1980s, where they were used primarily to explain the behavior of real neurons in brains. With recent advances in computing, dynamical systems, and deep learning, they have seen increasing application to general machine perception and signal processing tasks. The general idea of predictive coding is to form some hierarchical generative model of the data where each "layer" predicts the layer below based on its activations and parameters. The forward pass uses an iterative solver to find the activations of each layer which minimize prediction error given the parameters, while the backward pass updates the parameters to minimize prediction error given the activations. This is done in a layer-by-layer manner with analytically derived update rules, so no automatic differentiation is ever used at any point. This procedure efficiently yields excellent hierarchcial sparse codes, in particular for complex sequence modeling tasks. Other neurodynamical systems add extra functionality, such as classification via winner-takes-all neural circuits, agent behavior control via active inference, and short-term working memory via hippocampus models.

## Why use neurodynamical systems instead of other deep-learning architectures?

The current consensus is that feedforward networks trained via backpropagation perform at least as well as PC networks and may be more efficient to train. However, vastly more effort has gone into optimizing the performance of backpropagation-based systems, and there is sufficient evidence to show that PC ought to be taken more seriously. Below are the most compelling benefits: 

1. **Memory efficiency:** Deep learning requires extremely expensive hardware primarily due to the memory cost. PC networks can cut the memory cost by many orders of magnitude thanks to implicit layers and array mutation. In essence, feedforward networks independently train dozens if not hundreds of independent layers, while PC networks train a single layer to be applied iteratively. Miraculously, this does not compromise performance. Because feedforward networks are trained via backpropagation, all hidden layers must be stored for training. Even worse, automatic differentiation software typically requires these arrays to be allocated on the heap during runtime, which can be quite slow. PC networks can learn with just the final iteration output, allowing us to overwrite/mutate each intermediate array. This dramatically reduces the memory footprint and time spent allocating and doing garbage collection.

2. **Hierarchical models and feedback signals:** Hierarchical NDSs can be exceptionally powerful due to their ability to incorporate top-down feedback. Most DNNs only use feedforward bottom-up excitation/inhibition where each layer's activation only depends on the activation of the previous layer. This prevents the use of higher level information to more precisely analyze the simpler lower level signals. 

   For example, take an object detection CNN. Say it has a neuron to detect ping-pong balls, and another to detect golf balls (as well as lots of other object filters). Because golf balls and ping-pong balls look nearly identical in an image, such a network will struggle to distinguish between the two. Yet our brains almost never mix the two up, due to our ability to incorporate high-level feedback. How does this work? Well, say the image also contains lush grass and a bag of golf clubs. It would be clear that the image was taken at a golf course, which would correspond to some other higher level neuron. Because we know golf balls are common at golf courses while ping-pong balls are not, the activation of the high-level golf course neuron should send a positive feedback signal to the golf ball neuron, indicating that this neuron expects to see golf balls. PC networks do precisely this.

   Thus, NDSs can take a blurry input (both of the ball neurons partially active), form a rough high level scene (I am probably at a golf course), then sharpen the inputs (since I am probably at a golf course, that is likely a golf ball instead of a ping-pong ball), and finally sharpen the high level scene (since that was a golf ball, I am definitely at a golf course). This is done iteratively until converging on a fixed point. Incorporating feedback like this makes it much easier to distinguish between very similar objects.

3. **Infinitely long dependencies in sequence models:** The top-down feedback of PC networks allows them to capture arbitrarily long dependencies in sequences. Consider a PC network for language modeling, which works like a hierarchical word2vec network. Assume a context window of 3 and *K* layers. The input layer *L<sub>0</sub>* is one-hot encoded, and array *i* of the first layer *L<sub>1</sub>* encodes arrays *i-1:i+1* of the input. To form this encoding, layer *L<sub>1</sub>* attempts to predict layer *L<sub>0</sub>*, then incrementally adjusts its activations to improve this prediction via gradient descent. A context window of 3 means that element *i* of layer *L<sub>k</sub>* will learn weights to predict elements *i-1:i+1* of the layer *L<sub>k-1</sub>*.

   Now, we repeat this pattern for layers *L<sub>2</sub>:L<sub>K</sub>*, with each layer predicting the layer below. With each layer, the effective window of context widens by 1, always remaining finite. We must make one slight change to achieve infinite dependencies: the top-down feedback. To incorporate this, each layer *L<sub>k</sub>* will simultaneously update its activations to better predict layer L<sub>k-1</sub> and **to be more consistent with the predictions of layer *L<sub>k+1</sub>*. This allows the network to use higher-level context to refine lower-level encodings. One may also update the input data itself to be more consistent with the predictions, which is the essence of associative-memory models like Hopfield networks.

   A side effect of this modification is that all hidden neurons become indirectly dependent on each other, no matter how far apart they appear in the sequence (if you aren't convinced, try to draw the Bayesian graph of conditional independence, noting that all relationships are bidirectional. With this, you should be able to trace a path between any two hidden neurons). This could be a more efficient way to model long-range dependencies than the attention mechanism of transformers. While transformers identify pairs of words who impact each others' encodings, PC networks identify the presence of high-level concepts based on low-level encodings, then refine low level encodings based on the present concepts. This could be incredibly powerful for long-form text in which transformers must compute the attention between every single pair of words, even though nearly all pairs will be unrelated. This also dramatically reduces the memory cost and allocations. 

4. **Online/offline usage:** It is often desirable to pretrain a sequence model offline then use/fine-tune it in a real-time online environment. PC networks excel at this as the same network can be used to model time-varying or static inputs. Consider the same model as before with a buffer/context window size *M.* In online use, the buffer must cover the current sample as well as at least one past and one future sample. The future samples are always unobserved (input is a zero vector), but the network will still try to predict them. Their activations will then change to become more consistent with the predictions, meaning encodings are still influenced by predicted yet unobserved data. With each new incoming sample, the activations are shifted through the buffer with the oldest being dropped and the newest being the added zero vector.

   This provides speedy online inference through a typical recurrent structure. However, if the data is available offline, we can "convolve" the whole network with the input sequence. All samples of the sequence would be considered simultaneously at each iteration, leading to better encodings (especially for early elements). This is akin to letting the network reread/reconsider old samples in light of new samples (similar to how humans read and write) while the online case only considers each sample for a brief moment (similar to how humans listen/speak). 

5. **Connection to neuroscience:** Deep learning aims to replicate many of the unique capabilities of the human brain but uses a radically different mathematical framework. This makes it incredibly difficult to use neuroscience literature to guide the development of better neural networks. However, PC is directly inspired by sensory processing in the human brain. This link could allow deep learning researchers to use centuries of neuroscience findings to solve major open problems in deep learning, in particular with respect to "artificial general intelligence."

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
