# 6-Month AI/ML Interview Mastery Roadmap
**Timeline: January - June 2026**  
**Target: Top Tech Company Level AI/ML Engineering & Research Positions**

---

## GitHub Repository Structure

```
6-Month-AI-ML-Interview-Mastery/
├── README.md
├── month-1-foundations/
│   ├── week-1-ml-fundamentals/
│   │   ├── notes/
│   │   ├── code/
│   │   ├── problems/
│   │   └── interview-questions.md
│   ├── week-2-linear-models/
│   ├── week-3-optimization/
│   └── week-4-evaluation/
├── month-2-classical-ml/
├── month-3-deep-learning/
├── month-4-advanced-architectures/
├── month-5-llms-transformers/
├── month-6-rl-systems/
├── interview-prep/
│   ├── company-prep/
│   ├── behavioral/
│   ├── system-design/
│   └── mock-interviews/
└── daily-log.md
```

---

## Month 1: Foundations & Linear Models (January 2026)

### Week 1: ML Fundamentals & Linear Regression
**Core Concept**: Understand prediction through linear relationships

#### Topics Integrated:
- **Problem Setup**: Supervised learning, loss functions, hypothesis space
- **Linear Regression from Scratch**:
  - Matrix formulation: y = Xw + b
  - Derive Normal Equation: w = (X^T X)^(-1) X^T y
  - **Math embedded**: Linear algebra (matrix multiplication, inverses, rank)
  - Geometric interpretation: projection onto column space
- **Loss Functions**:
  - MSE derivation and why it's convex
  - MAE for robust regression
  - Huber loss (combining both)
- **Gradient Descent Introduction**:
  - **Calculus embedded**: Partial derivatives, chain rule
  - Batch vs SGD vs Mini-batch
  - Learning rate schedules
  - Implement from scratch in NumPy

#### Implementation Tasks:
1. Linear regression with analytical solution
2. Linear regression with gradient descent
3. Polynomial regression (overfitting demonstration)
4. Ridge regression (L2 regularization)
5. Lasso regression (L1 regularization)
6. **Regularization math**: Why L2 shrinks weights, L1 creates sparsity

#### Interview Focus:
- "Derive the gradient of MSE loss"
- "When would you use Ridge vs Lasso?"
- "Explain bias-variance tradeoff" (introduce here, revisit throughout)
- "Code linear regression in 20 minutes"

---

### Week 2: Logistic Regression & Classification
**Core Concept**: From regression to classification via probability

#### Topics Integrated:
- **Sigmoid Function**:
  - Why sigmoid? Derive from log-odds
  - **Probability theory embedded**: Bernoulli distribution, MLE
- **Cross-Entropy Loss**:
  - Derive from negative log-likelihood
  - Why not MSE for classification?
  - Connection to KL divergence
- **Optimization**:
  - Newton's method for logistic regression
  - **Second-order optimization**: Hessian matrix
  - Compare with gradient descent
- **Multi-class Extension**:
  - Softmax function derivation
  - Categorical cross-entropy
  - One-vs-rest vs softmax

#### Implementation Tasks:
1. Binary logistic regression from scratch
2. Multiclass softmax classifier
3. Newton's method optimizer
4. Regularized logistic regression (elastic net)
5. Visualization: decision boundaries

#### Interview Focus:
- "Derive gradient of cross-entropy loss"
- "Why sigmoid for binary, softmax for multi-class?"
- "Implement softmax with numerical stability"
- "What's the relationship between logistic regression and neural networks?"

---

### Week 3: Optimization Deep Dive
**Core Concept**: How models learn efficiently

#### Topics Integrated:
- **Gradient Descent Variants**:
  - SGD with momentum (exponential moving average)
  - AdaGrad (adaptive learning rates, why it helps with sparse gradients)
  - RMSprop (solution to AdaGrad's diminishing learning rate)
  - Adam (combining momentum + RMSprop)
  - **Calculus embedded**: Taylor series approximation of loss
- **Convergence Analysis**:
  - Convex vs non-convex optimization
  - Local vs global minima
  - Saddle points in high dimensions
- **Learning Rate Schedules**:
  - Step decay, exponential decay, cosine annealing
  - Warmup strategies
- **Batch Normalization Foundations**:
  - Internal covariate shift
  - Normalization as optimization technique

#### Implementation Tasks:
1. Implement all optimizers from scratch
2. Visualize optimization trajectories
3. Compare convergence on convex/non-convex surfaces
4. Learning rate finder implementation

#### Interview Focus:
- "Explain Adam optimizer step-by-step"
- "Why does momentum help?"
- "What problems does batch norm solve?"
- "How would you debug slow convergence?"

---

### Week 4: Model Evaluation & Validation
**Core Concept**: Measuring and ensuring generalization

#### Topics Integrated:
- **Cross-Validation**:
  - K-fold, stratified, time series split
  - **Statistical theory**: sampling, confidence intervals
- **Metrics Beyond Accuracy**:
  - Precision, recall, F1 (derive from confusion matrix)
  - ROC curves, AUC (interpretation)
  - PR curves for imbalanced data
- **Bias-Variance Tradeoff**:
  - Mathematical formulation
  - Learning curves
  - Regularization revisited
- **Hypothesis Testing**:
  - Statistical significance of model improvements
  - A/B testing fundamentals

#### Implementation Tasks:
1. Cross-validation framework
2. Metrics library (all classification/regression metrics)
3. ROC/PR curve plotting
4. Bootstrap confidence intervals
5. Learning curve analysis tool

#### Interview Focus:
- "How do you handle imbalanced datasets?"
- "Precision vs Recall tradeoff scenarios"
- "Design an A/B test for a model deployment"
- "How would you detect overfitting?"

---

## Month 2: Classical ML & Feature Engineering (February 2026)

### Week 5: Decision Trees & Ensemble Methods
**Core Concept**: Non-linear decision boundaries through tree-based models

#### Topics Integrated:
- **Decision Trees**:
  - Information gain, Gini impurity, entropy (from information theory)
  - **Probability theory**: conditional probability, mutual information
  - Splitting criteria derivation
  - Pruning strategies
- **Random Forests**:
  - Bootstrap aggregating (bagging)
  - Feature randomness
  - Out-of-bag error
  - **Statistical theory**: variance reduction through averaging
- **Gradient Boosting**:
  - Functional gradient descent
  - XGBoost, LightGBM, CatBoost internals
  - Learning rate and regularization in boosting
  - **Optimization**: second-order approximation

#### Implementation Tasks:
1. Decision tree from scratch (ID3/CART)
2. Random forest implementation
3. Gradient boosting from scratch (simple version)
4. Feature importance analysis
5. Hyperparameter tuning framework

#### Interview Focus:
- "Random Forest vs Gradient Boosting: when to use what?"
- "Explain XGBoost's advantages"
- "How does feature importance work in RF?"
- "Implement decision tree split finding"

---

### Week 6: Support Vector Machines & Kernel Methods
**Core Concept**: Maximum margin classification and kernel trick

#### Topics Integrated:
- **SVM Formulation**:
  - Margin maximization (geometric intuition)
  - **Optimization**: Lagrange multipliers, KKT conditions
  - Primal and dual formulations
  - Support vectors concept
- **Kernel Trick**:
  - Feature space mapping without explicit computation
  - Common kernels: polynomial, RBF, sigmoid
  - **Linear algebra**: kernel as inner product
  - Mercer's theorem (intuition)
- **Soft Margin SVM**:
  - Slack variables for non-separable data
  - C parameter interpretation
  - Hinge loss connection

#### Implementation Tasks:
1. Hard-margin SVM (linearly separable case)
2. Soft-margin SVM with SMO algorithm (simplified)
3. Kernel SVM with RBF kernel
4. Compare with logistic regression on various datasets

#### Interview Focus:
- "Explain the kernel trick intuitively"
- "What's the role of C in SVM?"
- "When would you use SVM vs neural networks?"
- "Derive the SVM objective function"

---

### Week 7: Dimensionality Reduction & Unsupervised Learning
**Core Concept**: Finding structure in unlabeled data

#### Topics Integrated:
- **PCA**:
  - Covariance matrix and eigenvectors
  - **Linear algebra**: eigenvalue decomposition, spectral theorem
  - Maximum variance vs minimum reconstruction error (equivalence proof)
  - Whitening transformation
- **t-SNE**:
  - Probability distributions in high/low dimensions
  - KL divergence minimization
  - Perplexity parameter
- **UMAP** (brief overview)
- **Clustering**:
  - K-means (EM algorithm connection)
  - Hierarchical clustering (linkage methods)
  - DBSCAN (density-based)
  - **Evaluation**: Silhouette score, Davies-Bouldin index

#### Implementation Tasks:
1. PCA from scratch (using SVD)
2. K-means with k-means++ initialization
3. Hierarchical clustering with dendrograms
4. t-SNE visualization (using libraries, understand internals)

#### Interview Focus:
- "Explain PCA mathematically"
- "K-means vs hierarchical clustering"
- "How to choose number of clusters?"
- "When would you use t-SNE vs PCA?"

---

### Week 8: Feature Engineering & Data Preprocessing
**Core Concept**: Raw data to model-ready features

#### Topics Integrated:
- **Feature Scaling**:
  - Standardization vs normalization (when and why)
  - Min-max, robust scaling, quantile transformation
  - Impact on different algorithms
- **Encoding Categorical Variables**:
  - One-hot, label encoding, target encoding
  - Handling high cardinality
  - Embedding-based approaches (preview)
- **Feature Selection**:
  - Filter methods (correlation, mutual information)
  - Wrapper methods (RFE)
  - Embedded methods (L1 regularization)
  - **Statistical tests**: chi-square, ANOVA
- **Handling Missing Data**:
  - MCAR, MAR, MNAR mechanisms
  - Imputation strategies
  - Multiple imputation

#### Implementation Tasks:
1. Complete preprocessing pipeline (scikit-learn pipeline)
2. Custom transformers
3. Feature selection framework
4. Automated feature engineering (polynomial features, interactions)
5. Handling time-series features

#### Interview Focus:
- "Design a feature engineering pipeline for [specific problem]"
- "How do you handle missing data?"
- "Feature selection for high-dimensional data"
- "Scaling impact on different algorithms"

---

## Month 3: Deep Learning Fundamentals (March 2026)

### Week 9: Neural Networks from Scratch
**Core Concept**: Building blocks of deep learning

#### Topics Integrated:
- **Perceptron to MLP**:
  - Single neuron as logistic regression
  - Universal approximation theorem (intuition)
  - Activation functions: sigmoid, tanh, ReLU (why ReLU works)
  - **Calculus**: derivatives of activations
- **Backpropagation**:
  - Chain rule in computational graphs
  - Forward pass, backward pass derivation
  - Automatic differentiation concepts
  - **Matrix calculus**: Jacobians, vectorization
- **Weight Initialization**:
  - Xavier/Glorot initialization (derivation)
  - He initialization for ReLU
  - Why random initialization matters

#### Implementation Tasks:
1. Neural network from scratch (pure NumPy)
   - Forward propagation
   - Backpropagation
   - Training loop
2. Automatic differentiation engine (simplified)
3. Various activation functions with gradients
4. Visualization: learning dynamics, activations

#### Interview Focus:
- "Derive backpropagation for a 2-layer network"
- "Explain vanishing/exploding gradients"
- "Why ReLU over sigmoid?"
- "Implement forward + backward pass in 30 minutes"

---

### Week 10: Convolutional Neural Networks
**Core Concept**: Spatial hierarchies and translation invariance

#### Topics Integrated:
- **Convolution Operation**:
  - 1D, 2D, 3D convolutions
  - **Signal processing**: convolution theorem, filters
  - Padding, stride, dilation
  - Receptive field calculation
- **CNN Architectures Evolution**:
  - LeNet → AlexNet → VGG → ResNet → EfficientNet
  - Why depth matters: feature hierarchies
  - Residual connections (addressing degradation)
  - **Optimization**: Batch Normalization in CNNs
- **Advanced Concepts**:
  - Pooling (max, average, global)
  - 1x1 convolutions (channel-wise mixing)
  - Depthwise separable convolutions
  - Transposed convolutions (deconvolution)

#### Implementation Tasks:
1. Convolution operation from scratch
2. Simple CNN (LeNet-style) on MNIST
3. ResNet implementation in PyTorch/TensorFlow
4. Feature visualization (activation maximization)
5. Transfer learning on ImageNet pretrained models

#### Interview Focus:
- "Explain 1x1 convolutions"
- "Why do residual connections help?"
- "Calculate output dimensions for a conv layer"
- "Design a CNN for [specific image task]"

---

### Week 11: Training Deep Networks
**Core Concept**: Making deep learning work in practice

#### Topics Integrated:
- **Regularization Techniques**:
  - Dropout (why it works: ensemble perspective)
  - L1/L2 regularization in neural networks
  - Data augmentation (implicit regularization)
  - Early stopping
  - **Bayesian interpretation**: regularization as prior
- **Normalization Layers**:
  - Batch Normalization (full derivation)
  - Layer Normalization (for sequences)
  - Instance Normalization (for style transfer)
  - Group Normalization
  - When to use which?
- **Advanced Optimization**:
  - Learning rate warmup
  - Gradient clipping
  - Mixed precision training
  - Distributed training basics

#### Implementation Tasks:
1. Implement all normalization layers
2. Dropout with inference-time scaling
3. Learning rate scheduler library
4. Data augmentation pipeline
5. Training utilities (checkpointing, logging, early stopping)

#### Interview Focus:
- "Explain Batch Normalization's effect on training"
- "Why does dropout work?"
- "How to debug NaN losses?"
- "Design training strategy for limited data"

---

### Week 12: Sequence Models & RNNs
**Core Concept**: Processing sequential data

#### Topics Integrated:
- **Recurrent Neural Networks**:
  - Unfolding through time
  - Backpropagation Through Time (BPTT)
  - **Calculus**: gradient flow in time
  - Vanishing/exploding gradients (mathematical analysis)
- **LSTM & GRU**:
  - Gating mechanisms derivation
  - Cell state vs hidden state
  - Why gates solve vanishing gradients
  - Forget gate importance
- **Bidirectional RNNs**:
  - Forward and backward passes
  - When to use bidirectional vs unidirectional
- **Sequence-to-Sequence**:
  - Encoder-decoder architecture
  - Teacher forcing

#### Implementation Tasks:
1. Vanilla RNN from scratch
2. LSTM cell implementation
3. GRU cell implementation
4. Bidirectional RNN
5. Character-level language model
6. Seq2seq for simple translation task

#### Interview Focus:
- "Explain LSTM gates in detail"
- "LSTM vs GRU: differences and when to use"
- "Implement LSTM forward pass"
- "Handle variable-length sequences"

---

## Month 4: Modern Architectures & Attention (April 2026)

### Week 13: Attention Mechanisms
**Core Concept**: Selective focus in neural networks

#### Topics Integrated:
- **Attention Fundamentals**:
  - Sequence-to-sequence attention (Bahdanau)
  - Query, Key, Value formulation
  - Attention weights as soft alignment
  - **Information theory**: attention as information routing
- **Scaled Dot-Product Attention**:
  - Why scaling by sqrt(d_k)?
  - Softmax stability
  - Attention score computation
- **Multi-Head Attention**:
  - Multiple representation subspaces
  - Parallel attention computation
  - Head concatenation and projection

#### Implementation Tasks:
1. Bahdanau attention from scratch
2. Scaled dot-product attention
3. Multi-head attention module
4. Attention visualization tools
5. Seq2seq with attention

#### Interview Focus:
- "Derive attention mechanism"
- "Why scale by sqrt(d_k) in attention?"
- "Multi-head attention: why multiple heads?"
- "Implement attention in 20 minutes"

---

### Week 14: Transformers Architecture
**Core Concept**: Attention is all you need

#### Topics Integrated:
- **Transformer Block**:
  - Self-attention vs cross-attention
  - Position-wise feed-forward networks
  - Residual connections and layer normalization
  - **Architecture design**: why this specific structure?
- **Positional Encoding**:
  - Sinusoidal encoding derivation
  - Learned positional embeddings
  - Relative positional encoding
  - **Fourier analysis**: why sinusoids capture position
- **Encoder-Decoder Architecture**:
  - Masked self-attention in decoder
  - Cross-attention between encoder-decoder
  - Autoregressive generation
- **Training Details**:
  - Label smoothing
  - Warmup + decay learning rate schedule
  - Gradient accumulation

#### Implementation Tasks:
1. Full Transformer from scratch (simplified)
2. Positional encoding implementations
3. Masked multi-head attention
4. Transformer for translation task
5. Attention pattern visualization

#### Interview Focus:
- "Walk through Transformer forward pass"
- "Why positional encoding?"
- "Masked attention: why and how?"
- "Transformer vs RNN: pros/cons"

---

### Week 15: Computer Vision Architectures
**Core Concept**: Modern vision models beyond CNNs

#### Topics Integrated:
- **Vision Transformer (ViT)**:
  - Image patches as tokens
  - Patch embedding
  - CLS token for classification
  - Hybrid architectures (CNN + Transformer)
- **Object Detection**:
  - R-CNN family evolution
  - YOLO series (one-stage detectors)
  - Anchor boxes, NMS
  - **Loss functions**: IoU, GIoU, DIoU
- **Segmentation**:
  - Semantic vs instance vs panoptic
  - U-Net architecture
  - Mask R-CNN
  - Transformer-based (DETR)
- **Self-Supervised Vision**:
  - Contrastive learning (SimCLR, MoCo)
  - Masked autoencoders (MAE)

#### Implementation Tasks:
1. Vision Transformer implementation
2. YOLO-style detector (simplified)
3. U-Net for segmentation
4. Contrastive learning framework
5. Transfer learning experiments

#### Interview Focus:
- "Explain Vision Transformer architecture"
- "Object detection: one-stage vs two-stage"
- "Design a segmentation pipeline"
- "Self-supervised learning intuition"

---

### Week 16: Generative Models
**Core Concept**: Learning to generate data

#### Topics Integrated:
- **Autoencoders**:
  - Encoder-decoder structure
  - Latent space representation
  - Denoising autoencoders
  - **Information theory**: bottleneck principle
- **Variational Autoencoders (VAE)**:
  - Probabilistic encoder/decoder
  - KL divergence in loss
  - Reparameterization trick (why needed)
  - **Bayesian inference**: variational inference intuition
- **Generative Adversarial Networks (GAN)**:
  - Minimax game formulation
  - Nash equilibrium
  - Mode collapse problem
  - Training stability techniques
  - **Game theory**: GAN as two-player game
- **Diffusion Models** (Introduction):
  - Forward and reverse diffusion process
  - Score matching
  - Denoising objective

#### Implementation Tasks:
1. Vanilla autoencoder
2. VAE implementation with visualization
3. Simple GAN (DCGAN architecture)
4. Conditional GAN
5. Latent space interpolation experiments

#### Interview Focus:
- "Explain VAE loss function"
- "Why is GAN training unstable?"
- "Compare VAE vs GAN vs Diffusion"
- "Implement reparameterization trick"

---

## Month 5: LLMs & Modern NLP (May 2026)

### Week 17: Language Model Foundations
**Core Concept**: Statistical language modeling to neural LMs

#### Topics Integrated:
- **Language Modeling Task**:
  - Next-token prediction
  - Perplexity metric (from information theory)
  - Causal vs masked language modeling
- **Tokenization**:
  - Byte-Pair Encoding (BPE)
  - WordPiece, SentencePiece
  - Subword regularization
- **Word Embeddings**:
  - Word2Vec (Skip-gram, CBOW)
  - GloVe (matrix factorization perspective)
  - **Linear algebra**: embedding as matrix lookup
  - Contextual embeddings (ELMo preview)
- **Pre-training Objectives**:
  - Masked Language Modeling (BERT)
  - Causal Language Modeling (GPT)
  - Span corruption (T5)

#### Implementation Tasks:
1. BPE tokenizer from scratch
2. Word2Vec implementation
3. Simple language model (LSTM-based)
4. GPT-style transformer decoder
5. BERT-style masked LM

#### Interview Focus:
- "Explain masked vs causal LM"
- "Why subword tokenization?"
- "Word2Vec training objective"
- "Design tokenization for code"

---

### Week 18: Large Language Models (LLMs)
**Core Concept**: Scaling transformers to billions of parameters

#### Topics Integrated:
- **GPT Architecture Evolution**:
  - GPT → GPT-2 → GPT-3 → GPT-4
  - Scaling laws (power law relationship)
  - Emergent abilities
- **BERT and Variants**:
  - BERT, RoBERTa, ALBERT, DeBERTa
  - Pre-training + fine-tuning paradigm
- **Efficient Transformers**:
  - Sparse attention patterns
  - Linear attention approximations
  - Flash Attention (memory efficiency)
  - **Complexity analysis**: O(n²) → O(n) attention
- **Model Compression**:
  - Knowledge distillation
  - Quantization (int8, int4)
  - Pruning strategies
  - Low-rank adaptation (LoRA)

#### Implementation Tasks:
1. GPT-2 from scratch (small version)
2. Fine-tuning pipeline
3. LoRA implementation
4. Knowledge distillation setup
5. Inference optimization (quantization)

#### Interview Focus:
- "Explain GPT architecture in detail"
- "BERT vs GPT: key differences"
- "How does LoRA work?"
- "Scaling laws: what do they tell us?"

---

### Week 19: Advanced LLM Techniques
**Core Concept**: Making LLMs useful and safe

#### Topics Integrated:
- **Instruction Tuning**:
  - Supervised fine-tuning (SFT)
  - Dataset curation
  - Multi-task learning
- **Reinforcement Learning from Human Feedback (RLHF)**:
  - Reward modeling
  - PPO for LLMs (Proximal Policy Optimization)
  - **RL fundamentals**: policy gradient, value functions
  - KL penalty for distribution matching
- **Prompt Engineering**:
  - Few-shot learning
  - Chain-of-thought prompting
  - ReAct (reasoning + acting)
- **Retrieval-Augmented Generation (RAG)**:
  - Dense retrieval (embedding-based)
  - Hybrid search
  - Context injection strategies
- **Model Alignment**:
  - Constitutional AI
  - Debate and critiquing

#### Implementation Tasks:
1. Simple reward model training
2. PPO implementation (conceptual)
3. RAG system from scratch
4. Prompt engineering experimentation framework
5. Evaluation suite for LLMs

#### Interview Focus:
- "Explain RLHF pipeline"
- "Design a RAG system for [specific domain]"
- "Prompt engineering best practices"
- "How to evaluate LLM outputs?"

---

### Week 20: Multimodal Models & Vision-Language
**Core Concept**: Bridging vision and language

#### Topics Integrated:
- **Vision-Language Pre-training**:
  - CLIP (contrastive learning)
  - ALIGN (noisy image-text pairs)
  - **Contrastive learning**: InfoNCE loss
- **Vision-Language Models**:
  - ViLT, BLIP, Flamingo
  - Cross-attention between modalities
- **Text-to-Image Generation**:
  - DALL-E, Stable Diffusion, Imagen
  - Diffusion models deep dive
  - Latent diffusion (efficiency)
  - Classifier-free guidance
- **Video Understanding**:
  - Temporal modeling
  - Video-language models

#### Implementation Tasks:
1. Simple CLIP-style model
2. Image captioning system
3. Stable Diffusion inference pipeline
4. Multimodal retrieval system
5. Visual question answering

#### Interview Focus:
- "Explain CLIP training objective"
- "How does Stable Diffusion work?"
- "Design a multimodal search system"
- "Challenges in video understanding"

---

## Month 6: Reinforcement Learning & Production Systems (June 2026)

### Week 21: Reinforcement Learning Foundations
**Core Concept**: Learning from interaction

#### Topics Integrated:
- **MDP Framework**:
  - States, actions, rewards, transitions
  - Policy, value function, Q-function
  - Bellman equations (derivation)
  - **Dynamic programming**: value iteration, policy iteration
- **Monte Carlo Methods**:
  - Model-free learning
  - Exploration-exploitation tradeoff
  - ε-greedy, UCB
- **Temporal Difference Learning**:
  - TD(0), TD(λ)
  - Q-learning, SARSA
  - Off-policy vs on-policy
  - **Bootstrapping**: why TD works

#### Implementation Tasks:
1. GridWorld environment
2. Value iteration, policy iteration
3. Q-learning implementation
4. SARSA implementation
5. Comparison on simple environments

#### Interview Focus:
- "Explain Bellman equation"
- "Q-learning vs SARSA differences"
- "Exploration-exploitation strategies"
- "When to use RL vs supervised learning?"

---

### Week 22: Deep Reinforcement Learning
**Core Concept**: Combining deep learning with RL

#### Topics Integrated:
- **Deep Q-Networks (DQN)**:
  - Experience replay (why it helps)
  - Target networks
  - Double DQN (addressing overestimation)
  - Prioritized experience replay
- **Policy Gradient Methods**:
  - REINFORCE algorithm
  - Actor-critic methods
  - A3C (asynchronous advantage actor-critic)
  - **Variance reduction**: baselines
- **Advanced Algorithms**:
  - PPO (trust region methods)
  - TRPO (constraint optimization)
  - SAC (maximum entropy RL)
  - DDPG (continuous actions)
- **Multi-Agent RL** (brief):
  - Cooperative vs competitive
  - Nash equilibria

#### Implementation Tasks:
1. DQN for Atari games
2. REINFORCE algorithm
3. A2C/A3C implementation
4. PPO from scratch
5. Continuous control with DDPG

#### Interview Focus:
- "Explain DQN improvements over Q-learning"
- "Policy gradient theorem derivation"
- "Why does PPO work?"
- "Design RL system for [specific problem]"

---

### Week 23: ML Systems Design & Production
**Core Concept**: Deploying models at scale

#### Topics Integrated:
- **System Design Principles**:
  - Data pipeline architecture
  - Model serving strategies
  - Latency vs throughput tradeoffs
  - **Distributed systems**: sharding, replication
- **Model Serving**:
  - Batch vs online inference
  - Model optimization (TensorRT, ONNX)
  - A/B testing infrastructure
  - Caching strategies
- **MLOps**:
  - Experiment tracking (MLflow, Weights & Biases)
  - Model versioning
  - Feature stores
  - Monitoring and alerting
- **Scalability**:
  - Data parallelism
  - Model parallelism
  - Pipeline parallelism
  - Distributed training (DDP, FSDP)

#### Implementation Tasks:
1. Model serving API (FastAPI)
2. Batch inference pipeline
3. A/B testing framework
4. Monitoring dashboard
5. End-to-end ML pipeline

#### Interview Focus:
- "Design Instagram's recommendation system"
- "How to handle model drift?"
- "Reduce inference latency by 10x"
- "Design distributed training system"

---

### Week 24: Advanced Topics & Interview Prep
**Core Concept**: Cutting-edge techniques and interview mastery

#### Topics Integrated:
- **Graph Neural Networks**:
  - Message passing framework
  - GCN, GAT, GraphSAGE
  - Applications: social networks, molecules
- **Causal Inference**:
  - Causal graphs
  - Intervention vs observation
  - Causal ML applications
- **Federated Learning**:
  - Privacy-preserving ML
  - Distributed optimization challenges
- **Neural Architecture Search**:
  - AutoML overview
  - DARTS, EfficientNet design
- **Interpretability**:
  - LIME, SHAP
  - Attention visualization
  - Feature importance

#### Implementation Tasks:
1. Simple GNN implementation
2. SHAP values calculation
3. NAS experiment (simplified)
4. Federated learning simulation
5. Model interpretation tools

#### Interview Focus:
- "Explain your most complex ML project"
- "Debug a failing ML system"
- "Design ML solution for [novel problem]"
- "Explain tradeoffs in [any topic]"

---