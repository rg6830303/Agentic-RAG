# Neural Networks and Deep Learning

A neural network is a function approximator built from layers of differentiable units. Deep learning is the practice of training such networks with many layers using gradient-based optimization on large datasets.

## Core architectures
- **Multi-layer perceptron (MLP)**: stacked fully connected layers with nonlinear activations (ReLU, GELU).
- **Convolutional neural network (CNN)**: weight-shared convolutional filters that exploit spatial locality, dominant in image processing until vision transformers emerged.
- **Recurrent neural network (RNN), LSTM, GRU**: process sequences token-by-token with hidden state. Largely superseded by Transformers for language tasks.
- **Transformer**: stacks of self-attention and feed-forward blocks. Dominant for language, increasingly used in vision (ViT) and audio.
- **Diffusion model**: trained to gradually denoise; used in image (Stable Diffusion, Imagen, DALL·E) and video generation (Sora, Veo).
- **Mixture of Experts (MoE)**: sparsely routes each token through a small subset of expert subnetworks, increasing parameter count without proportionally increasing compute. Used in models such as Mixtral and many frontier LLMs.

## Optimization
- **Stochastic gradient descent (SGD)** with momentum was the early workhorse.
- **Adam** and **AdamW** adapt per-parameter learning rates and remain the standard for LLM pretraining.
- **Learning rate schedules**: warmup followed by cosine or inverse-square-root decay.
- **Mixed precision training** uses bfloat16 or float16 to speed up GPU/TPU computation while keeping a master fp32 copy of weights.

## Regularization
- Dropout, weight decay, label smoothing, early stopping.
- Data augmentation in vision (random crops, flips) and language (token masking, instruction variation).

## Hardware
- GPUs: NVIDIA H100, H200, B100/B200; AMD MI300; consumer NVIDIA RTX cards.
- TPUs: Google's custom accelerators (v4, v5, v5e, v5p, Trillium).
- Specialized: AWS Trainium/Inferentia, Cerebras, Groq LPU for low-latency inference.
