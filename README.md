# Multimodal Misinformation Detection

A multimodal deep learning system for binary misinformation classification using textual and visual information.

## Architecture
```
Text → DistilBERT → Text Embedding
```
```
Image → ResNet-18 → Image Embedding
                  ↓
           Feature Fusion
                  ↓
              MLP Classifier
                  ↓
         Real / Fake + Confidence
```

## Core Technologies

- Python
- PyTorch
- Torchvision
- Hugging Face Transformers
- DistilBERT
- ResNet-18
- Computer Vision
- NLP
- Multimodal Feature Fusion
- Scikit-learn

## Project Status

Environment and repository initialization complete.

## Reproducibility

Experiments use a fixed random seed and centralized configuration.

## License

To be added.