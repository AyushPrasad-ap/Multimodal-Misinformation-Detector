# Multimodal Misinformation Detector — Architecture

```mermaid
flowchart LR
    A[Text / Claim] --> B[DistilBERT]
    C[Associated Image] --> D[ResNet-18]

    B --> E[768-D Text Embedding]
    D --> F[512-D Image Embedding]

    E --> G[Feature Fusion<br/>1280-D]
    F --> G

    G --> H[MLP<br/>1280 → 256 → 2]
    H --> I[Softmax]

    I --> J[REAL / FAKE]
    I --> K[Confidence / Probabilities]