# Multimodal-Emotion-Recognition
# Multimodal Emotion Recognition Using Speech and Text

## Overview
This project implements a **Multimodal Emotion Recognition System** using:
- Speech-based emotion recognition
- Text-based emotion recognition
- Multimodal fusion of speech and text

The project is built on the **TESS (Toronto Emotional Speech Set)** dataset and explores how acoustic and semantic information contribute to emotion classification.

The system compares the performance of:
1. Speech-only models
2. Text-only models
3. Fusion-based multimodal models

The experiments demonstrate that speech signals contain significantly stronger emotional cues than textual transcriptions in the TESS dataset.

---

## Project Structure

```bash
models/
│
├── fusion_pipeline/
│   ├── Train.ipynb
│   └── Test.ipynb
│
├── speech_pipeline/
│   ├── Train.ipynb
│   └── Test.ipynb
│
└── text_pipeline/
    ├── Train.ipynb
    └── Test.ipynb
```

---

## Features

### Speech Emotion Recognition
- Uses **HuBERT embeddings** for feature extraction
- Uses **BiLSTM** for temporal modeling
- Uses **Attention Mechanism** to focus on emotionally significant speech frames
- Achieves very high classification accuracy on the TESS dataset

### Text Emotion Recognition
- Converts speech to text using **HuBERT ASR**
- Uses embedding layers and **BiLSTM** for contextual learning
- Demonstrates limitations of text-only emotion recognition on TESS

### Multimodal Fusion
- Combines speech and text representations
- Uses:
  - Cross-attention
  - BERT text embeddings
  - HuBERT speech embeddings
- Improves robustness compared to text-only classification

---

## Dataset

### TESS Dataset
The Toronto Emotional Speech Set (TESS) contains emotional speech recordings with:
- 7 emotion classes:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Neutral
  - Pleasant Surprise
  - Sad

Each sample follows the sentence pattern:

```text
“Say the word ___”
```

Because the semantic content is limited, the dataset is highly suitable for evaluating acoustic emotion recognition.

---

## Model Architectures

### 1. Speech Model

Pipeline:

```text
Audio → HuBERT → BiLSTM → Attention → Softmax Classifier
```

Key Components:
- HuBERT feature extraction
- Bidirectional LSTM
- Temporal Attention
- Softmax classifier

---

### 2. Text Model

Pipeline:

```text
Speech → HuBERT ASR → Embedding Layer → BiLSTM → Classifier
```

Key Components:
- Automatic speech transcription
- Word embeddings
- Contextual sequence learning

---

### 3. Fusion Model

Pipeline:

```text
Speech Features + Text Features
        ↓
Cross Attention
        ↓
Fusion Layer
        ↓
MLP Classifier
```

Key Components:
- HuBERT speech embeddings
- BERT text embeddings
- Cross-attention fusion
- Multimodal classifier

---

## Technologies Used

### Deep Learning Frameworks
- PyTorch
- Transformers (Hugging Face)
- TensorFlow/Keras

### Models
- HuBERT
- BERT
- BiLSTM
- Multihead Attention

### Libraries
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Librosa

---

## Installation

### Clone the Repository

```bash
git clone <repository-url>
cd multimodal-emotion-recognition
```

### Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install librosa
pip install scikit-learn
pip install matplotlib pandas numpy
```

---

## Running the Project

### Speech Model

```bash
speech_pipeline/IIITH_Voice_Train.ipynb
speech_pipeline/IIITH_Voice_Test.ipynb
```

### Text Model

```bash
text_pipeline/IIITH_Text_Train.ipynb
text_pipeline/IIITH_Text_Test.ipynb
```

### Fusion Model

```bash
fusion_pipeline/IIITH_Multi_Train.ipynb
fusion_pipeline/IIITH_Multi_Test.ipynb
```

---

## Results

### Speech Model
- Achieved near-perfect emotion classification
- Strong acoustic separability across emotions
- Best performing model

### Text Model
- Performed poorly due to limited semantic information
- Most predictions collapsed into a single class

### Fusion Model
- Performed better than the text-only model
- Underperformed compared to speech-only model
- Demonstrated benefits of multimodal learning when modalities contain meaningful information

---

## Key Findings

### Why Speech Performed Best
Speech signals contain:
- Pitch variations
- Loudness changes
- Temporal rhythm
- Prosodic patterns

These acoustic features strongly encode emotional information.

### Why Text Performed Poorly
The TESS dataset has:
- Fixed sentence structure
- Limited semantic diversity
- Weak contextual emotional information

Therefore, text embeddings failed to learn meaningful emotion separability.

### When Fusion Helps
Fusion becomes more useful when:
- Text contains richer contextual meaning
- Speech is ambiguous
- Both modalities contribute complementary emotional information

---

## Future Improvements

Possible enhancements include:
- Speaker-independent training
- Larger conversational datasets
- Transformer-based fusion architectures
- Audio augmentation
- Real-time emotion recognition
- Multilingual support

---

## Conclusion

This project demonstrates that:
- Speech is the strongest modality for emotion recognition in TESS
- Text contributes limited emotional information
- Multimodal fusion improves robustness only when both modalities are informative

The work highlights the importance of acoustic feature learning in speech emotion recognition systems.

---
