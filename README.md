This is a professional, technical README designed for a GitHub repository or a
final project submission. It highlights the sophisticated use of Deep Speech
Embeddings and the high clinical accuracy you achieved.

Parkinson’s Disease Detection via Deep Speech Embeddings

This project implements a machine learning pipeline to detect Parkinson’s
Disease (PD) from short voice recordings. By leveraging OpenAI's Whisper (Small)
Transformer model for feature extraction and a Support Vector Machine (SVM) for
classification, the system achieves a clinical diagnostic accuracy of 95%.

🚀 Key Highlights

  - Feature Extraction: Utilizes the hidden states of a pre-trained Transformer
    (Whisper) rather than traditional acoustic features (Jitter/Shimmer).
  - Performance:
      - Segment-Level Accuracy: 73% (individual 2-second clips).
      - Speaker-Level Accuracy: 95% (aggregated diagnosis).
      - Sensitivity (Recall): 100% (Zero False Negatives for PD patients).
  - Methodology: Implements a speaker-independent split to ensure the model
    generalizes to new voices.

🛠️ Methodology

1. Data Pre-processing

  - Resampling: All audio files resampled to 16,000 Hz.
  - Padding: Recordings were padded to 30 seconds using whisper.pad_or_trim to
    satisfy the Transformer's fixed-length input requirements.
  - Spectrograms: Converted raw audio into Log-Mel Spectrograms (80 frequency
    bins).

2. Feature Extraction (The Whisper "Small" Encoder)

Instead of manual feature engineering, we utilized Transfer Learning:

  - Passed the 30s spectrograms through the Whisper-Small Encoder.
  - Extracted the Hidden States from the final layer, resulting in a tensor of
    (1, 1500, 768).
  - Pooling: Applied Mean/Max pooling across the time dimension to produce a
    single 768-dimensional vector per audio segment.

3. Classification Pipeline

  - Standardization: Features were scaled using StandardScaler to ensure zero
    mean and unit variance.
  - Classifier: A Linear SVM was trained on the 768-dimensional embeddings.
  - Aggregation (Majority Voting): To move from segment-level to clinical
    diagnosis, we grouped predictions by speaker_id. The final diagnosis is
    determined by the majority prediction across all segments for a specific
    speaker.

📊 Results

Segment-Level (Individual 2s Clips)

| Metric   | Score |
| :------- | :---- |
| Accuracy | 73%   |
| F1-Score | 0.73  |

Speaker-Level (Final Clinical Diagnosis)

| Metric                   | Score    |
| :----------------------- | :------- |
| **Accuracy**             | **95%**  |
| **Sensitivity (Recall)** | **100%** |
| **Specificity**          | **90%**  |

Conclusion: The model is exceptionally robust at identifying PD patients, with
a 0% False Negative rate, making it an ideal tool for clinical screening.

📁 Project Structure

  - feature_extraction.ipynb: Notebook for audio processing and Whisper
    embedding extraction.
  - classification.ipynb: Notebook for training the SVM and performing
    speaker-level evaluation.
  - parkinsons_whisper_features.csv: The generated dataset containing 768-dim
    features for all subjects.
  - models/: Saved parkinsons_svm_model.pkl and scaler.pkl.

💻 Installation & Usage

Prerequisites

pip install openai-whisper librosa torch scikit-learn pandas numpy

Quick Run

1.  Extract Features: Run the extraction loop to convert .wav files into the
    1x768 feature CSV.
2.  Train Model:
    from sklearn.svm import SVC
    model = SVC(kernel='linear')
    model.fit(X_train_scaled, y_train)
3.  Aggregate: Group results by speaker_id to achieve 95% accuracy.

📝 Acknowledgments

  - OpenAI Whisper: For providing the pre-trained speech transformer.
  - Scikit-Learn: For the classification and evaluation framework.

Author: [Your Name]
License: MIT
Status: Completed - 95% Accuracy Achieved.
