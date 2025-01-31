import cv2
import numpy as np
import tensorflow as tf
import joblib
from typing import List, Dict


class ActionPredictor:
    def __init__(self, i3d_model_path: str, label_map_path: str):
        self.i3d_model = self._load_i3d_model(i3d_model_path)
        self.labels = self._load_labels(label_map_path)

    def _load_i3d_model(self, model_dir: str) -> tf.keras.Model:
        """Load I3D model with proper configuration."""
        return tf.compat.v1.saved_model.load_v2(model_dir, tags=['train'])

    def _load_labels(self, label_file: str) -> List[str]:
        """Load action labels from file."""
        with open(label_file, "r") as f:
            return [line.strip() for line in f.readlines()]

    def preprocess_i3d_frames(self, frames: List[np.ndarray], frame_size) -> tf.Tensor:
        """Preprocess frame sequence for I3D model."""
        processed_frames = []
        for frame in frames:
            frame = cv2.resize(frame, frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frames.append(frame / 255.0)
        return tf.convert_to_tensor(processed_frames, dtype=tf.float32)

    def predict_i3d(self, frames: List[np.ndarray], frame_size, max_frames: int) -> Dict[str, float]:
        """Predict action using I3D model."""
        if len(frames) < max_frames:
            frames.extend([frames[-1]] * (max_frames - len(frames)))
        
        preprocessed = self.preprocess_i3d_frames(frames[:max_frames], frame_size)
        input_tensor = tf.expand_dims(preprocessed, axis=0)
        
        try:
            signature_key = list(self.i3d_model.signatures.keys())[0]
            predictions = self.i3d_model.signatures[signature_key](input_tensor)
            logits = predictions['default'].numpy()[0]
            probabilities = tf.nn.softmax(logits).numpy()
            
            top_indices = np.argsort(probabilities)[-5:][::-1]
            return {
                self.labels[idx]: float(probabilities[idx])
                for idx in top_indices
            }
        except Exception as e:
            print(f"I3D prediction error: {str(e)}")
            return {}