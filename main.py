import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import threading
import queue
import pyttsx3
from time import time
from collections import defaultdict

class DetectionTracker:
    def __init__(self, persistence_time=0.3, iou_threshold=0.5):
        self.detections = {}
        self.persistence_time = persistence_time
        self.iou_threshold = iou_threshold
        
    def calculate_iou(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection coordinates
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        # Calculate areas
        intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0
    
    def smooth_bbox(self, old_bbox, new_bbox, alpha=0.7):
        """Smooth bounding box transitions"""
        x1, y1, w1, h1 = old_bbox
        x2, y2, w2, h2 = new_bbox
        return (
            int(alpha * x1 + (1 - alpha) * x2),
            int(alpha * y1 + (1 - alpha) * y2),
            int(alpha * w1 + (1 - alpha) * w2),
            int(alpha * h1 + (1 - alpha) * h2)
        )
    
    def update(self, new_detections):
        current_time = time()
        updated_detections = {}
        
        # Update existing detections
        for detection in new_detections:
            class_name = detection['class_name']
            new_bbox = detection['bbox']
            
            # Skip "no stopping" signs
            if "no stopping" in class_name.lower():
                continue
                
            if class_name in self.detections:
                old_data = self.detections[class_name]
                old_bbox = old_data['bbox']
                
                # Calculate IoU between old and new bounding boxes
                iou = self.calculate_iou(old_bbox, new_bbox)
                
                if iou > self.iou_threshold:
                    # Smooth the transition of bounding box
                    smoothed_bbox = self.smooth_bbox(old_bbox, new_bbox)
                    updated_detections[class_name] = {
                        'bbox': smoothed_bbox,
                        'time': current_time,
                        'probability': detection['probability']
                    }
                else:
                    # New position, reset smoothing
                    updated_detections[class_name] = {
                        'bbox': new_bbox,
                        'time': current_time,
                        'probability': detection['probability']
                    }
            else:
                # New detection
                updated_detections[class_name] = {
                    'bbox': new_bbox,
                    'time': current_time,
                    'probability': detection['probability']
                }
        
        # Keep recent detections that weren't updated
        for class_name, data in self.detections.items():
            if (current_time - data['time'] < self.persistence_time and 
                class_name not in updated_detections):
                updated_detections[class_name] = data
        
        self.detections = updated_detections
        
        return self.get_current_detections()
    
    def get_current_detections(self):
        return [
            {
                'class_name': class_name,
                'bbox': data['bbox'],
                'probability': data['probability']
            }
            for class_name, data in self.detections.items()
        ]

class AudioFeedback:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.current_speech = None
        self.speech_queue = queue.Queue(maxsize=1)
        self.running = True
        self.speech_thread = threading.Thread(target=self._speech_worker)
        self.speech_thread.daemon = True
        self.speech_thread.start()
    
    def _speech_worker(self):
        while self.running:
            try:
                text = self.speech_queue.get(timeout=0.1)
                if text != self.current_speech:
                    self.current_speech = text
                    self.engine.stop()
                    self.engine.say(text)
                    self.engine.runAndWait()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Speech error: {e}")
                continue
    
    def speak(self, text):
        # Don't speak for "no stopping" signs
        if "no stopping" in text.lower():
            return
            
        try:
            while not self.speech_queue.empty():
                self.speech_queue.get_nowait()
            self.speech_queue.put(text)
        except queue.Full:
            pass
    
    def cleanup(self):
        self.running = False
        try:
            self.engine.stop()
        except:
            pass
        self.speech_thread.join(timeout=1)

class FrameBuffer:
    def __init__(self, maxsize=2):
        self.frame_queue = queue.Queue(maxsize=maxsize)
    
    def put_frame(self, frame):
        try:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
            self.frame_queue.put(frame)
        except queue.Full:
            pass
    
    def get_frame(self):
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 100))
    img = cv2.equalizeHist(img)
    img = img/255.0
    return img

class SignDetector:
    def __init__(self, model_path, labels_path):
        self.model = tf.keras.models.load_model(model_path)
        labels_df = pd.read_csv(labels_path)
        self.class_names = labels_df['Name'].values
        self.frame_buffer = FrameBuffer()
        self.result_queue = queue.Queue(maxsize=1)
        self.running = True
        self.audio_feedback = AudioFeedback()
        self.last_detection_time = 0
        self.detection_cooldown = 0.1
        self.detection_tracker = DetectionTracker(persistence_time=0.3)
        
    def process_frame(self, frame):
        self.frame_buffer.put_frame(frame)

    def inference_worker(self):
        while self.running:
            frame = self.frame_buffer.get_frame()
            if frame is None:
                continue

            current_time = time()
            if current_time - self.last_detection_time < self.detection_cooldown:
                continue

            self.last_detection_time = current_time
            
            try:
                results = []
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)
                
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                highest_prob = 0
                best_detection = None
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 1000:
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = float(w)/h
                        
                        if 0.8 <= aspect_ratio <= 1.2:
                            roi = frame[y:y+h, x:x+w]
                            if roi.size == 0:
                                continue
                            
                            try:
                                preprocessed_roi = preprocessing(roi)
                                preprocessed_roi = preprocessed_roi.reshape(1, 100, 100, 1)
                                
                                prediction = self.model.predict(preprocessed_roi, verbose=0)
                                class_index = np.argmax(prediction)
                                probability = np.max(prediction)
                                
                                if probability > 0.95:
                                    class_name = self.class_names[class_index] if class_index < len(self.class_names) else str(class_index)
                                    
                                    # Skip processing "no stopping" signs
                                    if "no stopping" in class_name.lower():
                                        continue
                                        
                                    results.append({
                                        'bbox': (x, y, w, h),
                                        'class_name': class_name,
                                        'probability': probability
                                    })
                                    
                                    if probability > highest_prob:
                                        highest_prob = probability
                                        best_detection = class_name
                            except Exception as e:
                                continue
                
                # Update tracker with new detections
                results = self.detection_tracker.update(results)
                
                if best_detection and "no stopping" not in best_detection.lower():
                    self.audio_feedback.speak(f"Detected {best_detection}")
                
                while not self.result_queue.empty():
                    self.result_queue.get_nowait()
                self.result_queue.put(results)
                
            except Exception as e:
                print(f"Error in inference worker: {e}")

    def get_latest_results(self):
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return []

    def cleanup(self):
        self.running = False
        self.audio_feedback.cleanup()

def draw_detection(frame, detection):
    x, y, w, h = detection['bbox']
    class_name = detection['class_name']
    probability = detection['probability']
    
    # Draw semi-transparent box
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Create label with higher contrast background
    label = f"{class_name} ({probability:.2f})"
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    
    # Draw background rectangle for text with better contrast
    cv2.rectangle(overlay, 
                (x, y - text_size[1] - 10), 
                (x + text_size[0], y), 
                (0, 255, 0), 
                -1)
    
    # Blend the overlay with the original frame
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw text last so it's clear
    cv2.putText(frame, 
              label, 
              (x, y - 5),
              cv2.FONT_HERSHEY_SIMPLEX, 
              0.5, 
              (0, 0, 0),
              2)
    
    return frame

def main():
    detector = SignDetector('my_traffic_sign_model.h5', 'labels.csv')
    
    inference_thread = threading.Thread(target=detector.inference_worker)
    inference_thread.daemon = True
    inference_thread.start()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera")
        detector.cleanup()
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            detector.process_frame(frame.copy())
            
            # Get and display results
            results = detector.get_latest_results()
            for result in results:
                frame = draw_detection(frame, result)
            
            cv2.imshow('Traffic Sign Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        detector.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        sys.exit(0)