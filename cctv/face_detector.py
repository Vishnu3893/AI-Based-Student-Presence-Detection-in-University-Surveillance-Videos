"""
Optimized Face Detection using OpenCV DNN
Uses pre-trained Caffe model for fast face detection
"""
import cv2
import numpy as np
import os
import pickle
import face_recognition

class FaceDetector:
    def __init__(self, photos_folder, cache_file="face_encodings.pkl"):
        self.photos_folder = photos_folder
        self.cache_file = os.path.join(os.path.dirname(photos_folder), cache_file)
        
        # Load OpenCV DNN face detector (Caffe model)
        # Using the built-in Haar cascade as fallback, but DNN is better
        self.use_dnn = False
        self.net = None
        
        # Try to use DNN model if available
        model_path = os.path.join(os.path.dirname(__file__), "models")
        prototxt = os.path.join(model_path, "deploy.prototxt")
        caffemodel = os.path.join(model_path, "res10_300x300_ssd_iter_140000.caffemodel")
        
        if os.path.exists(prototxt) and os.path.exists(caffemodel):
            self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
            self.use_dnn = True
            print("DEBUG: Using OpenCV DNN face detector (faster)")
        else:
            print("DEBUG: DNN model not found, using HOG detector")
        
        # Load or compute face encodings
        self.known_encodings = []
        self.known_regnos = []
        self._load_encodings()
    
    def _load_encodings(self):
        """Load face encodings from cache or compute them"""
        # Try to load from cache
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_encodings = data['encodings']
                    self.known_regnos = data['regnos']
                    print(f"DEBUG: Loaded {len(self.known_encodings)} face encodings from cache")
                    return
            except Exception as e:
                print(f"DEBUG: Error loading cache: {e}")
        
        # Compute encodings from photos
        self._compute_encodings()
    
    def _compute_encodings(self):
        """Compute face encodings from student photos"""
        print("DEBUG: Computing face encodings from photos...")
        self.known_encodings = []
        self.known_regnos = []
        
        if not os.path.exists(self.photos_folder):
            print(f"DEBUG: Photos folder not found: {self.photos_folder}")
            return
        
        files = [f for f in os.listdir(self.photos_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total = len(files)
        
        for i, filename in enumerate(files):
            try:
                filepath = os.path.join(self.photos_folder, filename)
                regno = filename.split("_")[0]
                
                # Load and encode face
                image = face_recognition.load_image_file(filepath)
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    self.known_encodings.append(encodings[0])
                    self.known_regnos.append(regno)
                
                if (i + 1) % 10 == 0:
                    print(f"DEBUG: Processed {i + 1}/{total} photos...")
                    
            except Exception as e:
                print(f"DEBUG: Error processing {filename}: {e}")
        
        # Save to cache
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump({
                    'encodings': self.known_encodings,
                    'regnos': self.known_regnos
                }, f)
            print(f"DEBUG: Saved {len(self.known_encodings)} encodings to cache")
        except Exception as e:
            print(f"DEBUG: Error saving cache: {e}")
    
    def reload_encodings(self):
        """Force reload encodings (call after adding new photos)"""
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        self._compute_encodings()
    
    def detect_faces_dnn(self, frame, confidence_threshold=0.5):
        """Detect faces using OpenCV DNN (fast)"""
        if not self.use_dnn or self.net is None:
            return self.detect_faces_hog(frame)
        
        h, w = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
            1.0, (300, 300), 
            (104.0, 177.0, 123.0)
        )
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                # Convert to top, right, bottom, left format
                faces.append((max(0, y1), min(w, x2), min(h, y2), max(0, x1)))
        
        return faces
    
    def detect_faces_hog(self, frame):
        """Detect faces using HOG (fallback)"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return face_recognition.face_locations(rgb, model='hog')
    
    def detect_and_recognize(self, frame, threshold=0.5):
        """Detect faces and recognize them"""
        results = []
        
        # Detect faces
        if self.use_dnn:
            face_locations = self.detect_faces_dnn(frame)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb, model='hog')
        
        if not face_locations:
            return results
        
        # Get face encodings
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)
        
        # Match faces
        for face_encoding, face_location in zip(face_encodings, face_locations):
            top, right, bottom, left = face_location
            
            reg_no = "UNKNOWN"
            confidence = 0.0
            
            if len(self.known_encodings) > 0:
                # Calculate distances to all known faces
                distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                best_idx = np.argmin(distances)
                best_distance = distances[best_idx]
                
                if best_distance <= threshold:
                    reg_no = self.known_regnos[best_idx]
                    confidence = 1.0 - best_distance
            
            results.append({
                'location': (top, right, bottom, left),
                'reg_no': reg_no,
                'confidence': confidence
            })
        
        return results


# Download DNN model files if needed
def download_dnn_models():
    """Download OpenCV DNN face detection models"""
    import urllib.request
    
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    caffemodel_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    
    prototxt_path = os.path.join(model_dir, "deploy.prototxt")
    caffemodel_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    
    if not os.path.exists(prototxt_path):
        print("Downloading deploy.prototxt...")
        urllib.request.urlretrieve(prototxt_url, prototxt_path)
    
    if not os.path.exists(caffemodel_path):
        print("Downloading caffemodel (10MB)...")
        urllib.request.urlretrieve(caffemodel_url, caffemodel_path)
    
    print("DNN models ready!")


if __name__ == "__main__":
    download_dnn_models()
