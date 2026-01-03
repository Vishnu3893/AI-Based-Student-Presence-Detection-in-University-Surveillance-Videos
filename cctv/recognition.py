import os, json, math, cv2, numpy as np
from pathlib import Path

# optional face_recognition import
try:
    import face_recognition
    HAS_FACE_REC = True
except Exception:
    HAS_FACE_REC = False

class RecognitionEngine:
    def __init__(self, photos_dir, cache_dir, gallery_dir):
        self.photos_dir = photos_dir
        self.cache_dir = Path(cache_dir)
        self.gallery_dir = Path(gallery_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.gallery_dir.mkdir(parents=True, exist_ok=True)

        self.FRAME_SKIP = 2
        self.MIN_TRACK_LENGTH = 8
        self.GEI_SIZE = (64,64)
        self.SIM_THRESHOLD = 0.95

        # face resources
        self.known_encodings = []
        self.known_ids = []
        self.lbph = None
        self.lbph_label_map = {}

        self.reload_faces()

    # ---- Face gallery (reload after student photo uploads) ----
    def reload_faces(self):
        self.known_encodings = []
        self.known_ids = []
        files = [f for f in os.listdir(self.photos_dir) if f.lower().endswith((".jpg",".png",".jpeg"))]
        if HAS_FACE_REC:
            for f in files:
                try:
                    path = os.path.join(self.photos_dir, f)
                    img = face_recognition.load_image_file(path)
                    encs = face_recognition.face_encodings(img)
                    if encs:
                        self.known_encodings.append(encs[0])
                        self.known_ids.append(os.path.splitext(f)[0])
                except Exception:
                    continue
        else:
            # LBPH fallback training
            faces = []
            labels = []
            label_map = {}
            next_label = 0
            for f in files:
                reg = os.path.splitext(f)[0]
                path = os.path.join(self.photos_dir, f)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                img = cv2.resize(img, (200,200))
                if reg not in label_map:
                    label_map[reg] = next_label; next_label += 1
                faces.append(img); labels.append(label_map[reg])
            if faces:
                try:
                    self.lbph = cv2.face.LBPHFaceRecognizer_create()
                    self.lbph.train(faces, np.array(labels))
                    self.lbph_label_map = {v:k for k,v in label_map.items()}
                except Exception:
                    self.lbph = None

    # ---- Frame reading helper ----
    def _read_frames(self, video_path, sampling=2):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        idx = 0
        ok = True
        while ok:
            ok, frame = cap.read()
            if not ok: break
            if idx % sampling == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append((idx, gray.copy()))
            idx += 1
        cap.release()
        return frames, fps

    def _bbox_iou(self, b1, b2):
        x1,y1,w1,h1 = b1; x2,y2,w2,h2 = b2
        xa, ya = max(x1,x2), max(y1,y2)
        xb, yb = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
        iw, ih = max(0, xb-xa), max(0, yb-ya)
        inter = iw*ih
        union = w1*h1 + w2*h2 - inter
        return inter/union if union>0 else 0.0

    # ---- Extract tracks by background subtraction + simple IOU linking ----
    def _extract_tracks(self, video_path):
        frames, fps = self._read_frames(video_path, self.FRAME_SKIP)
        if not frames: return [], fps, 0, 0
        backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
        detections = []
        fw=fh=0
        for (fidx, gray) in frames:
            fg = backSub.apply(gray)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
            _, th = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            h, w = th.shape; fw, fh = w, h
            dets = []
            for c in contours:
                if cv2.contourArea(c) < 400: continue
                x,y,ww,hh = cv2.boundingRect(c)
                dets.append((fidx, (x,y,ww,hh), th[y:y+hh, x:x+ww]))
            detections.append(dets)

        # link detections across frames by IOU
        tracks = []
        for t_i, dets in enumerate(detections):
            if t_i == 0:
                for d in dets: tracks.append([d])
                continue
            for d in dets:
                fidx, bbox, mask = d
                best=None; best_iou=0.0
                for tr in tracks:
                    lf, lb, _ = tr[-1]
                    if fidx <= lf: continue
                    iou = self._bbox_iou(lb, bbox)
                    if iou > best_iou:
                        best_iou = iou; best = tr
                if best is not None and best_iou > 0.15:
                    best.append(d)
                else:
                    tracks.append([d])
        good = [tr for tr in tracks if len(tr) >= self.MIN_TRACK_LENGTH]
        return good, fps, fw, fh

    def _compute_gei(self, masks):
        arr = []
        for m in masks:
            try:
                r = cv2.resize(m, self.GEI_SIZE, interpolation=cv2.INTER_NEAREST)
                arr.append((r>127).astype(np.float32))
            except Exception:
                continue
        if not arr: return None
        gei = np.mean(np.stack(arr, axis=0), axis=0)
        v = gei.flatten().astype(np.float32)
        n = np.linalg.norm(v)
        return v / (n + 1e-8)

    def _load_gallery(self):
        gallery = {}
        for p in Path(self.gallery_dir).glob("*.npy"):
            try:
                vec = np.load(p).astype(np.float32)
                n = np.linalg.norm(vec)
                if n>0: gallery[p.stem] = vec / n
            except Exception:
                continue
        return gallery

    def _cosine(self, a, b):
        num = float(np.dot(a,b))
        den = float(np.linalg.norm(a)*np.linalg.norm(b)) + 1e-8
        return num/den

    # ---- Public: prepare gait for a video and cache tracks ----
    def prepare_gait_for_video(self, video_filename):
        cache = self._cache_path(video_filename)
        if cache.exists(): return
        video_path = os.path.join("uploads", video_filename)
        if not os.path.exists(video_path): return
        tracks, fps, fw, fh = self._extract_tracks(video_path)
        gallery = self._load_gallery()
        export = []
        for tr in tracks:
            times=[]; bboxes=[]; masks=[]
            for (fidx, bbox, mask) in tr:
                times.append(fidx/(fps or 25.0))
                bboxes.append(bbox); masks.append(mask)
            gei = self._compute_gei(masks)
            best = "UNKNOWN"; score = 0.0
            if gei is not None and gallery:
                for reg, gvec in gallery.items():
                    s = self._cosine(gei, gvec)
                    if s > score:
                        score = s; best = reg
            mid = len(bboxes)//2
            nx,ny,nw,nh = self._bbox_norm(bboxes[mid], fw, fh)
            export.append({
                "t": round(times[mid], 2),
                "x": nx, "y": ny, "w": nw, "h": nh,
                "reg_no": best if score >= self.SIM_THRESHOLD else "UNKNOWN",
                "score": round(float(score), 3)
            })
        with open(cache, "w") as f:
            json.dump(export, f)

    def get_tracks_for_video(self, video_filename):
        cache = self._cache_path(video_filename)
        if not cache.exists():
            demo = [{"t": i/10.0, "x": 0.15+0.4*math.sin(i/15.0), "y":0.3, "w":0.16, "h":0.32,
                     "reg_no":"UNKNOWN","score":0.0} for i in range(120)]
            with open(cache, "w") as f: json.dump(demo, f)
        with open(cache, "r") as f:
            return json.load(f)

    def create_gallery_gei_from_video(self, video_path, reg_no):
        # extract tracks, choose the longest (assumed labeled subject), compute GEI and save
        tracks, fps, fw, fh = self._extract_tracks(video_path)
        if not tracks: return False
        longest = max(tracks, key=lambda t: len(t))
        masks = [m for (_,_,m) in longest]
        vec = self._compute_gei(masks)
        if vec is None: return False
        np.save(Path(self.gallery_dir)/f"{reg_no}.npy", vec)
        return True

    def _cache_path(self, filename):
        return self.cache_dir / (Path(filename).stem + ".json")

    def _bbox_norm(self, bbox, fw, fh):
        x,y,w,h = bbox
        if fw==0 or fh==0: return (0.0,0.0,0.0,0.0)
        return (x/fw, y/fh, w/fw, h/fh)
