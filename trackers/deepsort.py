"""
DeepSORT: Deep Learning based SORT Tracker
Implementation with appearance features
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from collections import deque


class FeatureExtractor(nn.Module):
    """
    Simple CNN for appearance feature extraction
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, 128)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return nn.functional.normalize(x, p=2, dim=1)


class DeepKalmanTracker:
    """
    Kalman tracker with appearance features
    """
    count = 0
    
    def __init__(self, bbox, feature):
        # Initialize Kalman filter (same as SORT)
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        
        # Track state
        self.time_since_update = 0
        self.id = DeepKalmanTracker.count
        DeepKalmanTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
        # Appearance features
        self.features = deque([feature], maxlen=100)
        
    def update(self, bbox, feature):
        """Update with bbox and feature"""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))
        self.features.append(feature)
    
    def predict(self):
        """Predict next state"""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.convert_x_to_bbox(self.kf.x)
    
    def get_state(self):
        """Get current state"""
        return self.convert_x_to_bbox(self.kf.x)
    
    def get_feature(self):
        """Get average feature"""
        return np.mean(self.features, axis=0)
    
    @staticmethod
    def convert_bbox_to_z(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h
        r = w / float(h) if h != 0 else 1
        return np.array([x, y, s, r]).reshape((4, 1))
    
    @staticmethod
    def convert_x_to_bbox(x):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        return np.array([
            x[0] - w / 2., x[1] - h / 2.,
            x[0] + w / 2., x[1] + h / 2.
        ]).reshape((1, 4))


class DeepSORTTracker:
    """
    DeepSORT tracker with appearance features
    """
    
    def __init__(self, frame_size, max_age=30, min_hits=3, 
                 iou_threshold=0.3, max_cosine_distance=0.2):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_cosine_distance = max_cosine_distance
        self.trackers = []
        self.frame_count = 0
        
        # Feature extractor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = FeatureExtractor().to(self.device)
        self.feature_extractor.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.frame_size = frame_size
        self.current_frame = None
    
    def extract_features(self, frame, bboxes):
        """Extract appearance features from detections"""
        if len(bboxes) == 0:
            return np.array([])
        
        features = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox[:4])
            
            # Clip to frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                features.append(np.zeros(128))
                continue
            
            # Crop and extract feature
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                features.append(np.zeros(128))
                continue
            
            # Transform and extract
            img_tensor = self.transform(crop).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feature = self.feature_extractor(img_tensor)
                features.append(feature.cpu().numpy().flatten())
        
        return np.array(features)
    
    def update(self, dets, frame=None):
        """
        Update tracker with detections
        dets: numpy array of detections [[x1,y1,x2,y2,score,class], ...]
        frame: current frame for feature extraction
        """
        self.frame_count += 1
        self.current_frame = frame
        
        # Extract features if frame is provided
        if frame is not None and len(dets) > 0:
            features = self.extract_features(frame, dets)
        else:
            features = np.array([])
        
        # Predict existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # Associate detections to trackers
        if len(dets) > 0:
            matched, unmatched_dets, unmatched_trks = self.associate(
                dets[:, :4], trks[:, :4], features
            )
            
            # Update matched trackers
            for m in matched:
                det_idx, trk_idx = m[0], m[1]
                feature = features[det_idx] if len(features) > 0 else np.zeros(128)
                self.trackers[trk_idx].update(dets[det_idx, :4], feature)
            
            # Create new trackers for unmatched detections
            for i in unmatched_dets:
                feature = features[i] if len(features) > 0 else np.zeros(128)
                trk = DeepKalmanTracker(dets[i, :4], feature)
                self.trackers.append(trk)
        
        # Prepare output
        ret = []
        for trk in self.trackers:
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and \
               (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
        
        # Remove dead trackers
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
    
    def associate(self, detections, trackers, det_features):
        """
        Associate detections with trackers using IoU and appearance
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        
        # IoU matrix
        iou_matrix = self.iou_batch(detections, trackers)
        
        # Appearance distance matrix
        if len(det_features) > 0:
            trk_features = np.array([t.get_feature() for t in self.trackers])
            cost_matrix = self.cosine_distance(det_features, trk_features)
            
            # Combine IoU and appearance
            cost_matrix = 0.5 * (1 - iou_matrix) + 0.5 * cost_matrix
        else:
            cost_matrix = 1 - iou_matrix
        
        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_indices = np.column_stack((row_ind, col_ind))
        
        # Filter matches
        matches = []
        unmatched_dets = []
        unmatched_trks = []
        
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_dets.append(m[0])
                unmatched_trks.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        
        # Find unmatched detections and trackers
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_dets.append(d)
        
        for t in range(len(trackers)):
            if t not in matched_indices[:, 1]:
                unmatched_trks.append(t)
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches, np.array(unmatched_dets), np.array(unmatched_trks)
    
    @staticmethod
    def iou_batch(bb_test, bb_gt):
        """Compute IoU between two sets of boxes"""
        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)
        
        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        
        o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
                  + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
        return o
    
    @staticmethod
    def cosine_distance(a, b):
        """Compute cosine distance between feature vectors"""
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        return 1 - np.dot(a, b.T)
