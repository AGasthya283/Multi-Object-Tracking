"""
BYTETrack: Multi-Object Tracking by Associating Every Detection Box
Implementation of BYTETrack algorithm
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque


class STrack:
    """
    Single target track with Kalman filtering
    """
    shared_kalman = None
    track_id = 0
    
    def __init__(self, tlwh, score):
        # Wait for activation
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        
        self.score = score
        self.tracklet_len = 0
        
        self.state = 'new'
    
    def predict(self):
        """Predict next state"""
        mean_state = self.mean.copy()
        if self.state != 'tracked':
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
    
    @staticmethod
    def multi_predict(stracks):
        """Predict multiple tracks"""
        if len(stracks) > 0:
            for st in stracks:
                st.predict()
    
    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        
        self.tracklet_len = 0
        self.state = 'tracked'
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
    
    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivate a track"""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = 'tracked'
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
    
    def update(self, new_track, frame_id):
        """Update a matched track"""
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.state = 'tracked'
        self.is_activated = True
        self.score = new_track.score
    
    @property
    def tlwh(self):
        """Get current position in bounding box format (top left x, top left y, width, height)"""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret
    
    @property
    def tlbr(self):
        """Get current position in bounding box format (min x, miny, max x, max y)"""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    
    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format (center x, center y, aspect ratio, height)"""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    
    @staticmethod
    def next_id():
        STrack.track_id += 1
        return STrack.track_id
    
    def mark_lost(self):
        self.state = 'lost'
    
    def mark_removed(self):
        self.state = 'removed'


class KalmanFilterXYAH:
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    
    The 8-dimensional state space (x, y, a, h, vx, vy, va, vh) contains
    the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.
    """
    
    def __init__(self):
        ndim, dt = 4, 1.
        
        # Create Kalman filter model matrices
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        
        # Motion and observation uncertainty
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
    
    def initiate(self, measurement):
        """Create track from unassociated measurement"""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance
    
    def predict(self, mean, covariance):
        """Run Kalman filter prediction step"""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        
        return mean, covariance
    
    def update(self, mean, covariance, measurement):
        """Run Kalman filter update step"""
        import scipy.linalg
        
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))
        
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        
        chol_factor, lower = scipy.linalg.cho_factor(
            covariance + innovation_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - mean
        
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, covariance))
        return new_mean, new_covariance


class ByteTracker:
    """
    BYTETrack implementation
    """
    
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        
        self.frame_id = 0
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        
        self.kalman_filter = KalmanFilterXYAH()
    
    def update(self, dets):
        """
        Update tracker with detections
        dets: numpy array [[x1, y1, x2, y2, score, class], ...]
        Returns: numpy array [[x1, y1, x2, y2, track_id], ...]
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        if len(dets) == 0:
            dets = np.empty((0, 6))
        
        # Separate detections by score
        remain_inds = dets[:, 4] > self.track_thresh
        dets_second = dets[~remain_inds]
        dets = dets[remain_inds]
        
        # Convert to STrack format
        if len(dets) > 0:
            detections = [STrack(self.xyxy_to_tlwh(tlbr[:4]), tlbr[4]) for tlbr in dets]
        else:
            detections = []
        
        if len(dets_second) > 0:
            detections_second = [STrack(self.xyxy_to_tlwh(tlbr[:4]), tlbr[4]) for tlbr in dets_second]
        else:
            detections_second = []
        
        # Add newly detected tracklets to tracked_stracks
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        # Step 1: First association with high score detections
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        
        dists = self.iou_distance(strack_pool, detections)
        matches, u_track, u_detection = self.linear_assignment(dists, thresh=self.match_thresh)
        
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == 'tracked':
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        # Step 2: Second association with low score detections
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == 'tracked']
        dists = self.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = self.linear_assignment(dists, thresh=0.5)
        
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == 'tracked':
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == 'lost':
                track.mark_lost()
                lost_stracks.append(track)
        
        # Deal with unconfirmed tracks
        detections = [detections[i] for i in u_detection]
        dists = self.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = self.linear_assignment(dists, thresh=0.7)
        
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        
        # Step 3: Init new stracks
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        
        # Step 4: Update state
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.track_buffer:
                track.mark_removed()
                removed_stracks.append(track)
        
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == 'tracked']
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )
        
        # Get output
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        
        # Convert to numpy array
        outputs = []
        for t in output_stracks:
            tlbr = t.tlbr
            outputs.append([tlbr[0], tlbr[1], tlbr[2], tlbr[3], t.track_id])
        
        if len(outputs) > 0:
            return np.array(outputs)
        return np.empty((0, 5))
    
    @staticmethod
    def xyxy_to_tlwh(bbox):
        """Convert xyxy to tlwh format"""
        return np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
    
    @staticmethod
    def iou_distance(atracks, btracks):
        """Compute IoU distance between tracks"""
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
        
        ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
        if ious.size == 0:
            return ious
        
        for i, atlbr in enumerate(atlbrs):
            for j, btlbr in enumerate(btlbrs):
                ious[i, j] = ByteTracker.bbox_iou(atlbr, btlbr)
        
        return 1 - ious
    
    @staticmethod
    def bbox_iou(boxA, boxB):
        """Compute IoU between two boxes"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou
    
    @staticmethod
    def linear_assignment(cost_matrix, thresh):
        """Linear assignment with threshold"""
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        
        matches, unmatched_a, unmatched_b = [], [], []
        
        try:
            import lap
            cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
            
            for ix, mx in enumerate(x):
                if mx >= 0:
                    matches.append([ix, mx])
            
            unmatched_a = np.where(x < 0)[0]
            unmatched_b = np.where(y < 0)[0]
        except ImportError:
            # Fallback to scipy linear_sum_assignment
            from scipy.optimize import linear_sum_assignment
            
            # Filter by threshold first
            indices = np.where(cost_matrix < thresh)
            valid_cost = cost_matrix.copy()
            valid_cost[cost_matrix >= thresh] = 1e6
            
            row_ind, col_ind = linear_sum_assignment(valid_cost)
            
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < thresh:
                    matches.append([i, j])
            
            matched_rows = set([m[0] for m in matches])
            matched_cols = set([m[1] for m in matches])
            
            unmatched_a = [i for i in range(cost_matrix.shape[0]) if i not in matched_rows]
            unmatched_b = [j for j in range(cost_matrix.shape[1]) if j not in matched_cols]
        
        matches = np.asarray(matches) if matches else np.empty((0, 2), dtype=int)
        
        return matches, np.array(unmatched_a), np.array(unmatched_b)
    
    @staticmethod
    def joint_stracks(tlista, tlistb):
        """Join two track lists"""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res
    
    @staticmethod
    def sub_stracks(tlista, tlistb):
        """Subtract track list b from a"""
        stracks = {}
        for t in tlista:
            stracks[t.track_id] = t
        for t in tlistb:
            tid = t.track_id
            if stracks.get(tid, 0):
                del stracks[tid]
        return list(stracks.values())
    
    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        """Remove duplicate tracks"""
        pdist = ByteTracker.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = list(), list()
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        
        resa = [t for i, t in enumerate(stracksa) if not i in dupa]
        resb = [t for i, t in enumerate(stracksb) if not i in dupb]
        return resa, resb