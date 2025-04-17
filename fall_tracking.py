# fall_tracking.py

import time
import math

class FallTracker:
    def __init__(self, cooldown=1.0, persistence=1.0, match_threshold=50):
        self.cooldown = cooldown
        self.persistence = persistence
        self.match_threshold = match_threshold
        self.active_tracks = {}  # {id: {"centroid": (x,y), "last_seen": t, "last_triggered": t, "falling": bool}}
        self.unique_fallers = set() 
        self.next_id = 0

    def _distance(self, c1, c2):
        return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

    def update(self, centroids_fallen):
        now = time.time()
        triggered_ids = []

        # Step 1: Remove stale tracks
        to_remove = [id for id, v in self.active_tracks.items() if now - v["last_seen"] > self.persistence]
        for id in to_remove:
            del self.active_tracks[id]

        # Step 2: Match incoming centroids
        for cx, cy, is_fallen in centroids_fallen:
            matched_id = None
            for id, data in self.active_tracks.items():
                if self._distance((cx, cy), data["centroid"]) < self.match_threshold:
                    matched_id = id
                    break

            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1

            # Update or create the track
            track = self.active_tracks.get(matched_id, {
                "centroid": (cx, cy),
                "last_seen": now,
                "last_triggered": 0,
                "falling": False
            })

            track["centroid"] = (cx, cy)
            track["last_seen"] = now

            if is_fallen:
                if not track["falling"] and now - track["last_triggered"] > self.cooldown:
                    triggered_ids.append(matched_id)
                    self.unique_fallers.add(matched_id)
                    track["last_triggered"] = now
                track["falling"] = True
            else:
                track["falling"] = False

            self.active_tracks[matched_id] = track

        return triggered_ids  # Only the IDs that caused a fall trigger this frame

    def get_unique_faller_count(self):
        return len(self.unique_fallers)
