from collections import defaultdict
import numpy as np
import time

class PlayerTracker:
    def __init__(self):

        self.positions = defaultdict(list) # positions history
        self.last_seen = {} # last seen frame for each player
        self.id_to_jersey = {} # mapping player ID to jersey number, in jersey_readeer we will update this mapping
        self.max_history = 300 # max history length for positions, after 300 frames we will start removing old positions to save memory
        self.total_distance = defaultdict(float) # total distance traveled for each player

    
    def update(self, detections, frame_number):
        # recieves a list of detections and updates the positions and distance traveled for each player
        for det in detections:
            track_id = det ["track_id"]
            if track_id <0:
                continue # skip invalid detections

            x1, y1, x2, y2 = det["bbox"]
            # calculate the center of the bounding box
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            if self.positions[track_id]: # if we have a previous position for this player, calculate the distance traveled since last update
                last_x, last_y, _ =self.positions[track_id][-1] # get last position from history
                dist = np.sqrt((x_center - last_x) ** 2 + (y_center - last_y) ** 2) # calculate distance between last position and current position
                self.total_distance[track_id] += dist # add distance to total distance traveled for this player

            self.positions[track_id].append((x_center, y_center, frame_number)) # add position to position history

            if len(self.positions[track_id]) > self.max_history:
                self.positions[track_id].pop(0) # remove old positions if history exceeds max length

            self.last_seen[track_id] = frame_number # update last seen frame for this player

    
    def get_position(self, track_id):
        # returns the current position of the player with the given track_id
        if track_id not in self.positions or not self.positions[track_id]:
            return None # no position available
        return self.positions[track_id][-1] #last entry in the position history is the current position
    

    def get_trajectory(self, track_id, last_n_frames = 60):
        # returns last n positions of the player with the given track_id
        # useful to draw the trajectory of the player in the last n frames
        if track_id not in self.positions:
            return []
        return self.positions[track_id][-last_n_frames:]
    

    def get_distance_traveled(self, track_id):
        # calculates the total distance traveled by the player with the given track_id adding the distance between consecutive positions in the position history
        history = self.positions.get(track_id, [])

        if len(history) < 2:
            return 0.0 # not enough data to calculate distance
        
        total = 0.0
        for i in range(1, len(history)):
            x1, y1, _ = history[i-1]
            x2, y2, _ = history[i]
            total += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) # calculate distance between consecutive positions and add to total
        return round(total, 2) # round to 2 decimal places for better readability
    

    def get_all_players(self):
        # returns a list of all currently tracked player IDs
        return list(self.positions.keys())
    

    def assign_jersey(self, track_id, jersey_number):
        # assigns a jersey number to a player with the given track_id
        self.id_to_jersey[track_id] = jersey_number

    
    def get_jersey(self, track_id):
        # returns the jersey number assigned to the player with the given track_id, or None if no jersey number is assigned
        return self.id_to_jersey.get(track_id, None)
    

    def get_summary(self):
        # returns the summary of all tracked players, including their jersey number (if assigned) and total distance traveled at the actual moment
        summary = {}
        for track_id in self.positions.keys():
            jersey = self.get_jersey(track_id)
            pos = self.get_position(track_id)
            dist = self.get_distance_traveled(track_id)
            summary[track_id] = {
                "jersey": jersey,
                "last_position": pos,
                "distance_px": dist,
                "frames_tracked": len(self.positions[track_id])
            }

        return summary


    
