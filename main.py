# main.py
import cv2
from tracking.detector import PlayerDetector
from tracking.tracker import PlayerTracker
from tracking.jersey_reader import JerseyReader

VIDEO_PATH = "video/match.mp4"

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: video not found.")
        return

    detector = PlayerDetector()
    tracker = PlayerTracker()
    jersey_reader = JerseyReader()

    frame_number = 0
    print("Analysis started...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, detections = detector.get_detections(frame) # get raw data and detections for the current frame
        tracker.update(detections, frame_number) # update the tracker with the new detections

        if frame_number % 5 == 0: # every 5 frames, try to read jersey numbers to reduce computational load
            jersey_reader.update(frame, detections) # update the jersey reader with the new detections and frame
            # update the tracker with confirmed jersey numbers from the jersey reader
            for track_id, jersey in jersey_reader.get_all_confirmed().items():
                print(f"Assigning jersey {jersey} to player ID {track_id}")
                tracker.assign_jersey(track_id, jersey)

        # display the bounding box with tracking and jersey info
        for det in detections:
            track_id = det['track_id']
            jersey = tracker.get_jersey(track_id)
            if jersey is not None:
                x1, y1, _, _ = det['bbox']
                cv2.putText(frame, f"ID:{track_id} Jersey:{jersey}",
                             (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


        cv2.imshow("Volley Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("\n--- SUMMARY TRACKING ---") 
    summary = tracker.get_summary()
    for track_id, data in summary.items():
        print(f"Player ID:{track_id} | "
              f"Jersey:{data['jersey']} | "
              f"Tracked Frames:{data['frames_tracked']} | "
              f"Distance:{data['distance_px']}px")


    cap.release()
    cv2.destroyAllWindows()
    print("Analysis completed.")

if __name__ == "__main__":
    main()