# main.py
import cv2
from tracking.detector import PlayerDetector
from tracking.tracker import PlayerTracker

VIDEO_PATH = "video/match.mp4"

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Errore: video non trovato.")
        return

    detector = PlayerDetector()
    tracker = PlayerTracker()
    frame_number = 0
    print("Analisi avviata...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, detections = detector.get_detections(frame) # get raw data and detections for the current frame
        tracker.update(detections, frame_number) # update the tracker with the new detections


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
    print("Analisi completata.")

if __name__ == "__main__":
    main()