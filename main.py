# main.py
import cv2
from tracking.detector import PlayerDetector

VIDEO_PATH = "video/match.mp4"

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Errore: video non trovato.")
        return

    detector = PlayerDetector()
    print("Analisi avviata...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detector.process_frame(frame)
        cv2.imshow("Volley Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Analisi completata.")

if __name__ == "__main__":
    main()