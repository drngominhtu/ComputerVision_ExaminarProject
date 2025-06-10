import cv2
import numpy as np

def real_time_hand_sign_detection(model):
    """Nhận diện thời gian thực từ webcam"""
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Không thể mở webcam!")
            return
        
        print("Webcam đã sẵn sàng. Nhấn 'q' để thoát...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Không thể đọc frame từ webcam!")
                break
            
            # Flip frame để như nhìn gương
            frame = cv2.flip(frame, 1)
            
            # Preprocess frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # ROI (Region of Interest)
            roi_x, roi_y, roi_w, roi_h = 100, 100, 300, 300
            roi = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            if roi.size > 0:
                # Resize và normalize
                roi_resized = cv2.resize(roi, (28, 28))
                roi_normalized = roi_resized / 255.0
                
                # Predict
                prediction = model.predict(roi_normalized.reshape(1, 28, 28))
                predicted_letter = chr(prediction[0] + 65)
                
                # Confidence (simple distance-based)
                confidence = "Medium"  # Placeholder
                
                # Display ROI nhỏ
                roi_display = cv2.resize(roi, (150, 150))
                frame[10:160, 10:160] = cv2.cvtColor(roi_display, cv2.COLOR_GRAY2BGR)
                
                # Draw rectangle and text
                cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0, 255, 0), 3)
                cv2.putText(frame, f"Prediction: {predicted_letter}", 
                           (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(frame, f"Confidence: {confidence}", 
                           (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to quit", 
                           (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, "ROI Preview:", 
                           (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Hand Sign Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Đã thoát real-time prediction")
        
    except Exception as e:
        print(f"Lỗi trong real-time prediction: {e}")
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()