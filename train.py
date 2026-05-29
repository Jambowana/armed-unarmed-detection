from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")  

    model.train(
        data=r"C:\Users\ASUS\Desktop\Gun-Detection-1\data.yaml",
        epochs=50,
        imgsz=640,
        device=0,        
        workers=0,
        batch=16
    )

