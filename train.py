from ultralytics import YOLO

if __name__ == "__main__":
    # 加载预训练模型
    model = YOLO("./ultralytics/cfg/models/v8/yolov8-attention-se.yaml").load('./yolov8n.pt')
    # 开始训练
    model.train(
        data="/mnt/g/WorkSpace/dataset/YOLOv8-loopy/dataset/loopy.yaml",
        epochs=200,
        imgsz=640,
        device="0",
    )