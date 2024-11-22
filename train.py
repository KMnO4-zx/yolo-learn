from ultralytics import YOLO

if __name__ == "__main__":
    # 加载预训练模型
    model = YOLO("./ultralytics/cfg/models/v8/yolov8n.yaml").load('./yolov8n.pt')
    # 开始训练
    model.train(
        data="./data/coco8/coco8.yaml",
        epochs=200,
        imgsz=640,
        batch=4,
        device="0",
    )