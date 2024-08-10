from ultralytics import YOLO

if __name__ == "__main__":
    # 加载预训练模型
    model = YOLO("./ultralytics/cfg/models/v8/yolov8-obb-attention-cot.yaml").load('./yolov8n-obb.pt')
    # 开始训练
    model.train(
        data="/mnt/g/WorkSpace/dataset/yolo-oob/dataset-radar/data.yaml",
        epochs=200,
        imgsz=640,
        device="0",
    )