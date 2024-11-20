from ultralytics import YOLO

if __name__ == "__main__":
    # 加载预训练模型
    model = YOLO("./ultralytics/cfg/models/v8/yolov8n-obb.yaml").load('./yolov8n.pt')
    # 开始训练
    model.train(
        data="/mnt/g/WorkSpace/github_project/yolo-learn/data/data.yaml",
        epochs=200,
        imgsz=640,
        batch=4,
        device="0",
    )