from ultralytics import YOLO

if __name__ == "__main__":
    # 加载预训练模型
    model = YOLO("./ultralytics/cfg/models/v8/yolov8l.yaml").load('yolov8l.pt')
    # 开始训练
    # model.train(
    #     data="./data/coco8/coco8.yaml",
    #     epochs=200,
    #     imgsz=640,
    #     batch=4,
    #     device="cpu",
    # )
    results = model.predict("./data/coco8/images/val/000000000036.jpg")
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        print('boxes', boxes)
        masks = result.masks  # Masks object for segmentation masks outputs
        print('masks', masks)
        keypoints = result.keypoints  # Keypoints object for pose outputs
        print('keypoints', keypoints)
        probs = result.probs  # Probs object for classification outputs
        print('probs', probs)
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        # result.save(filename="result.jpg")  # save to disk