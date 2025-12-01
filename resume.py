# from ultralytics import YOLO
# import cv2

# # # Create a new YOLO model from scratch
# model = YOLO("yolo11m.pt")

# # Load a pretrained YOLO model (recommended for training)
# # model = YOLO("runs/detect/train4/weights/best.pt")

# results = model.train(
#     data="/home/saif/Desktop/Capston1/data.yaml",
#     epochs=10000,
#     batch=5,          
#     imgsz=640,        
#     workers=2,
#     device=0,
#     patience=200,
#     cache=False      
# )


# # # Evaluate the model's performance on the validation set
# results = model.val()

# # # Perform object detection on an image using the model
# results = model("train/images/-01-15-1-1-1-1-1-mp40_jpg.rf.267f7b12dfa705397f4fed96098aae3d.jpg")
# annotated = results[0].plot()
# cv2.imwrite("runs/detect/inference_annotated.jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
# print("Saved: runs/detect/inference_annotated.jpg")
# # Export the model to ONNX format
# # success = model.export(format="onnx")



from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/last.pt")
results = model.train(resume=True)
results = model.val()
image_path = "train/images/-01-15-1-1-1-1-1-mp40_jpg.rf.267f7b12dfa705397f4fed96098aae3d.jpg"
results = model(image_path)
annotated = results[0].plot()

cv2.imwrite("runs/detect/inference_annotated.jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
print("Saved: runs/detect/inference_annotated.jpg")