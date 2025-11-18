import cv2
import torch
import torchvision.transforms as transforms

from src.realtime.q import Queue
from src.realtime.logger import Logger

from src.models.model_class.baseline_vgg16 import VGG16Baseline


class VideoInference:
    def __init__(self, model_path="dict_models/vgg16_baseline.pth"):
    
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        self.model = VGG16Baseline(num_classes=2, pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.class_names = {0: "Non-Accident", 1: "Accident"}

        self.logger = Logger(log_path="logs/", log_level=0)
        self.queue = Queue(n=120, check_range=30, density=20, logger=self.logger)

        self.logger.log(f"Loaded model from {model_path}", log_level=0)
        self.logger.log(f"Running on device: {self.device}", log_level=0)

    def preprocess(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = transforms.functional.to_pil_image(img)
        img_tensor = self.transform(img_pil)
        img_tensor = img_tensor.unsqueeze(0)  # (1, C, H, W)
        return img_tensor.to(self.device)

    def predict(self, frame):
        
        tensor = self.preprocess(frame)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, 1)

        label_id = pred.item()
        confidence = conf.item()
        return label_id, confidence

    def run(self, source=0):
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            self.logger.log("Cannot open video source:", source, log_level=0)
            return

        self.logger.log("Starting video stream...", log_level=2)

        while True:
            ret, frame = cap.read()
            if not ret:
                self.logger.log("Stream ended.", log_level=2)
                break

            label, conf = self.predict(frame)
            text = f"{self.class_names[label]} ({conf:.2f})"

            color = (0, 255, 0) if label == 0 else (0, 0, 255)
            cv2.putText(frame, text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            accident, noise = self.queue.check(conf if label == 1 else 1 - conf)

            if accident and not noise:
                cv2.putText(frame, "ACCIDENT DETECTED!", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            if accident and noise:
                cv2.putText(frame, "ACCIDENT WITH NOISE DETECTED!", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

            cv2.imshow("Real-time Accident Detection", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

