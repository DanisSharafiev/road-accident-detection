import cv2
import torch
import torchvision.transforms as transforms

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

        print(f"[INFO] Loaded model from {model_path}")
        print(f"[INFO] Running on device: {self.device}")

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
            print("[ERROR] Cannot open video source:", source)
            return

        print("[INFO] Starting video stream...")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Stream ended.")
                break

            label, conf = self.predict(frame)
            text = f"{self.class_names[label]} ({conf:.2f})"

            color = (0, 255, 0) if label == 0 else (0, 0, 255)
            cv2.putText(frame, text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow("Real-time Accident Detection", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

