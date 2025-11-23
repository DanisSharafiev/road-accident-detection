import argparse
from src.realtime.video_inference import VideoInference
from src.realtime.q import Queue
from src.realtime.logger import Logger

def parse_args():
    parser = argparse.ArgumentParser(description="Real-time Road Accident Detection")

    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Путь к видеофайлу (например: video.mp4)"
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="ID камеры (обычно 0)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="dict_models/vgg16_baseline.pth",
        help="Путь к весам модели .pth"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.video is None and args.camera is None:
        print("[ERROR] Укажите источник: --video path/to/file.mp4 или --camera 0")
        return

    inference = VideoInference(model_path=args.model)

    if args.video:
        print(f"[INFO] Запуск видеофайла: {args.video}")
        inference.run(source=args.video)

    elif args.camera is not None:
        print(f"[INFO] Запуск камеры: {args.camera}")
        inference.run(source=args.camera)


if __name__ == "__main__":
    main()
