from app.config import AppConfig
from app.pipeline import VideoPipeline


def main():
    config = AppConfig()
    pipeline = VideoPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
