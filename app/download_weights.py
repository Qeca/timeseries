import os
import boto3

# Настройки доступа к S3
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "IWGM8NXL28FBMXEXZ5UF")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "8WWRucvWYmuucCX5x28x9hQAH1FKbGWlJQXeAVfV")
S3_BUCKET = os.getenv("S3_BUCKET", "02275c7b-5137c2f2-5ba5-4d2e-be18-10ec6f2789af")
REGION = os.getenv("AWS_DEFAULT_REGION", "ru-1")
S3_ENDPOINT_URL = "https://s3.timeweb.cloud"

# Список ключей и соответствующих имен файлов
WEIGHTS_MAP = {
    "1": "weights/1_best_informer_model.pth",
    "5": "weights/5_best_informer_model.pth",
    "10": "weights/10_best_informer_model.pth",
    "20": "weights/20_best_informer_model.pth",
    "30": "weights/30_best_informer_model.pth",
    "scaler": "weights/scaler.pkl"  # Добавляем файл скейлера
}

LOCAL_DIR = os.path.join(os.path.dirname(__file__), "weights")
os.makedirs(LOCAL_DIR, exist_ok=True)

def download_file(s3_client, bucket, key, local_path):
    print(f"Downloading {key} from s3://{bucket} to {local_path}")
    s3_client.download_file(bucket, key, local_path)
    print(f"Downloaded: {local_path}")

if __name__ == "__main__":
    # Создаём сессию
    session = boto3.session.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=REGION
    )
    s3 = session.client(
        service_name='s3',
        endpoint_url=S3_ENDPOINT_URL
    )

    # Скачиваем файлы
    for name, key in WEIGHTS_MAP.items():
        local_file = os.path.join(LOCAL_DIR, os.path.basename(key))
        try:
            download_file(s3, S3_BUCKET, key, local_file)
        except Exception as e:
            print(f"❌ Error downloading {key}: {e}")

    print("✅ Все файлы успешно загружены.")