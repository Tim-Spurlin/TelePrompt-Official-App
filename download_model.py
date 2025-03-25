# download_model.py
from sentence_transformers import SentenceTransformer

def download_and_save():
    model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    print(f"Downloading {model_name}...")
    model = SentenceTransformer(model_name)
    model.save("local_models/multi-qa-mpnet-base-dot-v1")
    print("Model downloaded and saved locally!")

if __name__ == "__main__":
    download_and_save()
