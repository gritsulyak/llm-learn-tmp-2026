import os
from pathlib import Path
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTModelForFeatureExtraction
from transformers import AutoTokenizer


def export_classification():
    model_id = "huawei-noah/TinyBERT_General_4L_312D"
    output_dir = Path("model_repository/classification/1")

    print(f"Exporting classification model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(output_dir)

    model = ORTModelForSequenceClassification.from_pretrained(
        model_id, export=True
    )
    model.save_pretrained(output_dir)
    print(f"  Saved to {output_dir}")


def export_embedding():
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    output_dir = Path("model_repository/embedding/1")

    print(f"Exporting embedding model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(output_dir)

    model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
    model.save_pretrained(output_dir)
    print(f"  Saved to {output_dir}")


if __name__ == "__main__":
    os.makedirs("model_repository/classification/1", exist_ok=True)
    os.makedirs("model_repository/embedding/1", exist_ok=True)

    export_classification()
    export_embedding()

    print("\nExport complete. Generation model uses Python backend (no ONNX needed).")
