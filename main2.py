from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_NAME = "microsoft/trocr-large-handwritten"
MODEL_DIR = "./models/trocr_cache"

TrOCRProcessor.from_pretrained(MODEL_NAME, cache_dir=MODEL_DIR)
VisionEncoderDecoderModel.from_pretrained(MODEL_NAME, cache_dir=MODEL_DIR)

print("Model downloaded successfully")
