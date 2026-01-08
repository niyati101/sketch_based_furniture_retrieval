# image_search.py
import os
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer, util

def load_descriptions(file_path):
    img_to_desc = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                parts = line[1:-1].split(';')
                if len(parts) < 2:
                    continue
                image_id = parts[0].strip()
                description = ';'.join(parts[1:]).strip()
                image_path = image_id if image_id.lower().endswith(('.jpg', '.jpeg', '.png')) else f"{image_id}.jpg"
                img_to_desc[image_path] = description
            except Exception as e:
                print(f"⚠️ Error parsing line: {line}\nError: {e}")
    return img_to_desc

def index_all_images(image_dir):
    image_lookup = {}
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                full_path = os.path.join(root, file)
                image_lookup[file] = full_path
    return image_lookup

class TextSearchEngine:
    def __init__(self, desc_path, img_dir):
        self.img_to_desc = load_descriptions(desc_path)
        self.image_lookup = index_all_images(img_dir)
        self.image_paths = []
        self.captions = []

        for filename, desc in self.img_to_desc.items():
            full_path = self.image_lookup.get(filename)
            if full_path and os.path.exists(full_path):
                self.image_paths.append(full_path)
                self.captions.append(str(desc))
            else:
                print(f"⚠️ Image not found for: {filename}")

        print(f"✅ Found {len(self.image_paths)} valid images and captions.")
        self.model = SentenceTransformer("clip-ViT-B-32")
        self.caption_embeddings = self.model.encode(self.captions, convert_to_tensor=True, show_progress_bar=True)

    def search_text(self, query, top_k=5):
        query_emb = self.model.encode(query, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_emb, self.caption_embeddings)[0]
        top_results = torch.topk(scores, k=min(top_k, len(scores)))

        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            results.append({
                'image_path': self.image_paths[idx],
                'caption': self.captions[idx],
                'score': float(score)
            })
        return results