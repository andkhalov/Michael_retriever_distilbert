import os
import pickle
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from torch.utils.data import Dataset as torchDataset
import datasets
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

def mean_pool(token_embeds: torch.tensor, attention_mask: torch.tensor) -> torch.tensor:
    in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)
    return pool

'''
    model.eval(): Переводит модель в режим оценки (выключает тренировочные функции, такие как dropout).

    tokenizer(...): Токенизирует входные тексты, устанавливая максимальную длину 128, добавляя паддинг и
    обрезая длинные тексты. Результат возвращается в формате тензоров PyTorch.

    model(...): Применяет модель к токенизированным входным данным
    (идентификаторы токенов и маски внимания), получает выходные эмбеддинги (последние скрытые состояния).

    mean_pool(...): Применяет функцию усреднения к эмбеддингам, используя маску внимания, чтобы получить
    агрегированные эмбеддинги.

    return pooled_embeds: Возвращает усредненные эмбеддинги для дальнейшего использования.
'''

def encode(input_texts: list[str], tokenizer: AutoTokenizer, model: AutoModel, device: str = "cpu"
) -> torch.tensor:


    model.eval()
    tokenized_texts = tokenizer(input_texts, max_length=128,
                                padding='max_length', truncation=True, return_tensors="pt")
    token_embeds = model(tokenized_texts["input_ids"].to(device),
                         tokenized_texts["attention_mask"].to(device)).last_hidden_state
    pooled_embeds = mean_pool(token_embeds, tokenized_texts["attention_mask"].to(device))
    return pooled_embeds


class Sbert(torch.nn.Module):
    def __init__(self, max_length: int = 256, device=None):
        super().__init__()
        self.max_length = max_length
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model = AutoModel.from_pretrained('distilbert-base-uncased')
        self.bert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.linear = torch.nn.Linear(self.bert_model.config.hidden_size * 3, 1) # у нас бинарная классификация поэтому 1 нейрон

    def forward(self, data: datasets.arrow_dataset.Dataset) -> torch.tensor:
        request_input_ids = data["request_input_ids"].to(self.device)
        request_attention_mask = data["request_attention_mask"].to(self.device)
        responce_input_ids = data["responce_input_ids"].to(self.device)
        responce_attention_mask = data["responce_attention_mask"].to(self.device)

        """
    out_request = self.bert_model(...): Применяет модель BERT к входным данным (идентификаторы токенов и маска внимания) для вопроса, получая выходной объект.

    out_responce = self.bert_model(...): Применяет ту же модель BERT к входным данным для ответа.

    request_embeds = out_request.last_hidden_state: Извлекает последние скрытые состояния (эмбеддинги) для вопроса.

    responce_embeds = out_responce.last_hidden_state: Извлекает последние скрытые состояния (эмбеддинги) для ответа.

        """
        out_request = self.bert_model(request_input_ids, request_attention_mask)
        out_responce = self.bert_model(responce_input_ids, responce_attention_mask)
        request_embeds = out_request.last_hidden_state
        responce_embeds = out_responce.last_hidden_state

        """
    pooled_request_embeds = mean_pool(...): Усредняет эмбеддинги вопроса с учетом маски внимания, получая агрегированные эмбеддинги.

    pooled_responce_embeds = mean_pool(...): Усредняет эмбеддинги ответа аналогично.

    torch.cat([...], dim=-1): Объединяет (конкатенирует) три компонента:
        Усредненные эмбеддинги вопроса.
        Усредненные эмбеддинги ответа.
        Абсолютная разница между усредненными эмбеддингами вопроса и ответа.
        """
        pooled_request_embeds = mean_pool(request_embeds, request_attention_mask)
        pooled_responce_embeds = mean_pool(responce_embeds, responce_attention_mask)

        embeds =  torch.cat([pooled_request_embeds, pooled_responce_embeds,
                             torch.abs(pooled_request_embeds - pooled_responce_embeds)],
                            dim=-1)
        return self.linear(embeds)


# загружаем модель

# Предполагается, что device уже определён (например, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model_path = "michael_scott_model.bin"

# назначаем устройство
device = torch.device("cpu")

# Создаём экземпляр модели
model = Sbert().to(device)

print("Модель загрузили")

# Загружаем веса из сохранённого файла
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()  # переводим модель в режим инференса

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# теперь загрузим базу ответов с эмбедингами
with open("responces_base.pkl", "rb") as f:
    df = pickle.load(f)
    
print("Базу ответов загурзили")

class UniqueResponseRetriever:
    """
    Ретривер, который получает датасет с эмбеддингами и оставляет только уникальные ответы.
    Затем по входному запросу вычисляет его эмбеддинг (с использованием FT модели) и ищет по косинусной близости
    наиболее похожий ответ из базы.
    """
    def __init__(self, df, model, tokenizer, max_length: int = 256, device=None):
        """
        Args:
          hf_dataset: объект Hugging Face Dataset, содержащий как минимум колонки:
                      "response" и "ft_emb_responce"
          model: дообученная FT модель (экземпляр Sbert)
          tokenizer: токенайзер, соответствующий модели
          max_length: максимальная длина последовательности для токенизации
          device: устройство для вычислений (если None, определяется автоматически)
        """
        self.device = device
            
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model.eval()
        
        df_unique = df.drop_duplicates(subset=["response"])
        self.responses = df_unique["response"].tolist()
        # Предполагается, что "ft_emb_responce" хранится как список чисел для каждого примера
        self.embeddings = np.vstack(df_unique["ft_emb_responce"].apply(np.array).tolist())
        print(f"Загружено {len(self.responses)} уникальных ответов из базы.")
    
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        """
        Усредняет эмбеддинги токенов с учётом маски внимания.
        """
        token_embeddings = model_output.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def embed_text(self, text: str) -> np.array:
        """
        Вычисляет эмбеддинг для данного текста с использованием токенайзера и FT модели.
        """
        encoded = self.tokenizer(text, padding="max_length", truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            output = self.model.bert_model(**encoded)
            emb = self.mean_pooling(output, encoded["attention_mask"])
        return emb.cpu().numpy()  # shape: (1, D)
    
    def retrieve(self, query: str, top_k: int = 1) -> tuple[list[str], np.array]:
        """
        По входному запросу вычисляет его эмбеддинг, затем ищет top_k ответов из базы
        с наибольшей косинусной схожестью.
        Returns:
          - Список найденных ответов.
          - Соответствующие значения косинусной схожести.
        """
        query_emb = self.embed_text(query)  # shape: (1, D)
        similarities = cosine_similarity(query_emb, self.embeddings).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.responses[i] for i in top_indices], similarities[top_indices]


# Создаем экземпляр ретривера на базе уникальных ответов
retriever = UniqueResponseRetriever(df=df,
                                    model=model,
                                    tokenizer=tokenizer,
                                    max_length=256,
                                    device=device)

retriever.model.to(device)
retriever.model.eval()

app = Flask(__name__, static_folder="static")

CORS(app)  # Разрешаем кросс-доменный доступ для фронтенда

@app.route("/")
def index():
    # Если фронтенд работает в отдельном контейнере, этот маршрут можно оставить пустым
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400
    # Получаем ответ от ретривера
    resp, sim = retriever.retrieve(query, top_k=1)
    return jsonify({
        "query": query,
        "response": resp[0],
        "similarity": float(sim[0])
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)