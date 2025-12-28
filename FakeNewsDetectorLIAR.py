from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# Login to Hugging Face
login("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")


# # LOAD LLaMA 2-13B
model_id = "meta-llama/Llama-2-13b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",      # auto-assign layers to GPU/CPU
    load_in_4bit=True,      # 4-bit quantization for 12GB GPU
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.eval()


# Only If i want to use the Mistral model instead of LLaMA, uncomment below
# Mistral fake-news LoRA
# model_id = "mistralai/Mistral-7B-Instruct-v0.1"
# peft_model_name = "bpavlsh/Mistral-Fake-News-Detection"

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# base_model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     load_in_4bit=True,
#     device_map="auto",
#     torch_dtype=torch.float16
# )
# model = PeftModel.from_pretrained(base_model, peft_model_name)

# Sentence-transformer embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

filepath = "/home/tacticrabbit/LIARDATA.tsv"
liar_data = pd.read_csv(filepath, sep='\t', header=None)
liar_data.columns = [
    'id', 'label', 'text', 'subject', 'speaker', 'job_title', 'state_info', 'party',
    'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts',
    'pants_on_fire_counts', 'context'
]

liar_news = liar_data.dropna(subset=['text']).reset_index(drop=True)
# Filter for "pants-fire" samples only
sample_articles_LIAR = liar_news[liar_news['label'] == "false"].sample(5)
sample_articles_LIAR2 = liar_news[liar_news['label'] == "pants-fire"].sample(5)
sample_articles_LIAR3 = liar_news[liar_news['label'] == "barely-true"].sample(5)
# Combine samples into one dataframe 
combined_samples = pd.concat([sample_articles_LIAR, sample_articles_LIAR2, sample_articles_LIAR3]).reset_index(drop=True)

# To process each batch through the LLM, to avoid having too long prompts that exceed text length
def chunk_list(lst, size):
    return [lst[i:i+size] for i in range(0, len(lst), size)]

texts = combined_samples["text"].tolist()
batches = chunk_list(texts, 5) #sp

# all_analysis will hold the LLM outputs for each batch
all_analysis = []

for batch in batches:
    text_block = ""
    for t in batch:
        short = t[:1500] + "..." if len(t) > 1500 else t
        text_block += f"Text: {short}\n\n"

    prompt = f"""
    <s>[INST] <<SYS>>
    List the suspicious keywords and indicators of fake news from the following collection of articles.
    <</SYS>>
    {text_block}
    [/INST]
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=8192
    ).to("cuda")

    output = model.generate(**inputs, max_new_tokens=800)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    result = output_text.split('[/INST]')[-1].strip() if '[/INST]' in output_text else output_text.strip()

    all_analysis.append(result)

print("\n=== LLM Batch Analysis Results ===\n")
print(all_analysis)

# With regex, extract all phrases within quotes since the model outputs patterns are in quotes
detected_phrases = re.findall(r'"(.*?)"', " ".join(all_analysis))
detected_phrases = list(set(detected_phrases))

print("\n=== Extracted Phrases ===\n", detected_phrases)

print("\nComputing embeddings for semantic similarity...")

# gather all fake and real articles from the HF dataset

fake_articles = liar_news[liar_news["label"].isin(["pants-fire", "false", "barely-true"])
]["text"].tolist()

real_articles = liar_news[
    liar_news["label"].isin(["half-true", "mostly-true", "true"])
]["text"].tolist()

print(f"Fake articles: {len(fake_articles)}, Real articles: {len(real_articles)}")

# embed all texts so we can compute similarity to patterns
fake_embeddings = embedder.encode(fake_articles, convert_to_tensor=True, show_progress_bar=True)
real_embeddings = embedder.encode(real_articles, convert_to_tensor=True, show_progress_bar=True)
pattern_embeddings = embedder.encode(detected_phrases, convert_to_tensor=True)

# to compute similarity of each pattern to all fake and real articles with the HF dataset
fake_sim = cosine_similarity(pattern_embeddings.cpu(), fake_embeddings.cpu())
real_sim = cosine_similarity(pattern_embeddings.cpu(), real_embeddings.cpu())

# calculate mean similarity scores
fake_mean = fake_sim.mean(axis=1)
real_mean = real_sim.mean(axis=1)

# calculate fake_signal as a difference
fake_signal = fake_mean - real_mean  # higher indicates "more fake-like"

# Create a final dataframe with confidence scores for each pattern 
pattern_confidence = pd.DataFrame({
    "pattern": detected_phrases,
    "fake_similarity": np.round(fake_mean * 100, 2),
    "real_similarity": np.round(real_mean * 100, 2),
    "fake_signal": np.round(fake_signal * 100, 2)
}).sort_values(by="fake_signal", ascending=False)

print("\n=== MOST FAKE-LIKE PHRASES ===\n")
print(pattern_confidence.head(20))

# save to CSV file
pattern_confidence.to_csv("fake_news_patterns_confidenceLIARDATASET.csv", index=False)
