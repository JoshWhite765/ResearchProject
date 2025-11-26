
# SETUP AND IMPORTS
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

#Mistral fake-news LoRA
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
peft_model_name = "bpavlsh/Mistral-Fake-News-Detection"

tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(base_model, peft_model_name)


# # LOAD LLaMA 2-13B
# model_id = "meta-llama/Llama-2-13b-chat-hf"

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     device_map="auto",      # auto-assign layers to GPU/CPU
#     load_in_4bit=True,      # 4-bit quantization for 12GB GPU
#     torch_dtype=torch.float16,
#     trust_remote_code=True
# )
# model.eval()


# Embedding model for semantic matching
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# Load HF DATASET
splits = {'train': 'train.tsv'}
liarData = pd.read_csv(
    "hf://datasets/ErfanMoosaviMonazzah/fake-news-detection-dataset-English/" + splits["train"],
    sep="\t"
)
liarData.columns = ['unnamed', 'title', 'text', 'subject', 'date', 'label']
hf_news = liarData.dropna(subset=['text']).reset_index(drop=True) # drop rows with missing statements

# Load LIAR DATASET
filepath = "/home/tacticrabbit/LIARDATA.tsv" # file path to LIAR dataset
liar_data = pd.read_csv(filepath, sep='\t', header=None)

liar_data.columns = [
    'id', 'label', 'text', 'subject', 'speaker', 'job_title', 
    'state_info', 'party', 'barely_true_counts', 'false_counts', 
    'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 
    'context'
]

# Drop rows with missing statements
liar_news = liar_data.dropna(subset=['text']).reset_index(drop=True)
print(liar_data.head())


# take 20 samples from LIAR dataset with label "pants-fire"
sample_articles_LIAR = liar_news[liar_news['label'] == "pants-fire"].sample(5) 
# take 20 samples from  dataset with label "pants-fire"
sample_articles_HF = hf_news[hf_news['label'] == 0].sample(5)  # 0 = fake news

# Combine the 20 samples from both datasets
combined_samples = pd.concat([sample_articles_LIAR, sample_articles_HF]).reset_index(drop=True)



# for i, row in sample_articles.iterrows():
#     combined_text += f"Speaker: {row['speaker']}\nSubject: {row['subject']}\nText: {row['text'][:1000]}...\n\n"
#start building the combined text

combined_text = ""
for i, row in combined_samples.iterrows():
    combined_text += f"Text: {row['text'][:1500]}...\n\n" # take first 1000 characters of text

# for i, row in sample_articles.iterrows():
#     combined_text += f"Title: {row['title']}\nSubject: {row['subject']}\nText: {row['text'][:1000]}...\n\n"

# DEFINE KNOWN FAKE NEWS KEYWORDS

# Not yet used, but can be used to guide LLM or filter results
keywords = [
    "clickbait", "misinformation", "propaganda", "fake news", "hoax", "manipulation",
    "deep state", "conspiracy", "breaking", "shocking", "cover-up", "truth bomb",
    "exposed", "banned video", "globalist agenda", "massive scandal", "wake up America"
]


# LLM ANALYSIS STEP
# prompt = f"""<s>[INST] <<SYS>>
# You are an expert in analyzing fake news and propaganda. Identify manipulative patterns, biased language, and recurring fake-news indicators. List the main suspicious keywords, phrases, and topics.
# <</SYS>>
# Analyze the following articles:
# {combined_text}
# [/INST]
# """

prompt = f"""<s>[INST] <<SYS>>
List the main suspicious keywords/phrases that indicate fake news.
<</SYS>>
Analyze the following articles:
{combined_text}
[/INST]
"""

print (prompt)
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to("cuda")
output = model.generate(**inputs, max_new_tokens=1500)
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
analysis_result = output_text.split('[/INST]')[-1].strip()

print("\n=== LLM Analysis Result ===\n")
print(analysis_result)

# # Simple pattern extraction from model output
# detected_phrases = list(set(re.findall(r"\b[A-Za-z][A-Za-z ]{3,}\b", analysis_result.lower()))) # coverts to lowercase and extracts phrases

# # Keep only short phrases (5 words or fewer)
# filtered_phrases = []
# for phrase in detected_phrases:
#     word_count = len(phrase.split())
#     if word_count <= 5:
#         filtered_phrases.append(phrase)

# detected_phrases = filtered_phrases

# Extract phrases enclosed in double quotes
detected_phrases = re.findall(r'"(.*?)"', analysis_result)  # non-greedy match between quotes

# Optional: remove duplicates
detected_phrases = list(set(detected_phrases))



print("\n=== Extracted Phrases ===\n", detected_phrases)
print("\nComputing embeddings for semantic similarity...")

# Embed dataset text and extracted phrases
dataset_embeddings = embedder.encode(liar_news['text'].astype(str).tolist() and hf_news['text'].astype(str).tolist(), convert_to_tensor=True, show_progress_bar=True)
pattern_embeddings = embedder.encode(detected_phrases, convert_to_tensor=True)

# Compute cosine similarity
similarity_scores = cosine_similarity(pattern_embeddings.cpu(), dataset_embeddings.cpu())

# Average similarity per pattern
mean_scores = similarity_scores.mean(axis=1)

# create a DataFrame to show patterns with their confidence scores
pattern_confidence = pd.DataFrame({
    "pattern": detected_phrases,
    "confidence_score": np.round(mean_scores * 100, 2) # round 2 decimal places of a percentage for confdence score
}).sort_values(by="confidence_score", ascending=False)

print("\nHigh Confidence Fake-News Patterns\n")
print(pattern_confidence.head(20))


pattern_confidence.to_csv("fake_news_patterns_confidence.csv", index=False)
