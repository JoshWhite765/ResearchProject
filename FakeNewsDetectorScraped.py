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


# Only If i want to use the LLaMA instead of Mistral model, uncomment below
########
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
########


# Mistral fake-news model
########
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
# peft_model_name = "bpavlsh/Mistral-Fake-News-Detection"

tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)
# model = PeftModel.from_pretrained(base_model, peft_model_name)
########

# Sentence-transformer embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

filepath = "/home/tacticrabbit/Datacollectingscripts/healthfeedback_claims4.csv"
sciencefeed = pd.read_csv(filepath, sep=',', header=None)
sciencefeed.columns = ['title', 'text', 'label', 'url']

sciencefeed = sciencefeed.dropna(subset=['text']).reset_index(drop=True)


# Filter for Fake samples only (SAMPLE FAKE (0) AND REAL (1))
sample_fake = sciencefeed[sciencefeed['label'] == 'false'].sample(20)
# Combine samples into one dataframe 
combined_samples = pd.concat([sample_fake]).reset_index(drop=True)

# To process each batch through the LLM, to avoid having too long prompts that exceed text length
def chunk_list(lst, size):
    return [lst[i:i+size] for i in range(0, len(lst), size)]

texts = combined_samples["text"].tolist() 
batches = chunk_list(texts, 5) # process 5 articles at a time

# all_analysis will hold the LLM outputs for each batch
all_analysis = []

# Process each batch through the LLM as this helps avoid exceeding max token limits
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

    # this generates the output from the model and decodes it to text format for pattern extraction
    output = base_model.generate(**inputs, max_new_tokens=800)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    result = output_text.split('[/INST]')[-1].strip() if '[/INST]' in output_text else output_text.strip() # trim the response so we only have the model's output not the prompt

    all_analysis.append(result)

print("\n=== LLM Batch Analysis Results ===\n")
print(all_analysis)

# With regex, extract all phrases within quotes since the model outputs patterns are in quotes
detected_phrases = re.findall(r'"(.*?)"', " ".join(all_analysis)) 
detected_phrases = list(set(detected_phrases)) # this attempts to remove duplicates by converting to a set and back to list

print("\n=== Extracted Phrases ===\n", detected_phrases)

print("\nComputing embeddings for semantic similarity...")

# gather all fake articles from the Scraped dataset and the real articles from the ISOT dataset

# #Get real news samples from ISOT dataset
# filepath = "/home/tacticrabbit/ISOTTrue.csv"
# isot_dataTRUE = pd.read_csv(filepath, sep=',', header=None)
# isot_dataTRUE.columns = [
# 'title','text','subject', 'date']
filepath2 = "/home/tacticrabbit/MyOwnFakeAndReal.csv"
myOwn_dataTRUE = pd.read_csv(filepath2, sep=',')
myOwn_dataTRUE.columns = [
'id','headline','text', 'category', 'label']


print(myOwn_dataTRUE["label"].unique())


fake_articles = sciencefeed[sciencefeed["label"] == 'false']["text"].astype(str).tolist()
# this gets real articles from my own dataset
real_articles = myOwn_dataTRUE[myOwn_dataTRUE["label"] == 'real']["text"].astype(str).tolist()
print(f"Fake articles: {len(fake_articles)}, Real articles: {len(real_articles)}")

# embed all texts so we can compute similarity scores
# 
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
pattern_confidence.to_csv("fake_newsSCRAPED.csv", index=False)
