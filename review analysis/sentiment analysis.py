import json
import os
import pandas as pd
from transformers import pipeline
from langdetect import detect
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# Initialize sentiment analysis and translation pipelines
sentiment_analyzer = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
translator_en = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")

# Load data
with open('../../data/original data/reviews_testing.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    data = dict(list(data.items())[6722:])

# Lock for thread-safe printing
print_lock = Lock()

# Check existing files to skip already processed parts
# def get_processed_parts(thread_id):
#     processed_parts = set()
#     output_dir = 'sentiment_output'
#     for filename in os.listdir(output_dir):
#         if filename.startswith(f"thread_{thread_id}_part_") and filename.endswith(".json"):
#             part_number = int(filename.split("_part_")[1].split(".")[0])
#             processed_parts.add(part_number)
#     return processed_parts

def translate_to_english(text):
    try:
        lang = detect(text)
        if lang != 'en':
            translated_text = translator_en(text, max_length=512)[0]['translation_text']
            return translated_text
        return text
    except Exception as e:
        with print_lock:
            print(f"Translation error: {e}")
        return None

def analyze_sentiment(text):
    try:
        truncated_text = text[:128]
        result = sentiment_analyzer(truncated_text)
        return result[0]["label"], result[0]["score"]
    except Exception as e:
        with print_lock:
            print(f"Sentiment analysis error: {e}")
        return None, None

def process_properties(property_data, thread_id):
    results = {}
    file_counter = 1
    # processed_parts = get_processed_parts(thread_id)

    # for i, (property_id, reviews) in enumerate(
    #         tqdm(property_data.items(), desc=f"Thread {thread_id}", unit="property")):
        # if file_counter in processed_parts:
        #     if (i + 1) % 10 == 0 or (i + 1) == len(property_data):
        #         file_counter += 1
        #     continue
    for i, (property_id, reviews) in enumerate(
            tqdm(data.items(), desc=f"Thread {thread_id}", unit="property")): # new line

        if pd.isna(reviews):
            results[property_id] = None
        else:
            reviews_list = reviews.split('\n---------------------------------\n')
            sentiment_scores = []
            for review in reviews_list:
                translated_review = translate_to_english(review)
                if translated_review is not None:
                    sentiment_label, sentiment_score = analyze_sentiment(translated_review)
                    sentiment_scores.append({
                        "text": review,
                        "translated_text": translated_review,
                        "sentiment_label": sentiment_label,
                        "sentiment_score": sentiment_score
                    })
            results[property_id] = sentiment_scores if sentiment_scores else None

        if (i + 1) % 10 == 0 or (i + 1) == len(property_data):
            # output_file = f'sentiment_output/thread_{thread_id}_part_{file_counter}.json'
            output_file = '../../data/review data/sentiment_output/test_add.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            with print_lock:
                print(f"Thread {thread_id} saved results to {output_file}")
            results = {}
            file_counter += 1

# Divide data into chunks for each thread based on slicing
total_properties = len(data)
threads = 1
chunk_size = total_properties // threads
data_items = list(data.items())
data_chunks = [dict(data_items[i * chunk_size: (i + 1) * chunk_size]) for i in range(threads)]

# Start threads
with ThreadPoolExecutor(max_workers=threads) as executor:
    futures = [executor.submit(process_properties, data_chunks[thread_id], thread_id + 1) for thread_id in range(threads)]
    for future in futures:
        future.result()

print("All threads have completed processing.")
