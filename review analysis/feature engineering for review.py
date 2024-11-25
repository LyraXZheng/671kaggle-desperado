import json
import pandas as pd
from statistics import mean, stdev
import re
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 加载 Hugging Face 预训练情感分类模型
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# 加载 JSON 数据
json_path = "../../data/review data/review_sentiment.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 定义价格相关关键词
price_keywords = [
    "expensive", "too expensive", "very expensive", "not expensive",
    "really expensive", "cheap", "affordable", "overpriced", "pricey",
    "reasonable price", "high price", "low price"
]
price_keywords_regex = re.compile(r'\b(?:' + '|'.join(re.escape(keyword) for keyword in price_keywords) + r')\b', re.IGNORECASE)

# 定义处理单个房源的函数
def process_property(listing):
    property_features = []
    for property_id, comments in listing.items():
        reviews = []
        sentiment_scores = []
        review_lengths = []
        positive_count = 0
        neutral_count = 0
        negative_count = 0
        recommend_count = 0
        price_mentions = 0
        price_sentiments = {"expensive": 0, "reasonable": 0, "cheap": 0}
        price_contexts = []

        if comments is not None:
            for comment in comments:
                if "translated_text" in comment and comment["translated_text"] is not None:
                    text = comment["translated_text"]
                    reviews.append(comment)
                    review_lengths.append(len(text))

                    # 统计价格关键词提及次数
                    matches = price_keywords_regex.findall(text)
                    price_mentions += len(matches)

                    # 提取价格上下文并进行情感分析
                    for match in matches:
                        match_start = text.lower().find(match.lower())
                        context = text[max(0, match_start - 30):min(len(text), match_start + 30)]
                        price_contexts.append(context)

                        # 情感分类
                        sentiment = sentiment_analyzer(context)
                        if sentiment[0]["label"] == "NEGATIVE":
                            price_sentiments["expensive"] += 1
                        elif sentiment[0]["label"] == "POSITIVE":
                            if "cheap" in match.lower() or "affordable" in match.lower():
                                price_sentiments["cheap"] += 1
                            else:
                                price_sentiments["reasonable"] += 1

                if "sentiment_score" in comment and comment["sentiment_score"] is not None:
                    sentiment_scores.append(comment["sentiment_score"])

                if "sentiment_label" in comment:
                    if comment["sentiment_label"] == "POS":
                        positive_count += 1
                    elif comment["sentiment_label"] == "NEU":
                        neutral_count += 1
                    elif comment["sentiment_label"] == "NEG":
                        negative_count += 1

                if "text" in comment and "recommend" in comment["text"].lower():
                    recommend_count += 1

        if reviews:
            total_reviews = len(reviews)
            average_sentiment_score = mean(sentiment_scores) if sentiment_scores else None
            max_sentiment_score = max(sentiment_scores) if sentiment_scores else None
            min_sentiment_score = min(sentiment_scores) if sentiment_scores else None
            sentiment_score_stddev = stdev(sentiment_scores) if len(sentiment_scores) > 1 else None
            positive_ratio = positive_count / total_reviews if total_reviews else None
            neutral_ratio = neutral_count / total_reviews if total_reviews else None
            negative_ratio = negative_count / total_reviews if total_reviews else None
            average_review_length = mean(review_lengths) if review_lengths else None
        else:
            total_reviews = None
            average_sentiment_score = None
            max_sentiment_score = None
            min_sentiment_score = None
            sentiment_score_stddev = None
            positive_ratio = None
            neutral_ratio = None
            negative_ratio = None
            average_review_length = None
            recommend_count = None
            price_mentions = None
            price_sentiments = {"expensive": None, "reasonable": None, "cheap": None}
            price_contexts = None

        # 保存特征
        property_features.append({
            "property_id": property_id,
            "total_reviews": total_reviews,
            "average_sentiment_score": average_sentiment_score,
            "max_sentiment_score": max_sentiment_score,
            "min_sentiment_score": min_sentiment_score,
            "sentiment_score_stddev": sentiment_score_stddev,
            "positive_ratio": positive_ratio,
            "neutral_ratio": neutral_ratio,
            "negative_ratio": negative_ratio,
            "average_review_length": average_review_length,
            "recommend_count": recommend_count,
            "price_mentions": price_mentions,
            "price_sentiments": price_sentiments,
            "price_contexts": price_contexts
        })
    return property_features

# 使用多线程处理房源
property_features = []
with ThreadPoolExecutor(max_workers=16) as executor:
    futures = {executor.submit(process_property, listing): listing for listing in data}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing listings with multithreading"):
        property_features.extend(future.result())

# 转换为 DataFrame
df = pd.DataFrame(property_features)

# 清理数据逻辑
def clean_data(df):
    # 分解 price_sentiments 列为独立列
    df['price_sentiments_expensive'] = df['price_sentiments'].apply(lambda x: x['expensive'] if isinstance(x, dict) else 0)
    df['price_sentiments_reasonable'] = df['price_sentiments'].apply(lambda x: x['reasonable'] if isinstance(x, dict) else 0)
    df['price_sentiments_cheap'] = df['price_sentiments'].apply(lambda x: x['cheap'] if isinstance(x, dict) else 0)

    # 将 price_contexts 替换为其长度
    df['price_contexts'] = df['price_contexts'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # 删除原始 price_sentiments 列
    df = df.drop(columns=['price_sentiments'])

    # 确保所有数据类型均为 int 或 float
    for column in df.columns:
        if df[column].dtype == 'object':
            # 对于无法转换为数值的列，用 0 填充空值
            df[column] = df[column].fillna(0)
            df[column] = df[column].apply(lambda x: float(x) if isinstance(x, (int, float)) else 0)

    return df

# 清理数据
df_cleaned = clean_data(df)

# 保存清理后的数据
cleaned_file_path = "../../data/important data/review_feature_train_cleaned.csv"
df_cleaned.to_csv(cleaned_file_path, index=False)

print(f"Cleaned data saved to {cleaned_file_path}")
