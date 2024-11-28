import streamlit as st
import openai
import faiss
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
import time

# 환경 변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")

# FAISS 및 데이터 로드
faiss_index_path = "./faiss_index.bin"
csv_data_path = "./reviews_embeddings.csv"

index = faiss.read_index(faiss_index_path)
metadata = pd.read_csv(csv_data_path)

def get_embedding_with_retry(text, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response['data'][0]['embedding']
        except openai.error.APIConnectionError as e:
            st.warning(f"OpenAI API 연결 오류: {e}. 재시도 중 ({attempt + 1}/{max_retries})...")
            time.sleep(2 ** attempt)
    raise Exception("OpenAI API 호출 실패")

def get_location(name, address, max_retries=3):
    for attempt in range(max_retries):
        try:
            url = f"https://maps.googleapis.com/maps/api/geocode/json?address={name},+{address}&key={google_maps_api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data['status'] == 'OK' and data['results']:
                location = data['results'][0]['geometry']['location']
                return location['lat'], location['lng']
        except requests.exceptions.RequestException:
            time.sleep(2)
    return None, None

st.title("장소 추천 및 지도 표시")
user_input = st.text_input("검색어를 입력하세요", placeholder="장소를 입력하세요.")

if user_input:
    st.write("임베딩 생성 중...")
    query_embedding = np.array(get_embedding_with_retry(user_input)).astype('float32').reshape(1, -1)

    st.write("유사도 계산 중...")
    distances, indices = index.search(query_embedding, k=5)
    results = metadata.iloc[indices[0]].copy()
    results['similarity'] = 1 - distances[0] / 2

    st.write("추천된 장소:")
    st.dataframe(results[['name', 'address', 'review_text', 'similarity']])

    locations = []
    for _, row in results.iterrows():
        lat, lng = get_location(row['name'], row['address'])
        if lat and lng:
            locations.append({
                "name": row['name'],
                "address": row['address'],
                "review_text": row['review_text'],
                "similarity": row['similarity'],
                "latitude": lat,
                "longitude": lng
            })

    if locations:
        html_code = """
        <!DOCTYPE html>
        <html>
        ...
        </html>
        """
        st.components.v1.html(html_code, height=600)
