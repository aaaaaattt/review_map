import streamlit as st
import faiss
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
import time
from openai import OpenAI

# .env 파일 로드
load_dotenv()

# Secrets에서 API 키 가져오기
openai_api_key = st.secrets["OPENAI_API_KEY"]
google_maps_api_key = st.secrets["GOOGLE_MAPS_API_KEY"]

# FAISS 및 데이터 로드
faiss_index_path = "./faiss_index.bin"  # 저장된 FAISS 파일 경로
csv_data_path = "./reviews_embeddings.csv"  # 메타데이터 CSV 파일 경로

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=openai_api_key)

def get_embedding(text):
    """텍스트의 임베딩을 생성하는 함수"""
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"임베딩 생성 중 오류 발생: {e}")
        return None

# FAISS 인덱스와 메타데이터 로드
index = faiss.read_index(faiss_index_path)
metadata = pd.read_csv(csv_data_path)

# Google Maps에서 위치 정보 가져오기
def get_location(name, address, max_retries=3):
    """장소의 위도와 경도를 가져오는 함수"""
    for attempt in range(max_retries):
        try:
            url = f"https://maps.googleapis.com/maps/api/geocode/json?address={name},+{address}&key={google_maps_api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data['status'] == 'OK' and data['results']:
                location = data['results'][0]['geometry']['location']
                return location['lat'], location['lng']
            elif data['status'] == 'ZERO_RESULTS':
                st.warning(f"위치를 찾을 수 없습니다: {name}, {address}")
            else:
                st.error(f"Google Maps API 오류: {data['status']}")
            return None, None
        except requests.exceptions.RequestException as e:
            st.error(f"네트워크 오류 발생 (시도 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                st.error("위치 정보를 가져오는 데 실패했습니다.")
                return None, None

def main():
    """Streamlit 애플리케이션의 메인 함수"""
    st.title("장소 추천 및 지도 표시 서비스")
    user_input = st.text_input("검색어를 입력하세요", placeholder="찾는 장소를 입력하세요.")

    if user_input:
        query_embedding = np.array(get_embedding(user_input)).astype('float32').reshape(1, -1)
        
        st.write("유사도 계산 중...")
        distances, indices = index.search(query_embedding, k=5)  # 상위 5개 결과 반환
        
        # 상위 결과 추출
        results = metadata.iloc[indices[0]].copy()
        results['similarity'] = 1 - distances[0] / 2  # 코사인 유사도 계산 (1 - L2 거리 / 2)

        # 추천된 장소 및 리뷰 표시
        st.write("추천된 장소 및 리뷰:")
        st.dataframe(results[['name', 'address', 'review_text', 'similarity']])

        st.write("**Google Maps 동적 지도**")
        
        # 위치 정보와 유사도를 정규화하여 JSON 형태로 변환
        locations = []
        max_similarity = results['similarity'].max()  # 최대 유사도 계산
        min_similarity = results['similarity'].min()  # 최소 유사도 계산

        for _, row in results.iterrows():
            lat, lng = get_location(row['name'], row['address'])
            if lat and lng:
                # 유사도를 0~1로 정규화
                normalized_similarity = (row['similarity'] - min_similarity) / (max_similarity - min_similarity)
                
                locations.append({
                    "name": row['name'],
                    "address": row['address'],
                    "review_text": row['review_text'],
                    "similarity": normalized_similarity,  # 정규화된 유사도
                    "latitude": lat,
                    "longitude": lng
                })

        # 위치 정보가 있는 경우에만 지도 생성
        if locations:
            html_code = f"""
            <!DOCTYPE html>
            <html>
              <head>
                <title>추천된 장소 지도</title>
                <script async defer src="https://maps.googleapis.com/maps/api/js?key={google_maps_api_key}&callback=initMap"></script>
                <script>
                  function initMap() {{
                    const map = new google.maps.Map(document.getElementById('map'), {{
                      zoom: 12,
                      center: {{ lat: 37.5665, lng: 126.9780 }}  // 서울 중심 좌표
                    }});

                    const bounds = new google.maps.LatLngBounds();
                    const locations = {locations};

                    locations.forEach((location) => {{
                      if (location.latitude && location.longitude) {{
                        const red = Math.floor((1 - location.similarity) * 255);
                        const green = Math.floor(location.similarity * 255);
                        const color = `rgb(${red}, ${green}, 0)`;

                        const size = 10 + location.similarity * 40;

                        const marker = new google.maps.Marker({{
                          position: {{ lat: location.latitude, lng: location.longitude }},
                          map: map,
                          title: location.name,
                          icon: {{
                            path: google.maps.SymbolPath.CIRCLE,
                            scale: size,
                            fillColor: color,
                            fillOpacity: 0.9,
                            strokeWeight: 1,
                            strokeColor: "#000"
                          }}
                        }});

                        const infoWindow = new google.maps.InfoWindow({{
                          content: `
                            <div style="max-width: 200px;">
                              <h3>${location.name}</h3>
                              <p>주소: ${location.address}</p>
                              <p>리뷰: ${location.review_text}</p>
                              <p>유사도: ${(location.similarity * 100).toFixed(2)}%</p>
                            </div>`
                        }});

                        marker.addListener('click', () => {{
                          infoWindow.open(map, marker);
                        }});

                        bounds.extend(marker.position);
                      }}
                    }});

                    map.fitBounds(bounds);
                  }}
                </script>
              </head>
              <body>
                <div id="map" style="width: 100%; height: 500px;"></div>
              </body>
            </html>
            """
            st.components.v1.html(html_code, height=600)
        else:
            st.warning("선택된 장소들의 위치 정보를 가져올 수 없습니다.")

if __name__ == "__main__":
    main()