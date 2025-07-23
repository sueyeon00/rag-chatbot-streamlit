# 🚀 RAG Chatbot with Streamlit

문서를 업로드하고 자유롭게 질문할 수 있는 문서 기반 챗봇입니다.  
OpenAI의 LLM과 FAISS 벡터스토어, LangChain을 활용하여 RAG(Retrieval-Augmented Generation) 기능을 구현했습니다.

---

## 📸 주요 기능

### 🖥️ 초기화면

<img width="100%" alt="초기화면" src="https://github.com/user-attachments/assets/2e315920-9210-4147-8b5c-d1ab4d18a97e" />

### 1️⃣ 프로젝트 생성

사용자는 새로운 프로젝트를 생성하고 문서 기반 챗봇을 구성할 수 있습니다.

### 2️⃣ 문서 업로드

학습하고 싶은 PDF, TXT 파일을 업로드합니다.  
<img width="100%" alt="파일 업로드" src="https://github.com/user-attachments/assets/27a01dee-15b4-4128-8f96-2bea26a75f15" />

### 3️⃣ 모델 및 청크 설정

OpenAI 모델, 청크 크기, 오버랩, 리랭커 등 다양한 옵션을 설정할 수 있습니다.

### 4️⃣ 채팅

업로드한 문서를 기반으로 자유롭게 질문하고 답변을 받을 수 있습니다.  
<img width="100%" alt="채팅 화면" src="https://github.com/user-attachments/assets/480f1ac0-c9fd-4d02-851f-7c57413734d5" />

---

## 🚀 실행 방법

1. `.env` 파일에 OpenAI API 키를 설정합니다.
2. 패키지 설치 : pip install -r requirements.txt
3. 실행 : streamlit run app.py

--

## 📌 관련 포스팅

https://sueeee-e.tistory.com/70
