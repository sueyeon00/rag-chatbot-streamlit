import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from sentence_transformers import CrossEncoder
import tempfile
import os
import dotenv
import PyPDF2
import io
import json
from datetime import datetime
import uuid

# 환경변수 로드
dotenv.load_dotenv()

# Streamlit 페이지 설정
st.set_page_config(
    page_title="🚀 Advanced RAG ChatBot",
    page_icon="🚀",
    layout="wide"
)

# 세션 상태 초기화
if "projects" not in st.session_state:
    st.session_state.projects = {}
if "current_project" not in st.session_state:
    st.session_state.current_project = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

# 프로젝트 관리 함수들
def create_new_project(name, description=""):
    """새 프로젝트 생성"""
    project_id = str(uuid.uuid4())[:8]
    project = {
        "id": project_id,
        "name": name,
        "description": description,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "documents": [],
        "vector_store": None,
        "total_chunks": 0,
        "processed_documents": []  # 실제 처리된 문서 내용 저장
    }
    st.session_state.projects[project_id] = project
    st.session_state.chat_history[project_id] = []
    return project_id

def get_current_project():
    """현재 선택된 프로젝트 반환"""
    if st.session_state.current_project and st.session_state.current_project in st.session_state.projects:
        return st.session_state.projects[st.session_state.current_project]
    return None

# Reranker 설정 (캐시됨)
@st.cache_resource
def load_reranker():
    """BGE reranker 모델 로드"""
    try:
        return CrossEncoder("BAAI/bge-reranker-base")
    except:
        st.warning("⚠️ Reranker 모델 로드 실패. 기본 검색을 사용합니다.")
        return None

# 점수 기반 rerank 함수
def rerank_docs(query, docs, model, top_n=5):
    """문서들을 점수 기반으로 재순위 조정"""
    if model is None:
        return docs[:top_n]
    
    pairs = [[query, doc.page_content] for doc in docs]
    scores = model.predict(pairs)
    reranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for score, doc in reranked[:top_n]]

def process_pdf_file(file_content, file_name):
    """PDF 파일 처리 - 개선된 버전"""
    documents = []
    
    try:
        # PyPDF2로 먼저 시도
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        st.info(f"📄 {file_name}: {len(pdf_reader.pages)}페이지 감지됨")
        
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text.strip():  # 빈 페이지 제외
                doc = Document(
                    page_content=text,
                    metadata={
                        "page": page_num + 1, 
                        "source": file_name, 
                        "file_type": "pdf",
                        "total_pages": len(pdf_reader.pages)
                    }
                )
                documents.append(doc)
                
        st.success(f"✅ {file_name}: {len(documents)}페이지 텍스트 추출 완료")
        return documents
        
    except Exception as e:
        st.warning(f"⚠️ PyPDF2 실패, LangChain 로더로 재시도: {str(e)}")
        
        # 대체 방법: LangChain PyPDFLoader
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name
            
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            
            # 메타데이터 업데이트
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    "source": file_name,
                    "file_type": "pdf",
                    "total_pages": len(documents)
                })
            
            os.unlink(tmp_path)
            st.success(f"✅ {file_name}: LangChain으로 {len(documents)}페이지 처리 완료")
            return documents
            
        except Exception as e2:
            st.error(f"❌ {file_name} PDF 처리 실패: {str(e2)}")
            return []

def process_text_file(file_content, file_name):
    """텍스트 파일 처리 - 개선된 버전"""
    try:
        # 인코딩 자동 감지
        try:
            text = file_content.decode('utf-8')
            encoding_used = 'utf-8'
        except UnicodeDecodeError:
            try:
                text = file_content.decode('cp949')
                encoding_used = 'cp949'
            except UnicodeDecodeError:
                text = file_content.decode('latin-1')
                encoding_used = 'latin-1'
        
        st.info(f"📝 {file_name}: {encoding_used} 인코딩으로 읽기 완료")
        
        doc = Document(
            page_content=text,
            metadata={
                "source": file_name, 
                "file_type": "txt",
                "encoding": encoding_used,
                "char_count": len(text)
            }
        )
        
        st.success(f"✅ {file_name}: {len(text):,}자 텍스트 처리 완료")
        return [doc]
        
    except Exception as e:
        st.error(f"❌ {file_name} 텍스트 처리 실패: {str(e)}")
        return []

def process_multiple_documents(uploaded_files):
    """여러 문서 처리 - 개선된 버전"""
    all_documents = []
    processed_files = []
    
    st.info(f"📚 총 {len(uploaded_files)}개 파일 처리 시작...")
    
    for i, uploaded_file in enumerate(uploaded_files):
        st.write(f"**{i+1}/{len(uploaded_files)}**: {uploaded_file.name} 처리중...")
        
        file_content = uploaded_file.read()
        file_name = uploaded_file.name
        file_type = file_name.split('.')[-1].lower()
        
        try:
            if file_type == 'pdf':
                documents = process_pdf_file(file_content, file_name)
            elif file_type == 'txt':
                documents = process_text_file(file_content, file_name)
            else:
                st.warning(f"⚠️ {file_name}: 지원하지 않는 파일 형식입니다.")
                continue
            
            if documents:  # 문서가 성공적으로 처리된 경우만
                all_documents.extend(documents)
                processed_files.append({
                    "name": file_name,
                    "type": file_type,
                    "size": len(file_content),
                    "pages": len(documents) if file_type == 'pdf' else 1,
                    "doc_count": len(documents)
                })
                
                # 각 문서의 첫 100자 미리보기
                preview = documents[0].page_content[:100] + "..." if len(documents[0].page_content) > 100 else documents[0].page_content
                st.text_area(f"📖 {file_name} 미리보기", preview, height=100, key=f"preview_{i}")
            
        except Exception as e:
            st.error(f"❌ {file_name} 처리 실패: {str(e)}")
    
    st.success(f"🎉 총 {len(all_documents)}개 문서 청크 생성 완료!")
    return all_documents, processed_files

def create_vector_store(documents, project_id, chunk_size=500, chunk_overlap=50):
    """벡터 저장소 생성 - 청크 크기 조정 가능"""
    if not documents:
        return None, 0
    
    st.info(f"🔄 청크 크기 {chunk_size}, 오버랩 {chunk_overlap}로 문서 분할 중...")
    
    # 텍스트 스플리터
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ".", " ", ""]
    )
    
    # 문서 분할
    split_docs = splitter.split_documents(documents)
    
    # 분할 결과 상세 정보
    st.info(f"📊 분할 결과: {len(documents)}개 원본 문서 → {len(split_docs)}개 청크")
    
    # 각 원본 문서별 청크 수 표시
    source_chunks = {}
    for doc in split_docs:
        source = doc.metadata.get('source', 'Unknown')
        source_chunks[source] = source_chunks.get(source, 0) + 1
    
    for source, count in source_chunks.items():
        st.write(f"  - {source}: {count}개 청크")
    
    # 임베딩 생성
    st.info("🔄 임베딩 생성 중...")
    embeddings = OpenAIEmbeddings()
    
    # 벡터 저장소 생성
    vector_store = FAISS.from_documents(embedding=embeddings, documents=split_docs)
    
    st.success("✅ 벡터 저장소 생성 완료!")
    return vector_store, len(split_docs)

def format_docs(docs):
    """문서 포맷팅 - 출처 정보 포함"""
    formatted = []
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', '')
        page_info = f" (페이지 {page})" if page else ""
        
        formatted.append(f"[출처: {source}{page_info}]\n{doc.page_content}")
    
    return "\n\n---\n\n".join(formatted)

# 메인 UI
st.title("🚀 Advanced RAG ChatBot")
st.markdown("**여러 문서 업로드 + 프로젝트별 관리 + 스마트 채팅**")

# 사이드바 - 프로젝트 관리
with st.sidebar:
    st.header("📁 프로젝트 관리")
    
    # 새 프로젝트 생성
    with st.expander("➕ 새 프로젝트 만들기"):
        new_project_name = st.text_input("프로젝트 이름", placeholder="예: AI 트렌드 분석")
        new_project_desc = st.text_area("설명 (선택)", placeholder="이 프로젝트에 대한 간단한 설명")
        
        if st.button("프로젝트 생성", type="primary"):
            if new_project_name:
                project_id = create_new_project(new_project_name, new_project_desc)
                st.session_state.current_project = project_id
                st.success(f"✅ '{new_project_name}' 프로젝트가 생성되었습니다!")
                st.rerun()
            else:
                st.error("프로젝트 이름을 입력해주세요.")
    
    # 기존 프로젝트 목록
    st.subheader("📋 프로젝트 목록")
    
    if st.session_state.projects:
        for project_id, project in st.session_state.projects.items():
            is_current = project_id == st.session_state.current_project
            
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if st.button(
                        f"{'🟢' if is_current else '⚪'} {project['name']}", 
                        key=f"select_{project_id}",
                        help=f"생성일: {project['created_at']}\n문서 수: {len(project['documents'])}",
                        use_container_width=True
                    ):
                        st.session_state.current_project = project_id
                        st.rerun()
                
                with col2:
                    if st.button("🗑️", key=f"delete_{project_id}", help="프로젝트 삭제"):
                        del st.session_state.projects[project_id]
                        del st.session_state.chat_history[project_id]
                        if st.session_state.current_project == project_id:
                            st.session_state.current_project = None
                        st.rerun()
    else:
        st.info("프로젝트가 없습니다. 새 프로젝트를 만들어보세요!")
    
    # 현재 프로젝트 정보
    current_project = get_current_project()
    if current_project:
        st.markdown("---")
        st.subheader("📊 현재 프로젝트")
        st.write(f"**이름:** {current_project['name']}")
        if current_project['description']:
            st.write(f"**설명:** {current_project['description']}")
        st.write(f"**문서 수:** {len(current_project['documents'])}")
        st.write(f"**청크 수:** {current_project['total_chunks']}")
        st.write(f"**생성일:** {current_project['created_at']}")

# 메인 영역
if not current_project:
    st.info("👈 먼저 프로젝트를 선택하거나 새로 만들어주세요!")
else:
    # 탭으로 구성
    tab1, tab2, tab3 = st.tabs(["📤 문서 업로드", "💬 채팅", "📊 프로젝트 상세"])
    
    with tab1:
        st.subheader("📤 문서 업로드")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                "문서를 선택하세요 (PDF, TXT)",
                type=['pdf', 'txt'],
                accept_multiple_files=True,
                help="여러 파일을 동시에 업로드할 수 있습니다."
            )
        
        with col2:
            # LLM 설정
            st.subheader("⚙️ AI 설정")
            model_name = st.selectbox(
                "모델 선택",
                ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                index=0
            )
            temperature = st.slider("창의성", 0.0, 1.0, 0.1, 0.1)
            
            # 청크 크기 설정
            st.subheader("📏 청크 설정")
            chunk_size = st.slider(
                "청크 크기", 
                min_value=100, 
                max_value=2000, 
                value=500, 
                step=50,
                help="청크 크기가 작을수록 더 정확한 검색이 가능하지만 문맥이 단절될 수 있습니다. 큰 청크는 더 많은 문맥을 포함하지만 노이즈가 증가할 수 있습니다."
            )
            
            chunk_overlap = st.slider(
                "청크 오버랩", 
                min_value=0, 
                max_value=200, 
                value=50, 
                step=10,
                help="청크 간 겹치는 부분입니다. 높을수록 문맥 연결이 좋아지지만 중복 정보가 증가합니다."
            )
            
            # 추가 설정
            st.subheader("🔧 고급 설정")
            use_reranker = st.checkbox("🔄 Reranker 사용", value=False, help="검색 결과 재순위 조정으로 정확도 향상")
            use_compression = st.checkbox("🗜️ 문서 압축", value=False, help="LLM 기반 문서 압축으로 핵심 내용만 추출")
        
        # 청크 크기 가이드
        st.info(f"""
        **📏 현재 청크 설정**
        - **청크 크기**: {chunk_size}자 
        - **오버랩**: {chunk_overlap}자
        
        **💡 청크 크기 가이드**
        - **100-300자**: 매우 정확한 검색, 짧은 답변에 적합
        - **300-500자**: 균형잡힌 설정, 대부분의 용도에 적합 
        - **500-1000자**: 긴 문맥 유지, 복잡한 질문에 적합
        - **1000자+**: 매우 긴 문맥, 요약이나 전체적인 이해에 적합
        """)
        
        if uploaded_files:
            if st.button("📚 문서 처리하기", type="primary"):
                with st.spinner("📖 문서들을 분석하고 있습니다..."):
                    try:
                        documents, processed_files = process_multiple_documents(uploaded_files)
                        
                        if documents:
                            vector_store, total_chunks = create_vector_store(
                                documents, 
                                current_project['id'],
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap
                            )
                            
                            # 프로젝트 업데이트
                            current_project['documents'].extend(processed_files)
                            current_project['vector_store'] = vector_store
                            current_project['total_chunks'] = total_chunks
                            current_project['processed_documents'] = documents  # 원본 문서 저장
                            current_project['chunk_settings'] = {
                                'chunk_size': chunk_size,
                                'chunk_overlap': chunk_overlap
                            }
                            
                            st.success(f"✅ {len(processed_files)}개 문서 처리 완료!")
                            
                            # 처리된 파일 정보 표시
                            for file_info in processed_files:
                                st.write(f"- **{file_info['name']}** ({file_info['type'].upper()}) - {file_info['size']:,} bytes - {file_info['doc_count']}개 청크")
                        else:
                            st.error("처리할 수 있는 문서가 없습니다.")
                            
                    except Exception as e:
                        st.error(f"❌ 문서 처리 중 오류: {str(e)}")
                        st.exception(e)  # 디버깅용
    
    with tab2:
        st.subheader("💬 AI와 채팅하기")
        
        if current_project['vector_store'] is None:
            st.warning("📤 먼저 문서를 업로드하고 처리해주세요!")
        else:
            # LLM 초기화
            llm = ChatOpenAI(model=model_name, temperature=temperature)
            
            # 검색 설정 표시
            if 'chunk_settings' in current_project:
                settings = current_project['chunk_settings']
                st.info(f"🔧 현재 설정: 청크 크기 {settings['chunk_size']}자, 오버랩 {settings['chunk_overlap']}자")
            
            # 기본 리트리버 (더 많은 문서 검색)
            base_retriever = current_project['vector_store'].as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 15}  # 더 많은 문서 검색
            )
            
            # 고급 검색 설정
            retriever = base_retriever
            
            # 1. 문서 압축 옵션
            if use_compression:
                compressor = LLMChainExtractor.from_llm(llm)
                retriever = ContextualCompressionRetriever(
                    base_retriever=base_retriever,
                    base_compressor=compressor
                )
                st.info("🗜️ LLM 기반 문서 압축이 활성화되었습니다.")
            
            # 2. Reranker 옵션
            reranker_model = None
            if use_reranker:
                with st.spinner("🔄 Reranker 모델을 로드하고 있습니다..."):
                    reranker_model = load_reranker()
                if reranker_model:
                    st.info("🔄 BGE Reranker가 활성화되었습니다.")
            
            # 프롬프트 템플릿
            prompt = ChatPromptTemplate([
                ("system", 
                 "문서: {context}\n\n"
                 "당신은 업로드된 문서들의 전문가입니다. "
                 "항상 문서 내용을 기반으로 정확하고 도움이 되는 답변을 제공하세요.\n"
                 "문서에서 답변을 찾을 수 없는 경우 '문서에서 관련 정보를 찾을 수 없습니다'라고 답변하세요.\n"
                 "답변할 때는 어느 문서에서 가져온 정보인지도 언급해주세요.\n"
                 "여러 문서에서 정보를 찾은 경우 모든 출처를 명시하세요."), 
                ("user", "{query}")
            ])
            
            # RAG 체인 생성 (Reranker 포함)
            def enhanced_retrieval(query):
                docs = retriever.invoke(query)
                if reranker_model and docs:
                    docs = rerank_docs(query, docs, reranker_model, top_n=10)
                return format_docs(docs)
            
            chain = {
                "context": RunnableLambda(enhanced_retrieval),
                "query": RunnablePassthrough()
            } | prompt | llm
            
            # 채팅 기록 표시
            chat_history = st.session_state.chat_history[current_project['id']]
            
            for message in chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # 새로운 질문 입력
            if query := st.chat_input("문서에 대해 질문해보세요!"):
                # 사용자 메시지 추가
                chat_history.append({"role": "user", "content": query})
                with st.chat_message("user"):
                    st.markdown(query)
                
                # AI 응답 생성
                with st.chat_message("assistant"):
                    with st.spinner("🤔 답변을 생성하고 있습니다..."):
                        try:
                            # 디버깅: 검색된 문서 표시
                            retrieved_docs = current_project['vector_store'].similarity_search(query, k=5)
                            
                            with st.expander("🔍 검색된 문서 (디버깅)"):
                                for i, doc in enumerate(retrieved_docs):
                                    st.write(f"**문서 {i+1}**: {doc.metadata.get('source', 'Unknown')}")
                                    st.write(f"**내용**: {doc.page_content[:200]}...")
                                    st.write("---")
                            
                            response = chain.invoke(query)
                            answer = response.content
                            st.markdown(answer)
                            
                            # 어시스턴트 메시지 추가
                            chat_history.append({"role": "assistant", "content": answer})
                            
                        except Exception as e:
                            error_msg = f"죄송합니다. 오류가 발생했습니다: {str(e)}"
                            st.error(error_msg)
                            st.exception(e)  # 디버깅용
                            chat_history.append({"role": "assistant", "content": error_msg})
            
            # 채팅 초기화 버튼
            if st.button("🗑️ 채팅 기록 삭제"):
                st.session_state.chat_history[current_project['id']] = []
                st.rerun()
    
    with tab3:
        st.subheader("📊 프로젝트 상세 정보")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("총 문서 수", len(current_project['documents']))
            st.metric("총 청크 수", current_project['total_chunks'])
        
        with col2:
            total_chat = len(st.session_state.chat_history[current_project['id']])
            st.metric("채팅 메시지 수", total_chat)
            st.metric("프로젝트 ID", current_project['id'])
        
        # 청크 설정 정보
        if 'chunk_settings' in current_project:
            settings = current_project['chunk_settings']
            st.subheader("📏 청크 설정")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("청크 크기", f"{settings['chunk_size']}자")
            with col2:
                st.metric("청크 오버랩", f"{settings['chunk_overlap']}자")
        
        # 업로드된 문서 목록
        if current_project['documents']:
            st.subheader("📄 업로드된 문서들")
            for doc in current_project['documents']:
                with st.expander(f"📎 {doc['name']}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**타입:** {doc['type'].upper()}")
                    with col2:
                        st.write(f"**크기:** {doc['size']:,} bytes")
                    with col3:
                        st.write(f"**청크 수:** {doc['doc_count']}")
                        if doc['type'] == 'pdf':
                            st.write(f"**페이지:** {doc['pages']}")
        
        # 벡터 저장소 테스트
        if current_project['vector_store'] is not None:
            st.subheader("🔍 벡터 저장소 테스트")
            test_query = st.text_input("테스트 검색어를 입력하세요:")
            if test_query:
                with st.spinner("검색 중..."):
                    results = current_project['vector_store'].similarity_search(test_query, k=3)
                    st.write(f"**검색 결과 ({len(results)}개):**")
                    for i, doc in enumerate(results):
                        st.write(f"**{i+1}. {doc.metadata.get('source', 'Unknown')}**")
                        st.write(f"내용: {doc.page_content[:300]}...")
                        st.write("---")

# 하단 정보
st.markdown("---")
st.markdown("🚀 **Advanced RAG ChatBot** - LangChain + Streamlit + FAISS")