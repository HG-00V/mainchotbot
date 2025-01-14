import os
import streamlit as st
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import ChatMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI

# 프롬프트 템플릿
RAG_PROMPT_TEMPLATE = """당신은 조선소 해양플랜트 설계전문가입니다. 검색된 다음 문맥을 사용하여 질문에 답하세요. 답을 모른다면 모른다고 답변하세요. 항상 한글로 답변하세요.
Question: {question} 
Context: {context} 
Answer:"""

st.set_page_config(page_title="삼성중공업 해양설계 OpenAI 모델 테스트", page_icon="💬")

def set_page_style():
    st.markdown(
        """
        <style>
        /* 사이드바 스타일 */
        [data-testid="stSidebar"] {
            background-color: skyblue; /* 사이드바 전체 배경색 설정 */
            position: relative;
        }
        [data-testid="stSidebar"]::before {
            content: "";
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 30%; /* 사이드바 아래쪽 절반에 이미지 적용 */
            background-image: url("https://thumb.mt.co.kr/06/2024/04/2024041114245137762_1.jpg/dims/optimize/");
            background-size: cover;
            background-position: center bottom;
            background-repeat: no-repeat;
            opacity: 1.0; /* 이미지 투명도 조절 (0.0 ~ 1.0) */
            z-index: 0; /* 배경을 콘텐츠 뒤로 보냄 */
        }
        [data-testid="stSidebar"] > div:first-child {
            position: relative;
            z-index: 1; /* 사이드바 콘텐츠를 배경 위로 올림 */
        }
        /* 사이드바의 스크롤바 스타일링 (선택적) */
        [data-testid="stSidebar"] ::-webkit-scrollbar {
            width: 8px;
        }
        [data-testid="stSidebar"] ::-webkit-scrollbar-track {
            background-color: #f0f0f0;
        }
        [data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
            background-color: #888;
            border-radius: 4px;
        }

        /* 채팅 영역 배경색 설정 */
        .stApp {
            background-color: #e6f2ff;         
        }
        /* 채팅 메시지 배경 스타일 (선택적) */
        .stChatMessage {
            background-color: white;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }     
        /* 제목 글자 스타일 */
        .stApp > header > h1 {
            font-weight: bold; /* 제목 글자 굵게 */
            color: #000000 !important; /* 제목 글자 색상 */
        }
        /* 모든 텍스트 요소에 대한 글자색 지정 */
        .stMarkdown, .stText, p, span, h1, h2, h3, h4, h5, h6 {
            color: #000000 !important;
        }
        
        /* 모바일 환경을 위한 미디어 쿼리 */
        @media (max-width: 768px) {
            .stApp, .stMarkdown, .stText, p, span, h1, h2, h3, h4, h5, h6,
            .stApp > header > div > div > div > a > h1,
            .stApp > header > div > div > div > h1 {
                color: #000000 !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# 배경 이미지 설정 함수 호출
set_page_style()

st.markdown("<h1>삼성중공업 해양설계1팀<br>OpenAI 모델 테스트</h1>", unsafe_allow_html=True)

# 사이드바 내용
with st.sidebar:
    st.header("설정")
    api_key = st.text_input("OpenAI API 키를 입력하세요", type="password")
    file = st.file_uploader("파일 업로드", type=["pdf", "txt", "docx"])

# API 키 저장
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    st.sidebar.success("API 키가 설정되었습니다!")

# 필수 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""],
        length_function=len,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=text_splitter)

    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, embedding=cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(role="assistant", content="무엇을 도와드릴까요?")
    ]

def print_history():
    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)

def add_history(role, content):
    st.session_state.messages.append(ChatMessage(role=role, content=content))

print_history()

if user_input := st.chat_input():
    if not api_key:
        st.error("OpenAI API 키를 입력해주세요.")
    else:
        add_history("user", user_input)
        st.chat_message("user").write(user_input)
        with st.chat_message("assistant"):
            try:
                # OpenAI 모델 설정
                openai_model = ChatOpenAI(
                    model_name="gpt-4o-mini",  # 또는 원하는 모델
                    streaming=True,
                )
                chat_container = st.empty()
                if file is not None:
                    retriever = embed_file(file)
                    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
                    rag_chain = (
                        {
                            "context": retriever | format_docs,
                            "question": RunnablePassthrough(),
                        }
                        | prompt
                        | openai_model
                        | StrOutputParser()
                    )
                    answer = rag_chain.stream(user_input)
                else:
                    prompt = ChatPromptTemplate.from_template(
                        "당신은 조선소 해양플랜트 설계전문가답게 다음의 질문에 간결하게 답변해 주세요:\n{input}"
                    )
                    chain = prompt | openai_model | StrOutputParser()
                    answer = chain.stream(user_input)

                full_response = ""
                for chunk in answer:
                    full_response += chunk
                    chat_container.markdown(full_response)
                add_history("assistant", full_response)
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")
                if "Incorrect API key provided" in str(e):
                    st.error("잘못된 API 키입니다. 올바른 OpenAI API 키를 입력해주세요.")
                elif "You exceeded your current quota" in str(e):
                    st.error("API 사용량 한도를 초과했습니다. API 사용량을 확인해주세요.")
                else:
                    st.error("알 수 없는 오류가 발생했습니다. API 키와 네트워크 연결을 확인해주세요.")