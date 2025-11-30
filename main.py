import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
import dotenv

# 1. 載入環境變數
dotenv.load_dotenv()

# 2. 初始化 Gemini Client
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    # 如果找不到金鑰，拋出錯誤，讓應用程式啟動失敗
    raise ValueError("GEMINI_API_KEY 環境變數遺失。請檢查 .env 檔案或環境設定。")

# 建立 Client 實例
# 註: 如果您使用的方法一是 genai.Client()，則這裡可以省略 api_key 參數
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    raise RuntimeError(f"無法初始化 Gemini Client: {e}")

# 3. 初始化 FastAPI
app = FastAPI(
    title="Gemini Search API",
    description="一個用於利用 Gemini 2.5 Flash 和 Google Search Grounding 的 API 服務。",
    version="1.0.0"
)

# 4. 定義輸入資料模型 (使用 Pydantic)
class PromptRequest(BaseModel):
    """定義 POST 請求的 JSON 格式"""
    keyword: str  # 預計傳入的關鍵字
    
# 5. 定義輸出的資料結構
class SearchSource(BaseModel):
    """單一引文來源的資料結構"""
    uri: str
    title: str

class SearchResponse(BaseModel):
    """API 最終回傳的資料結構"""
    response: str
    source: list[SearchSource]


# 核心邏輯函式 (修改自您的原始程式碼)
def generate_grounded_content(keyword: str) -> dict:
    """
    使用 Gemini 模型並啟用 Google Search Grounding 來生成內容。
    """
    # 針對您的需求，建構更精確的提示詞
    prompt = (
        f"請針對關鍵字『{keyword}』，搜尋並總結其最近一個月內最相關的新聞與事件。"
        f"請只回傳新聞標題以及新聞網址，並將標題和網址整理成一個Markdown格式的列表。"
    )
    
    # --- 關鍵設定 ---
    # 啟用 'google_search' 工具
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(
        tools=[grounding_tool]
    )
    # --- 關鍵設定 ---

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=config,
        )
        
        # 用來儲存引文的列表，注意：每次呼叫都應初始化，不能使用全域變數
        source_list = [] 
        
        # 處理引文（Citations）
        if response.candidates and response.candidates[0].grounding_metadata:
            metadata = response.candidates[0].grounding_metadata
            if metadata.grounding_chunks:
                 for chunk in metadata.grounding_chunks:
                     if chunk.web:
                         source = {
                             "uri": chunk.web.uri,
                             "title": chunk.web.title
                         }
                         source_list.append(source)
                         
        # 返回標準的字典格式
        output = {"response": response.text, "source": source_list}
        return output

    except Exception as e:
        # 如果 API 呼叫失敗，拋出 FastAPI 錯誤
        raise HTTPException(
            status_code=500, 
            detail=f"Gemini API 處理錯誤: {e}"
        )


# 6. 定義 API 端點
@app.post("/search-news", response_model=SearchResponse)
async def search_news_api(request: PromptRequest):
    """
    透過 Gemini 2.5 Flash 結合 Google 搜尋，查詢特定關鍵字最近的新聞並返回標題和來源。
    """
    # 調用核心邏輯
    result = generate_grounded_content(request.keyword)
    return result

@app.get("/callback")
def read_root():
    return {"status": "OK"}
