import os
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from uuid import uuid4
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="Car Mechanical Chatbot",
    description="مساعد ذكي للسيارات ",
    version="1.0.1"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-8b-8192",
        max_tokens=400,
        temperature=0.2
    )
except ValueError as e:
    logger.error(f"فشل في تهيئة : {str(e)}")
    raise RuntimeError("apiHF not defined, please check your environment variables.")

sessions: Dict[str, ConversationBufferMemory] = {}

class ChatMessage(BaseModel):
    message: str
    session_id: str = None

# بدء جلسة جديدة
@app.get("/start_session")
async def start_session():
    session_id = str(uuid4())
    memory = ConversationBufferMemory(return_messages=True)
    system_message = SystemMessage(
        content="أنت مساعد ذكي ومتخصص في مجال الميكانيكا والسيارات. تجيب دائمًا باللغة العربية الفصحى، بأسلوب واضح وبسيط، مع شرح مفصل عند الحاجة وبأمثلة عملية. هدفك هو تقديم حلول دقيقة وسريعة لكل استفسارات الصيانة، الأعطال، نصائح الصيانة الدورية، واختيار قطع الغيار المناسبة."
    )
    memory.chat_memory.add_message(system_message)
    sessions[session_id] = memory
    logger.info(f"جلسة جديدة: {session_id}")
    return {"session_id": session_id}

# نقطة المحادثة
@app.post("/chat")
async def chat(request: ChatMessage):
    try:
        if not request.session_id or request.session_id not in sessions:
            session_id = str(uuid4())
            memory = ConversationBufferMemory(return_messages=True)
            system_message = SystemMessage(
                content= "أنت مساعد ذكي متخصص في الميكانيكا والسيارات، تجيب بالعربية أو الإنجليزية بأسلوب واضح وبسيط، مع شرح مفصل وأمثلة عملية عند الحاجة. تقدم حلولاً دقيقة وسريعة لاستفسارات الصيانة، الأعطال، نصائح الصيانة الدورية، واختيار قطع الغيار المناسبة."
            )
            memory.chat_memory.add_message(system_message)
            sessions[session_id] = memory
        else:
            session_id = request.session_id
            memory = sessions[session_id]

        memory.save_context({"input": request.message}, {"output": ""})
        messages = memory.chat_memory.messages

        response = llm.invoke(messages)
        memory.chat_memory.messages[-1] = response  

        logger.info(f"رد مرسل للجلسة: {session_id}")
        return {"session_id": session_id, "response": response.content}
    except ValueError as ve:
        logger.error(f"خطأ في المدخلات: {str(ve)}")
        raise HTTPException(status_code=400, detail="مدخلات غير صحيحة")
    except Exception as e:
        logger.error(f"خطأ غير متوقع: {str(e)}")
        raise HTTPException(status_code=500, detail="خطأ داخلي في الخادم")

# الحصول على تاريخ المحادثة
@app.get("/history/{session_id}")
async def get_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="الجلسة غير موجودة")
    memory = sessions[session_id]
    messages = [{"type": "system" if "SystemMessage" in str(type(msg)) else "user" if "HumanMessage" in str(type(msg)) else "assistant", "content": msg.content} for msg in memory.chat_memory.messages if hasattr(msg, 'content')]
    return {"session_id": session_id, "messages": messages}

# delete a session
@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return {"message": "تم حذف الجلسة"}
    raise HTTPException(status_code=404, detail="الجلسة غير موجودة")

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "مرحباً بك في الميكانيكي الذكي",
        "active_sessions": len(sessions),
        "endpoints": [
            "/start_session - بدء جلسة جديدة",
            "/chat - إرسال رسالة",
            "/history/{session_id} - عرض التاريخ",
            "/session/{session_id} - حذف جلسة"
        ]
    }