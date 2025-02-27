from flask import Flask, request, jsonify, send_from_directory, Response
import os
import asyncio
import openai
import pandas as pd
import ast
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
import redis
import azure.cognitiveservices.speech as speechsdk
from rtclient import ResponseCreateMessage, RTLowLevelClient, ResponseCreateParams
import json
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import nest_asyncio
nest_asyncio.apply()
 
# Import Bot Framework dependencies
from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings, TurnContext
from botbuilder.schema import Activity
 
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
 
# ------------------------------------------------------------------
# Configuration for Azure OpenAI, GPT‑4o realtime, Azure Search, Redis, Speech
# ------------------------------------------------------------------
 
# Azure OpenAI configuration for embeddings
OPENAI_API_KEY = "8929107a6a6b4f37b293a0fa0584ffc3"
OPENAI_API_VERSION = "2023-03-15-preview"
OPENAI_ENDPOINT = "https://genral-openai.openai.azure.com/"
EMBEDDING_MODEL = "text-embedding-ada-002"  # Fast embedding model
 
# GPT‑4o realtime
RT_API_KEY = "9e76306d48fb4e6684e4094d217695ac"
RT_ENDPOINT = "https://general-openai02.openai.azure.com/"
RT_DEPLOYMENT = "gpt-4o-realtime-preview"
RT_API_VERSION = "2024-10-17"
 
# Azure Cognitive Search
SEARCH_SERVICE_NAME = "mainsearch01"          
SEARCH_INDEX_NAME = "id"                      
SEARCH_API_KEY = "Y6dbb3ljV5z33htXQEMR8ICM8nAHxOpNLwEPwKwKB9AzSeBtGPav"
 
# Redis
REDIS_HOST = "AiKr.redis.cache.windows.net"
REDIS_PORT = 6380
REDIS_PASSWORD = "OD8wyo8NiVxse6DDkEY19481Xr7ZhQAnfAzCaOZKR2U="
 
# Speech
SPEECH_KEY = "3c358ec45fdc4e6daeecb7a30002a9df"
SPEECH_REGION = "westus2"
 
# Thresholds for determining whether a search result is “good enough.”
SEMANTIC_THRESHOLD = 3.4
VECTOR_THRESHOLD = 0.91
 
# ------------------------------------------------------------------
# Initialize clients and load data
# ------------------------------------------------------------------
 
# Initialize the Azure OpenAI client (for embeddings)
client = openai.AzureOpenAI(
    api_key=OPENAI_API_KEY,
    api_version=OPENAI_API_VERSION,
    azure_endpoint=OPENAI_ENDPOINT
)
 
# Load Q&A data
try:
    qa_data = pd.read_csv("qa_data.csv", encoding="windows-1256")
    print("✅ CSV file loaded successfully!")
    print(qa_data.head())
except Exception as e:
    print(f"❌ Failed to load CSV file: {e}")
    exit()
 
# Normalize column names (convert to lowercase, trim spaces)
qa_data.rename(columns=lambda x: x.strip().lower(), inplace=True)
 
# Convert the 'id' column to string (fix type conversion error)
if "id" in qa_data.columns:
    qa_data["id"] = qa_data["id"].astype(str)
 
# Verify required columns exist
if "question" not in qa_data.columns or "answer" not in qa_data.columns:
    print("❌ CSV file must contain 'question' and 'answer' columns.")
    exit()
 
# EMBEDDING GENERATION
def get_embedding(text):
    """
    Generate an embedding for the given text using the OpenAI model.
    """
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        embedding = response.data[0].embedding
        print(f"✅ Embedding generated for text: {text}")
        return embedding
    except Exception as e:
        print(f"❌ Failed to generate embedding for text '{text}': {e}")
        return None
 
# Generate embeddings if not already present
if "embedding" not in qa_data.columns or qa_data["embedding"].isnull().all():
    qa_data["embedding"] = qa_data["question"].apply(get_embedding)
    qa_data.to_csv("embedded_qa_data.csv", index=False)
    print("✅ Embeddings generated and saved to 'embedded_qa_data.csv'.")
else:
    def convert_embedding(x):
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except Exception as e:
                print("❌ Failed to parse embedding:", e)
                return None
        return x
    qa_data["embedding"] = qa_data["embedding"].apply(convert_embedding)
    print("✅ Using existing embeddings from CSV.")
 
# Normalize question text for consistent matching.
qa_data["question"] = qa_data["question"].str.strip().str.lower()
 
# UPLOAD DOCUMENTS TO AZURE COGNITIVE SEARCH
search_client = SearchClient(
    endpoint=f"https://{SEARCH_SERVICE_NAME}.search.windows.net/",
    index_name=SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_API_KEY)
)
 
documents = qa_data.to_dict(orient="records")
try:
    upload_result = search_client.upload_documents(documents=documents)
    print(f"✅ Uploaded {len(documents)} documents to Azure Search. Upload result: {upload_result}")
except Exception as e:
    print(f"❌ Failed to upload documents: {e}")
 
# Debug: Run a simple query to verify that documents are in the index.
try:
    simple_results = search_client.search(
        search_text="*",
        select=["question", "answer"],
        top=3
    )
    print("Simple query results:")
    for doc in simple_results:
        print(doc)
except Exception as e:
    print(f"❌ Simple query error: {e}")
 
# INITIALIZE REDIS CLIENT
try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=0,
        ssl=True,
        decode_responses=True,
        password=REDIS_PASSWORD
    )
    redis_client.ping()
    print("✅ Successfully connected to Redis!")
except Exception as e:
    print(f"❌ Failed to connect to Redis: {e}")
 
# ------------------------------------------------------------------
# SEARCH & RESPONSE FUNCTIONS
# ------------------------------------------------------------------
 
def check_redis_cache(query):
    """Return cached answer if it exists."""
    try:
        cached_answer = redis_client.get(query)
        if cached_answer:
            print(f"✅ Using cached answer for query: {query}")
            return cached_answer
    except Exception as e:
        print(f"❌ Redis error: {e}")
    return None
 
def get_best_match(query):
    """
    Retrieve the best answer for the query by trying semantic then vector search.
    """
    cached_response = check_redis_cache(query)
    if cached_response:
        return cached_response
 
    # --- Semantic Search ---
    try:
        semantic_results = search_client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name="my-semantic-config-default",
            query_caption="extractive",
            select=["question", "answer"],
            top=3
        )
        semantic_answer = next(semantic_results, None)
        if semantic_answer:
            reranker_score = semantic_answer["@search.reranker_score"]
            if reranker_score is not None and reranker_score >= SEMANTIC_THRESHOLD:
                answer = semantic_answer["answer"]
                print("✅ Found match using Semantic Search with score", reranker_score)
                redis_client.set(query, answer, ex=3600)
                return answer
            else:
                print("❌ Semantic search result score below threshold:", reranker_score)
        else:
            print("❌ No semantic search answers found.")
    except Exception as e:
        print(f"❌ Semantic search error: {e}")
 
    # --- Vector Search ---
    try:
        query_embedding = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query
        ).data[0].embedding
 
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=50,
            fields="embedding"
        )
        vector_results = search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["question", "answer"],
            top=3
        )
        best_vector = next(vector_results, None)
        if best_vector:
            score = best_vector.get("@search.score", 0)
            if score >= VECTOR_THRESHOLD:
                answer = best_vector["answer"]
                print("✅ Found match using Vector Search with score", score)
                redis_client.set(query, answer, ex=3600)
                return answer
            else:
                print("❌ Vector search result score below threshold:", score)
        else:
            print("❌ No vector search results found.")
    except Exception as e:
        print(f"❌ Vector search error: {e}")
 
    print("❌ No match found using Semantic or Vector Search")
    return None
 
# GPT‑4o REALTIME FALLBACK (ASYNC)
async def get_realtime_response(user_query):
    """
    Fallback function: Uses GPT‑4o realtime to generate an answer if both searches fail.
    Now with added instructions so that the model responds as an Egyptian man.
    """
    try:
        async with RTLowLevelClient(
            url=RT_ENDPOINT,
            azure_deployment=RT_DEPLOYMENT,
            key_credential=AzureKeyCredential(RT_API_KEY)
        ) as client_rt:
            # Prepend instruction for Egyptian persona
            instructions = "أنت رجل عربي. انا لا اريد ايضا اي bold points  في الاجابة  و لا اريد عنواين مرقمة" + user_query
            await client_rt.send(
                ResponseCreateMessage(
                    response=ResponseCreateParams(
                        modalities={"text"},
                        instructions=instructions
                    )
                )
            )
            done = False
            response_text = ""
            while not done:
                message = await client_rt.recv()
                if message is None:
                    print("❌ No message received from the real-time service.")
                    break
                if message.type == "response.done":
                    done = True
                elif message.type == "error":
                    done = True
                    print(f"Error: {message.error}")
                elif message.type == "response.text.delta":
                    response_text += message.delta
            return response_text
    except Exception as e:
        print(f"❌ Failed to get real-time response: {e}")
        return "عذرًا، حدث خطأ أثناء محاولة الاتصال بخدمة الدعم الفوري. يرجى المحاولة مرة أخرى لاحقًا."
 
async def get_response(user_query):
    """
    Retrieve a response by first trying search (semantic then vector),
    then falling back to GPT‑4o realtime if no match is found.
    """
    print(f"🔍 Processing query: {user_query}")
    response = get_best_match(user_query)
    if response:
        print(f"✅ Found response in cache or search: {response}")
        return response
 
    print("🔍 No match found, falling back to GPT‑4o realtime...")
    realtime_response = await get_realtime_response(user_query)
    if realtime_response:
        print(f"✅ GPT‑4o realtime response: {realtime_response}")
        try:
            redis_client.set(user_query, realtime_response, ex=3600)
            print("✅ Response cached in Redis.")
        except Exception as e:
            print(f"❌ Failed to cache response in Redis: {e}")
        return realtime_response
    else:
        return "عذرًا، لم أتمكن من العثور على إجابة. يرجى المحاولة مرة أخرى لاحقًا."
 
# ------------------------------------------------------------------
# SPEECH RECOGNITION & SYNTHESIS SETUP
# ------------------------------------------------------------------
 
speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
speech_config.speech_recognition_language = "ar-EG"
speech_config.speech_synthesis_voice_name = "ar-EG-ShakirNeural"
 
def recognize_speech():
    """Listen for a single utterance using the default microphone."""
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    print("Listening... (Speak in Egyptian Arabic)")
    result = recognizer.recognize_once_async().get()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(f"You said: {result.text}")
        return result.text
    else:
        print(f"Speech not recognized: {result.reason}")
        return ""
 
def speak_response(text):
    """Convert the given text to speech and output via the default speaker."""
    audio_output = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output)
    result = synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.Canceled:
        cancellation = result.cancellation_details
        print("Speech synthesis canceled:")
        print("  Reason: {}".format(cancellation.reason))
        print("  Error Details: {}".format(cancellation.error_details))
 
# HELPER: CLEAN TEXT FOR EXIT CHECK
def clean_text(text):
    """
    Remove common punctuation and whitespace from the beginning and end of the text,
    then convert to lower case.
    """
    return text.strip(" .،!؛؟").lower()
 
# CRITICAL ISSUE DETECTION
def detect_critical_issue(text):
    """
    Detect if the user's input contains a critical issue that should be passed to a human.
    """
    # Arabic Trigger Sentences
    trigger_sentences = [
        "تم اكتشاف اختراق أمني كبير.",
        "تمكن قراصنة من الوصول إلى بيانات حساسة.",
        "هناك هجوم إلكتروني على النظام الخاص بنا.",
        "تم تسريب بيانات المستخدمين إلى الإنترنت.",
        "رصدنا محاولة تصيد إلكتروني ضد موظفينا.",
        "تم استغلال ثغرة أمنية في الشبكة.",
        "هناك محاولة وصول غير مصرح بها إلى الملفات السرية."
    ]
 
    # Get embeddings for trigger sentences
    trigger_embeddings = np.array([get_embedding(sent) for sent in trigger_sentences])
 
    # Get embedding for the input text
    text_embedding = np.array(get_embedding(text)).reshape(1, -1)
 
    # Calculate cosine similarity between the input text and trigger sentences
    similarities = cosine_similarity(text_embedding, trigger_embeddings)
    max_similarity = np.max(similarities)
 
    # If the similarity is above a threshold, consider it a critical issue
    if max_similarity > 0.9:
        print("This issue should be passed to a human.")
        return True
    return False
 
# ------------------------------------------------------------------
# ASYNCHRONOUS VOICE CHAT LOOP & ROUTES
# ------------------------------------------------------------------
 
async def voice_chat_loop():
    print("🤖 Arabic Voice Bot Ready! Say 'إنهاء' or 'خروج' to exit.")
    while True:
        user_query = recognize_speech()
        if clean_text(user_query) in ["إنهاء", "خروج"]:
            print("👋 Goodbye!")
            speak_response("مع السلامة!")
            break
 
        if detect_critical_issue(user_query):
            response = "هذه المشكلة تحتاج إلى تدخل بشري. سأقوم بالاتصال بخدمة العملاء لدعمك."
            print(f"🤖 Bot: {response}")
            speak_response(response)
            continue
 
        response = await get_response(user_query)
        print(f"🤖 Bot: {response}")
        speak_response(response)
 
async def voice_chat(user_query):
    try:
        if not user_query:
            return "في انتظار اوامرك"
        if clean_text(user_query) in ["إنهاء", "خروج"]:
            return "مع السلامة"
        if detect_critical_issue(user_query):
            response = "هذه المشكلة تحتاج إلى تدخل بشري. سأقوم بالاتصال بخدمة العملاء لدعمك."
            return response
 
        # Directly await the get_response coroutine without creating a new event loop.
        response = await get_response(user_query)
        return response
    except Exception as e:
        print(f"Error in /voice-chat: {e}")
        return "حدث خطأ أثناء معالجة طلبك."
 
 
@app.route("/")
def index():
    return send_from_directory("static", "index.html")
 
# ------------------------------------------------------------------
# Bot Framework Integration
# ------------------------------------------------------------------
from botbuilder.schema import ConversationAccount
 
# Bot Framework credentials (set via environment or hard-code for testing)
MICROSOFT_APP_ID = "b0a29017-ea3f-4697-aef7-0cb05979d16c"
MICROSOFT_APP_PASSWORD = "2fc8Q~YUZMbD8E7hEb4.vQoDFortq3Tvt~CLCcEQ"
 
# Initialize Bot Framework adapter
adapter_settings = BotFrameworkAdapterSettings(MICROSOFT_APP_ID, MICROSOFT_APP_PASSWORD)
adapter = BotFrameworkAdapter(adapter_settings)
 
# Define a bot class that uses your get_response logic
class VoiceChatBot:
    async def on_turn(self, turn_context: TurnContext):
        if turn_context.activity.type == "message":
            user_query = turn_context.activity.text
            print(f"Received message: {user_query}")
            response_text = await voice_chat(user_query)
            await turn_context.send_activity(response_text)
        elif turn_context.activity.type == "conversationUpdate":
            for member in turn_context.activity.members_added or []:
                if member.id != turn_context.activity.recipient.id:
                    welcome_message = "مرحبًا! كيف يمكنني مساعدتك اليوم؟"
                    await turn_context.send_activity(welcome_message)
        else:
            await turn_context.send_activity(f"Received activity of type: {turn_context.activity.type}")
 
# Create an instance of the bot
bot = VoiceChatBot()
 
@app.route("/api/messages", methods=["POST"])
def messages():
    if request.headers.get("Content-Type", "") != "application/json":
        return Response("Invalid Content-Type", status=415)
   
    try:
        body = request.json
        print(body)
        if not body:
            print("❌ Empty request body received")
            return Response("Empty request body", status=400)
   
        print("🔍 Incoming request JSON:", json.dumps(body, indent=2, ensure_ascii=False))
   
        # Ensure the activity type is set
        if "type" not in body:
            body["type"] = "message"
            print("🔍 Updated request JSON:", json.dumps(body, indent=2, ensure_ascii=False))
                   
        # Deserialize the incoming JSON into an Activity object
        activity = Activity().deserialize(body)
       
        if not activity.channel_id:
            activity.channel_id = body.get("channelId", "webchat")
        if not activity.service_url:
            # Make sure this URL is correct and reachable
            activity.service_url = "https://linkdev-poc-cfb2fbaxbgf9d4dd.westeurope-01.azurewebsites.net"
       
        auth_header = request.headers.get("Authorization", "")
        if not auth_header:
            auth_header="Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6ImltaTBZMnowZFlLeEJ0dEFxS19UdDVoWUJUayJ9.eyJhdWQiOiJiMGEyOTAxNy1lYTNmLTQ2OTctYWVmNy0wY2IwNTk3OWQxNmMiLCJpc3MiOiJodHRwczovL2xvZ2luLm1pY3Jvc29mdG9ubGluZS5jb20vZDZkNDk0MjAtZjM5Yi00ZGY3LWExZGMtZDU5YTkzNTg3MWRiL3YyLjAiLCJpYXQiOjE3NDA0NzY2MzcsIm5iZiI6MTc0MDQ3NjYzNywiZXhwIjoxNzQwNTYzMzM3LCJhaW8iOiJBU1FBMi84WkFBQUFzVnYrNXRUT0JkV3ZnVEYrQmVpUkd3azcyRTNKOXl6c1BjdHZjZmR5YU5ZPSIsImF6cCI6ImIwYTI5MDE3LWVhM2YtNDY5Ny1hZWY3LTBjYjA1OTc5ZDE2YyIsImF6cGFjciI6IjEiLCJyaCI6IjEuQVc0QUlKVFUxcHZ6OTAyaDNOV2FrMWh4MnhlUW9yQV82cGRHcnZjTXNGbDUwV3hlQVFCdUFBLiIsInRpZCI6ImQ2ZDQ5NDIwLWYzOWItNGRmNy1hMWRjLWQ1OWE5MzU4NzFkYiIsInV0aSI6IkhxVi1ZcHFoalVtZlJmXzlOXzhuQUEiLCJ2ZXIiOiIyLjAifQ.tkkP-QoPHHc4PqiUJNVUW-VsQwkhHmbFbbf_ZPviliEI7ldAmSYNbEbde9JsZwSHzFNsrYm_Ke3keSa_CVuRshFV2xXoMHTJtDdrU5NyfvN0ifIR1eUoLjIWMUDt0mDNXpHUjvBXKSbO-H7vejz3pk8xTejOMSR36iT6jpxPBEVH-5UdonJPAWGFHjouisOgfginuMJa4ZAFFeivdnGyubw67K8tEJejgwkFllevYaVDM5NEPTZMpDFFhwQKrPZQw_8spE1XEA_LK-SdrzIyWPO1rHbcDkKP5lhD2bHZHBKtrWiZzR_n1D7gZZ0AdT_bHDmJI26NplBEw7F9wNstoA"
        print("auth: ", auth_header)
 
        # Fix 2: Use shared event loop policy
        loop = asyncio.get_event_loop()
       
        # Fix 3: Add timeout handling for the entire operation
        async def process_activity():
            try:
                # Fix: Await process_activity() directly inside wait_for()
                await adapter.process_activity(activity, auth_header, bot.on_turn)
                 
            except asyncio.TimeoutError:
                print("⚠️ Bot processing timed out after 60s")
                raise
            except Exception as e:
                print(f"❌ Error in adapter processing: {e}")
                raise
 
        try:
             #Fix 5: Use shorter overall timeout
            loop.run_until_complete(process_activity())
        except asyncio.TimeoutError:
            print("❌ Total processing time exceeded 150 seconds")
            return Response("Request timeout", status=504)
           
        #return response
 
    except Exception as e:
        print(f"❌ Critical error in /api/messages: {str(e)}")
        return Response("Internal server error", status=500)
 
 
 
if __name__ == "__main__":
    app.run(debug=True)
 
