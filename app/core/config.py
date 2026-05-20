import os
from dotenv import load_dotenv

load_dotenv(override=True)

class Settings:
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
    DB_URL = os.getenv("DATABASE_URL")
    SECRET_KEY = os.getenv("SECRET_KEY")
    ALGORITHM = os.getenv("ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))
    TWILIO_NUMBER = os.getenv("TWILIO_NUMBER")

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")
    ELEVENLABS_AGENT_ID: str = "agent_3601krk52znzet18mpgbxm0pzvqz"



from pinecone import Pinecone, PodSpec, ServerlessSpec



settings = Settings()


pc = Pinecone(api_key=settings.PINECONE_API_KEY)
if settings.PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=settings.PINECONE_INDEX_NAME,
        dimension=512,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
            )
        )
#     pc.create_index(
#     name="my-high-performance-index",
#     dimension=1536, # Standard for OpenAI
#     metric="cosine", # p2 supports cosine, euclidean, and dotproduct
    
#     spec=PodSpec(
#         environment="gcp-starter", # Standard free environment
#         pod_type="p2.x1",         # x1 is the single-pod size
#         pods=1
#     ), 
# )
