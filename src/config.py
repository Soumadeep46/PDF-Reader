from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    huggingface_api_token: str
    pdf_text: str = ""

    class Config:
        env_file = ".env"

