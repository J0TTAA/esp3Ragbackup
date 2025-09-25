# app.py

from dotenv import load_dotenv
from providers.deepseek import DeepSeekProvider
from providers.chatgpt import ChatGPTProvider
from providers.openrouter import OpenRouterProvider
import os

load_dotenv()

if __name__ == "__main__":
    providers = []

    # DeepSeek
    try:
        deepseek = DeepSeekProvider()
        providers.append(("DeepSeek", deepseek))
    except Exception as e:
        print(f"Error cargando DeepSeek: {e}")

   
    # OpenRouter (puede usar OpenAI, DeepSeek u otros modelos según el .env)
    try:
        model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        openrouter = OpenRouterProvider(model=model)
        providers.append(("OpenRouter", openrouter))
    except Exception as e:
        print(f"Error cargando OpenRouter: {e}")

    if not providers:
        print("No hay proveedores disponibles")
        exit(1)

    messages = [{"role": "user", "content": "Hola, ¿quién eres?"}]

    for name, provider in providers:
        try:
            response = provider.chat(messages)
            print(f"{name}: {response}")
        except Exception as e:
            print(f"Error con {name}: {e}")
