"""Test rapido para ver el error exacto de Groq."""
import requests
from core.config import LLM_API_KEY, LLM_API_URL, LLM_MODEL

print(f"Modelo configurado: {LLM_MODEL}")
print(f"URL: {LLM_API_URL}")
print()

# Probar con el modelo actual
resp = requests.post(
    LLM_API_URL,
    json={
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": "Hola"}],
        "max_tokens": 10,
    },
    headers={"Authorization": f"Bearer {LLM_API_KEY}"},
    timeout=15,
)
print(f"Status: {resp.status_code}")
print(f"Respuesta: {resp.text}")

if resp.status_code != 200:
    # Intentar con modelos alternativos
    print("\n--- Probando modelos alternativos ---")
    for modelo in ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "gemma2-9b-it", "mixtral-8x7b-32768"]:
        r = requests.post(
            LLM_API_URL,
            json={"model": modelo, "messages": [{"role": "user", "content": "Hola"}], "max_tokens": 10},
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            timeout=15,
        )
        status = "[OK]" if r.status_code == 200 else f"[{r.status_code}]"
        print(f"  {status} {modelo}")
        if r.status_code == 200:
            print(f"        Respuesta: {r.json()['choices'][0]['message']['content']}")
