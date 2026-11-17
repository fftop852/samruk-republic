# Samruk Republic — Document Analyzer

**EN.** Evidence-grounded question answering over official PDF documents.
Upload a PDF and ask a question. The app finds the most relevant passages
(multilingual MiniLM embeddings), asks Gemini to answer **only** from them as
short claims with page numbers, then checks every claim in a separate,
independent call. If nothing is confirmed by the document, it answers
"Не знаю / I don't know" instead of guessing.

**RU.** Система отвечает на вопросы по официальным PDF-документам только со
ссылкой на страницу. Если в документе нет подтверждения, она честно говорит
«Не знаю». Каждое утверждение ответа проверяется отдельным, независимым
запросом.

Pipeline: PDF → text / OCR (Gemini Vision) → paragraph chunks → semantic
search (top-8) → Gemini answer as claims with pages → independent claim
check → answer with evidence or "Не знаю".

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
Put the key in `.streamlit/secrets.toml` (never commit this file):
```
GOOGLE_API_KEY = "your_key_here"
```

## Streamlit Community Cloud
Main file: `app.py`. In *Advanced settings → Secrets* paste:
```
GOOGLE_API_KEY = "your_key_here"
MAX_QUESTIONS_PER_SESSION = 3   # new questions per visitor session (0 = no limit)
MAX_QUESTIONS_PER_DAY = 30      # new questions per day for the whole app (0 = no limit)
```
Repeated questions are served from cache and are not counted.

Built-in sample (button «Использовать пример»): `samples/bolashak_rules_573.pdf` — Rules for the Bolashak scholarship (Government Decree of the RK No. 573, official text at adilet.zan.kz).
