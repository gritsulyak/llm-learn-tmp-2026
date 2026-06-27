# NER Pipeline: Named Entity Recognition with YandexGPT

Homework assignment: extract named entities from legal service contracts using the YandexGPT API.

The pipeline extracts 4 entity types from Russian legal documents:
- **parties** — contract parties with name, INN, KPP
- **dates** — signing date and validity period
- **amounts** — monetary values with currency
- **obligations** — deadlines and obligation terms


## Assignment

Build a NER system using YandexGPT API that automatically extracts requisites, dates, amounts,
and deadlines from legal documents and returns them in structured JSON format.

Acceptance criteria:
- Correct integration with YandexGPT API
- Prompts successfully extract all 4 entity types
- Results are machine-readable (JSON)


## Project Structure

```
.
├── .env                          # API keys (not committed to git)
├── .gitignore
├── pyproject.toml                # uv dependencies
├── uv.lock                       # locked dependency versions (commit to git)
├── ner_pipeline.py               # main pipeline entry point
├── ner_comparison.py             # comparison: YandexGPT vs Natasha vs spaCy
├── data/
│   └── documents/                # input .txt contracts (7 test documents)
├── results/                      # output JSON files per document
│   └── _summary.json             # aggregated results
├── notebooks/
│   └── ner_report.ipynb          # full report with examples and conclusions
└── src/
    ├── ner_client.py             # YandexGPT API client + prompts
    └── chunker.py                # document loading, truncation, chunking
```


## Setup

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- Yandex Cloud account with an active folder and API key

### Install

```bash
cd github/gritsulyak/llm-learn-tmp-2026/lessons/03_l8_ya_cloud_ya_gpt

# Initialize uv project
uv init --no-workspace

# Install core dependencies
uv add requests python-dotenv jupyter

# Install classical NER libraries for comparison
uv add natasha
uv add click spacy
uv run python -m spacy download ru_core_news_sm
```

### Configure credentials

```bash
cat > .env << 'EOF'
YC_FOLDER_ID=b1g...
YC_API_KEY=AQV...
YC_API_KEY_ID=...
EOF

echo ".env" >> .gitignore
```

API keys are loaded via `python-dotenv` from `.env`. Never hardcode them in source files.

To use local Ollama instead of Yandex Cloud, add to `.env`:

```
IS_LOCAL=true
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3
```


## Usage

### 1. Diagnose document lengths (no API calls)

Checks each document against the token limit and recommends a processing strategy.

```bash
uv run src/chunker.py
```

Example output:
```
File                           Chars     Tokens ~     Strategy
----------------------------------------------------------------------
doc_01_consulting.txt          2,242        1,724           ok
doc_02_it_support.txt          2,041        1,570           ok
doc_03_cleaning.txt            9,294        7,149     truncate
doc_04_marketing.txt           1,704        1,310           ok
doc_05_accounting.txt          2,021        1,554           ok
doc_06_nda_no_amount.txt       1,894        1,456           ok
doc_07_training_ocr_noise.txt  1,936        1,489           ok
```

### 2. Run the full NER pipeline

Processes all `.txt` files in `data/documents/`, saves results to `results/`.

```bash
uv run ner_pipeline.py
```

Example output:
```
=== NER Pipeline ===

 [=] Processing: doc_01_consulting.txt
     Length: 2,242 chars -> 1 chunk(s)
     Chunk 1/1...
 [V] Saved -> results/doc_01_consulting_ner.json
```

### 3. Run classical NER comparison

Runs the same documents through Natasha and spaCy, prints a comparison table.

```bash
uv run ner_comparison.py
```

### 4. Open the report notebook

Contains prompts, extraction examples, edge case analysis, and conclusions.

```bash
uv run jupyter notebook notebooks/ner_report.ipynb
```


## Prompts

### System prompt

```
You are an accurate AI legal assistant.
Extract facts only from the text, without assumptions.
If data is missing, return null. Never invent data.
```

Design decisions:
- `temperature=0.01` — minimal creativity for structured extraction
- Few-shot JSON example in the user prompt — fixes output schema
- Explicit `null` instruction — prevents hallucinations on missing fields

### User prompt (template)

Extracts `parties`, `dates`, `amounts`, `obligations` from the contract text
with a JSON few-shot example. See `src/ner_client.py` for the full template.


## Test Documents

| File | Type | Edge case |
|---|---|---|
| doc_01_consulting.txt | Consulting services | Full entity set |
| doc_02_it_support.txt | IT support | Monthly + annual amount |
| doc_03_cleaning.txt | Cleaning services | Long document, truncation |
| doc_04_marketing.txt | Marketing | Sole proprietor without KPP |
| doc_05_accounting.txt | Accounting | Variable pricing structure |
| doc_06_nda_no_amount.txt | NDA | No amount — hallucination test |
| doc_07_training_ocr_noise.txt | Training | OCR noise: "O" instead of "0", typos |


## Chunking Strategy

The `chunker.py` module supports two strategies controlled by `STRATEGY` in `ner_pipeline.py`:

- `truncate` — takes start and end of the document, skips the middle; fast, suitable for most contracts where requisites appear at the beginning
- `chunk` — splits by paragraphs, processes each chunk, then merges results via `merge_chunks()`

Token limit estimate: ~4500 characters (~3500 tokens), leaving room for the prompt and response.


## Results

Output per document is saved to `results/<name>_ner.json`. Example:

```json
{
  "parties": [
    {"name": "OOO AlphaTech", "inn": "7701234567", "kpp": "770101001"},
    {"name": "IP Fedorov I.V.", "inn": "540312345678", "kpp": null}
  ],
  "dates": {
    "signed": "15.03.2024",
    "valid_from": "01.04.2024",
    "valid_until": "31.12.2024"
  },
  "amounts": "350 000 rub., incl. VAT 20%",
  "obligations": "payment within 10 business days of signing the acceptance certificate"
}
```

An aggregated `results/_summary.json` is created after each full pipeline run.


## Conclusions

### Strengths of the LLM approach (YandexGPT)

1. Zero labeling effort: no training data, no rules, no model fine-tuning required.
   Works out of the box on any document type via prompt alone.

2. Extracts entities unavailable to classical models:
   INN, KPP, bank accounts, obligation deadlines — entity types absent from Natasha and spaCy.

3. Understands context: correctly distinguishes "Contractor" from "Customer"
   and associates requisites with the right party.

4. Robust to OCR noise: handles "INN 63O1234567" (letter O instead of digit 0)
   and typos in company names — classical models degrade on such input.

5. Flexible via prompt: adding a new entity type requires editing the prompt,
   not retraining a model.

### Weaknesses of the LLM approach

1. Hallucinations: not observed in the test set, but expected on weaker models
   and structurally complex documents at scale.

2. Unstable JSON: not observed in the test set, but likely to occur on larger
   volumes and more complex documents based on general LLM behavior.

3. Speed and cost: 1-3 seconds per document vs under 200ms for Natasha.
   At 10,000 documents/day this becomes a significant bottleneck and token cost.

4. Cloud dependency: the cloud model outperformed llama3 running locally via Ollama
   with CPU inference on a laptop with integrated graphics.

5. Non-determinism: two identical requests can produce different results
   (mitigated by temperature=0.01, but not fully eliminated).

### Production recommendation

Optimal approach is hybrid:

**Classical (Natasha + regex)**
- INN, KPP, bank accounts — deterministic, instant, free
- Full names, organizations, dates — fast, offline, no token cost
- Use as validation layer against LLM output

**LLM (YandexGPT)**
- Obligation deadlines, party context, OCR-degraded documents
- Documents with non-linear structure (e.g., requisites not in reading order)

A pure LLM approach is justified for prototypes or low-volume human-in-the-loop workflows.
For high-load production on well-structured documents, Natasha+regex is cheaper and more
reliable in fully automated mode.


## Security

- API keys are stored in `.env` only
- `.env` is listed in `.gitignore`
- No keys appear in source files, notebooks, or result files
- Before submitting: run `grep -r "AQV\|YC_" src/ notebooks/` to verify


## References

- [YandexGPT API reference](https://github.com/pueraeternis/yandex-gpt-api)
- [Yandex Cloud setup guide (Habr)](https://habr.com/ru/articles/780008/)
- [RuLegalNER dataset](https://github.com/zeino8/RuLegalNER)
