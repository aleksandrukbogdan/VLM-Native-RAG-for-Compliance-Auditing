import json
import utils
import config
import chromadb
from pipeline.step_5_indexing import LocalEmbeddingFunction

# --- System Prompts ---

GENERATOR_PROMPT = """
Ты - эксперт по строительной документации (П87). Твоя задача - помочь найти доказательства выполнения правила в документации.

ПРАВИЛО: "{rule}"

Сформулируй 3 разных поисковых запроса, которые помогут найти нужную информацию в тексте Проектной Документации.
Запросы должны быть краткими, но содержать ключевые термины (например: "технико-экономические показатели", "класс бетона", "назначение здания").

Верни результат в формате JSON списка строк:
["запрос 1", "запрос 2", "запрос 3"]
"""

VALIDATOR_PROMPT = """
Ты - аудитор строительной документации. Твоя задача - проверить соответствие ПРАВИЛА и ДАННЫХ.

ПРАВИЛО (Требование П87):
"{rule}"

НАЙДЕННЫЕ ДАННЫЕ ИЗ ПРОЕКТА:
{context}

---
ЗАДАЧА:
Проанализируй данные и ответь, выполняется ли правило.
1. Если информация найдена и соответствует - статус "ВЫПОЛНЕНО".
2. Если информации нет или недостаточно - статус "НЕ НАЙДЕНО".
3. Если информация есть, но противоречит - статус "НАРУШЕНИЕ".

Верни ответ в формате JSON:
{{
  "status": "ВЫПОЛНЕНО" | "НЕ НАЙДЕНО" | "НАРУШЕНИЕ",
  "reason": "Краткое пояснение своими словами",
  "evidence": "Цитата из найденных данных (если есть)",
  "source_page": "Номер страницы (если известно)"
}}
"""

def get_relevant_context(queries: list, n_results: int = 3) -> list:
    """
    Searches ChromaDB for multiple queries and deduplicates results.
    """
    client = chromadb.PersistentClient(path=str(config.CHROMA_DB_PATH))
    embedding_fn = LocalEmbeddingFunction(config.EMBEDDING_MODEL_NAME)
    try:
        collection = client.get_collection(name=config.COLLECTION_NAME, embedding_function=embedding_fn)
    except:
        print("Error: Collection not found. Run Step 5 first.")
        return []

    unique_docs = {} # Map content -> metadata
    
    for q in queries:
        results = collection.query(query_texts=[q], n_results=n_results)
        
        ids = results['ids'][0]
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        
        for i, doc in enumerate(docs):
            if doc not in unique_docs:
                unique_docs[doc] = metas[i]
                
    # Convert back to list of formatted strings
    context_list = []
    for doc, meta in unique_docs.items():
        page = meta.get('page_number', '?')
        doc_type = meta.get('type', 'text')
        context_list.append(f"[Стр. {page}, Тип: {doc_type}] {doc}")
        
    return context_list

def check_rule_compliance(rule_text: str):
    """
    Main agent loop for a single rule.
    """
    print(f"\nChecking Rule: {rule_text[:50]}...")
    
    # 1. Generate Queries
    print("  Thinking about search queries...")
    gen_prompt = GENERATOR_PROMPT.format(rule=rule_text)
    response = utils.call_llm_text(gen_prompt)
    
    try:
        queries = utils.parse_json_from_response(response)
        if not isinstance(queries, list): raise ValueError
    except:
        print("  Failed to parse queries, using rule text as query.")
        queries = [rule_text]
        
    print(f"  Generated Queries: {queries}")
    
    # 2. Retrieve Context
    print("  Searching database...")
    context_items = get_relevant_context(queries)
    
    if not context_items:
        return {
            "status": "НЕ НАЙДЕНО",
            "reason": "База знаний не вернула релевантных данных по запросам.",
            "evidence": None
        }
    
    # Join context for LLM (limit length roughly)
    full_context = "\n\n".join(context_items)[:10000] # Limit to avoid context overflow
    
    # 3. Validate
    print("  Analysing evidence...")
    val_prompt = VALIDATOR_PROMPT.format(rule=rule_text, context=full_context)
    val_response = utils.call_llm_text(val_prompt)
    
    try:
        result = utils.parse_json_from_response(val_response)
        return result
    except:
        return {
            "status": "ERROR",
            "reason": "Ошибка парсинга ответа LLM",
            "raw_response": val_response
        }

if __name__ == "__main__":
    # Test with a dummy rule
    test_rule = "В пояснительной записке должны быть указаны реквизиты договора на выполнение проектных работ."
    result = check_rule_compliance(test_rule)
    print("\n=== FINAL REPORT ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

