# מדריך הכנה לראיון — `rag-production`

## 1. The 30-Second Elevator Pitch

זהו אב־טיפוס RAG לתשובות על מסמכי ידע/מדיניות: מסמכים נטענים, מנוקים, מחולקים ל־chunks, מוטמעים ונשמרים; בשאילתה המערכת מאחזרת קטעים באמצעות dense ו־BM25, משלבת אותם (במסלול ה־API הרגיל: RRF), מריצה reranking אופציונלי, ומעבירה את ההקשר ל־LLM לתשובה עם מקורות. FastAPI חושף גם חיפוש, streaming, ניהול מסמכים ו־agent מבוקר; Streamlit הוא לקוח HTTP. יש גם eval offline ו־tests.

חשוב לדייק: זהו פרויקט לימודי/אב־טיפוס עם רכיבים שימושיים, לא מערכת production מלאה. בפרט, חלק מה־fallbacks הם mock, ה־state המרכזי בזיכרון, וה־agent בפועל מבצע כלי יחיד בכל ריצה.

## 2. The Happy Path (Step-by-Step Flow)

מסלול השאילתה הרגיל, ללא streaming:

1. המשתמש מזין שאלה ב־`ui/streamlit_app.py` — `main()` קורא ל־`call_ask()`; במסלול JSON, `call_ask()` משתמש ב־`call()` מול `POST /ask`.
2. `src/api/main.py` — `ask_question()` מאמת/מסנן את הקלט, בודק rate limit ו־cache; cache hit מחזיר מיד.
3. האתחול בונה את הצינור: `src/api/main.py` — `startup_event()` קורא ל־`initialize_rag_pipeline()`, שמבצע load → clean → chunk → embed → store ובונה retrievers, reranker ו־answer generator.
4. באותה בקשה, `src/api/main.py` — `ask_question()` קורא `rag_pipeline["retriever"].retrieve()`; כאשר reranking מופעל, הוא מבקש עד `top_k * 2` מועמדים. ה־hybrid retriever מאחד dense ו־BM25 באמצעות RRF.
5. `src/api/main.py` — `ask_question()` קורא ל־`reranker.rerank()` אם הופעל וזמין, ומצמצם ל־`top_k` קטעים.
6. `src/generation/answer.py` — `AnswerGenerator.generate_answer_with_context()` בונה הקשר בתקציב משוער, יוצר grounded prompt ומייצר תשובה דרך ה־provider שנבחר; הוא גם בודק grounding ומפיק מקורות/metadata.
7. `src/api/main.py` — `ask_question()` מסנן את התשובה, בונה `QueryResponse`, שומר cache ומחזיר JSON; `ui/streamlit_app.py` — `main()` מציג תשובה, מקורות ונתוני provider/latency.

**הסתעפות streaming:** `ui/streamlit_app.py` — `call_ask()` שולח ל־`/ask/stream`; ב־`src/api/main.py`, `ask_stream()` מייצר SSE דרך `event_gen()`, שמפעיל `stream_answer_with_context()`.

## 3. Actual Tech Stack & Design Choices

- **Serving/UI:** `FastAPI` ו־`Pydantic` ב־`src/api/main.py`; `Streamlit` ו־`requests` ב־`ui/streamlit_app.py`.
- **Ingestion:** loaders, cleaner ו־chunker ב־`src/ingestion/`; שלוש אסטרטגיות: `fixed_size`, `recursive`, `structure_aware` (`src/ingestion/chunker.py`). הגודל וה־overlap הם תווים, לא tokens.
- **Embeddings:** `Embedder` ב־`src/embedding/embedder.py`; ברירת המחדל `all-MiniLM-L6-v2` דרך `sentence-transformers`, עם disk cache מבוסס MD5, retry/rate limit ו־hash-based mock fallback. ה־fallback אינו embedding סמנטי אמיתי. `providers/embeddings.py` כולל גם `SentenceTransformerEmbeddingProvider` שמוצהר במפורש כ־placeholder.
- **Storage:** `ChromaDB PersistentClient` עם collection `rag_academy` ב־`src/store/vector_store.py`; בהיעדר Chroma או בכשל, יש fallback בזיכרון עם cosine similarity. אין backend DB נוסף שמוגדר כ־source of truth.
- **Retrieval:** dense retrieval (`src/retrieval/retriever.py`), מימוש BM25 פנימי בזיכרון (`src/retrieval/sparse_retriever.py`), ו־RRF (`src/retrieval/hybrid.py`). RRF מאחד לפי rank; `weight_dense`/`weight_sparse` נשמרים אך אינם מיושמים בפונקציית המיזוג. יש גם `HybridRetriever` המבוסס score weighting, אך זה אינו הנתיב הרגיל ב־API.
- **Reranking:** `sentence_transformers.CrossEncoder` עם `cross-encoder/ms-marco-MiniLM-L-6-v2` ב־`src/retrieval/reranker.py`; אם המודל חסר/נכשל, ההתנהגות היא passthrough של הציונים המקוריים.
- **Generation/provider:** `AnswerGenerator` ו־`GenerationResult` (`dataclass`) ב־`src/generation/answer.py`; `LLMProvider` abstraction, `MockLLMProvider` ו־provider selection לפי env ב־`providers/llm.py`; NVIDIA NIM ממומש ב־`providers/llm_nvidia.py`. OpenAI ו־Anthropic שם הם placeholders בלבד. תקציב context מוערך לפי 4 תווים ל־token, לא tokenizer אמיתי.
- **State/contracts:** בקשות ותגובות HTTP הן `Pydantic BaseModel`; agent משתמש ב־`ToolContext`, `ToolSpec`, `ToolCall`, `AgentResult` כ־dataclasses/structures וב־Pydantic schemas לאימות args (`src/agent/tools.py`, `src/agent/agent.py`). מסמכים הם dicts עם `content`, `embedding` ו־`metadata` — לא `TypedDict` מרכזי.
- **Agent pattern:** planner היוריסטי כברירת מחדל, LLM planner אופציונלי; tool allow-list → schema validation → authorization → timeout → audit (`src/agent/tools.py`). למרות מגבלת `MAX_STEPS = 5`, `RAGAgent.run()` מחזיר אחרי tool call יחיד בפועל (`src/agent/agent.py`).

## 4. Testing & Evaluation Reality Check

- `tests/` כולל בדיקות ל־agent, אבטחה, chunker, document lifecycle, end-to-end, generation, key pool, loaders, metrics, NVIDIA provider, observability, RRF ו־streaming. `tests/conftest.py` מאפס את ה־globals של `src.api.main` לפני/אחרי כל test: rate limiter, cache, feedback list ו־pipeline.
- **Mocking:** `tests/test_nvidia_provider.py` משתמש ב־`httpx.MockTransport` — אין רשת או API key אמיתי; הוא בודק payload, auth, usage, timeout/retries, 429/401, streaming ו־structured output. `tests/test_generation_real.py` מזריק `FakeProvider` שמחזיר תשובה ו־usage קבועים. בדיקות agent בונות store/retriever מקומיים; provider מוגדר `mock` בחלק מהן. ה־E2E יוצר מסמך זמני, אך ה־embedding/LLM עשויים להיות mock או fallback בהתאם לסביבה. בדיקות root כמו `test_evaluation.py` ו־`test_advanced_components.py` הן בעיקר בדיקות import/טעינת dataset, לא הוכחת איכות.
- **מה באמת נבדק:** יחידות metrics עם דוגמאות קטנות; end-to-end עם assertions בסיסיים על retrieval/תשובה ומילות מפתח; agent allow-list/validation/admin/timeout/audit; provider דרך HTTP mock. אין כאן הוכחה שכל מודל אמיתי, corpus production או איכות עסקית נבדקו מקצה לקצה.
- **Metrics בקוד:** `src/evaluation/metrics.py` מחשב `hit_rate@5/10`, `recall@5/10`, `precision@5`, `MRR`, `nDCG@5`, `average_precision`; generation: `answer_relevance`, `faithfulness`, `contradiction`, ואופציונלית `answer_completeness`. יש LLM-as-judge או heuristic fallbacks מבוססי overlap. `src/evaluation/run_eval.py` מחשב ממוצעים על שאלות תקינות.
- **הסתייגות קריטית על eval:** `evaluate_single_question()` מסיק `relevant_doc_ids` מה־`expected_answer_contains` בתוך המסמכים שהמערכת כבר החזירה; אם אין התאמה, הוא מסמן את שלושת המסמכים הראשונים כ־pseudo-relevant. לכן retrieval scores אינם מדד עצמאי עם judgments קבועים, ויש לפרש אותם בזהירות. ה־judge יכול להיות `mock`; ה־fallbacks הם heuristics, לא הערכה סמנטית אמינה.

## 5. 5 "Gotcha" Interview Questions (With Code-Grounded Answers)

1. **שאלה:** איך תוודא שהתוצאות הסמנטיות אמינות אם SentenceTransformers לא זמין? האם ה־fallback שקול למודל?
   **תשובה:** לא. `src/embedding/embedder.py` ו־`providers/embeddings.py` מייצרים hash-based vectors; זה שומר על פעולה ודטרמיניזם, אבל אינו ייצוג סמנטי. לפרודקשן הייתי נכשל ב־startup או מסמן במפורש degraded mode, בודק model/dimension/health, ומונע ערבוב אינדקסים שנוצרו במודלים שונים. גם הייתי מוסיף בדיקת איכות retrieval מול eval set קבוע.

2. **שאלה:** האם ה־evaluation מוכיח שה־retriever משתפר, ואם לא — מה חסר?
   **תשובה:** לא באופן בלתי תלוי. ב־`src/evaluation/run_eval.py`, הרלוונטיות נגזרת מהתוצאות עצמן באמצעות expected terms, ובאין התאמה top-3 נהיים pseudo-relevant; זה עלול לנפח או לעוות metrics. נדרש golden set עם relevance judgments שנקבעו מראש ברמת document/chunk, חלוקה לפי query class, והשוואה חוזרת עם confidence/variance; ה־LLM judge צריך להיות מתועד ומכויל מול human labels.

3. **שאלה:** ה־agent מתואר כ־multi-step ויש לו `MAX_STEPS = 5`. האם הוא באמת מתכנן כמה פעולות?
   **תשובה:** לא כרגע. `src/agent/agent.py` — `RAGAgent.run()` מבצע tool call, ואז חוזר מיד במסלולי הצלחה/כשל; לכן loop, step cap ו־repeat cap אינם נותנים multi-step בפועל. גם `_docs_from_search()` בונה מסמכים מה־snippets שהוחזרו, ולא טוען מחדש chunk מלא. הייתי מממש state/continuation מפורש עם תנאי עצירה, timeout כולל, max tool budget, מקורות מלאים ו־tests לתרחישים מרובי־שלבים.

4. **שאלה:** האם ה־RRF נותן משקל 0.7 ל־dense ו־0.3 ל־BM25 כפי שמשתמע מה־API?
   **תשובה:** לא ב־RRF. `src/retrieval/hybrid.py` — `_reciprocal_rank_fusion()` מסכם `1 / (k + rank + 1)` בלבד; `weight_dense` ו־`weight_sparse` נשמרים אך אינם בשימוש. יש גם `HybridRetriever` עם weighted score blending, אבל זה מסלול נפרד. הייתי מסיר/מתקן פרמטרים מטעים, או מיישם weighted RRF ובודק אותו ב־`tests/test_rrf.py` ובהשוואת eval.

5. **שאלה:** מה יקרה ל־cache, feedback והאינדקס כשמריצים כמה API workers או מבצעים ingest במקביל?
   **תשובה:** ה־`query_cache`, `feedback_store`, `rag_pipeline` ו־BM25 state מנוהלים כ־globals בתהליך (`src/api/main.py`, `src/retrieval/sparse_retriever.py`); הם אינם משותפים בין workers ולא שורדים restart. גם pipeline rebinding ב־`initialize_rag_pipeline()` אינו מוגן מפני ingest מקביל. לפרודקשן הייתי מעביר cache ל־Redis עם TTL/version-aware keys, feedback ו־ingestion status ל־DB, ושומר אינדקסים ב־backend משותף; בונה index חדש בגרסה נפרדת ומחליף alias אטומית, עם lock/idempotency ו־health/rollback במקום mutation גלובלי.
