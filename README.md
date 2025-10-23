
# Setup

Befülle die `.env` Datei mit den notwendigen Schlüsseln und starte die App via:

```sh
uv run streamlit run main.py
```

# Aufgaben

1. `get_retriever` (siehe `utils.py`)
2. `get_response_generator` (siehe `utils.py`)
3. Bonus:
    * eigene Ideen / Verbesserungen der App
    * Aufgabe: Query router (keine pdfs retrieven falls nicht relevant)
    * Aufgabe: neuer Button um weitere Dokumente hinzuzufügen
    * Aufgabe: Implementiere Query Expansion:
        * Generiere 2-3 alternative Formulierungen der Frage
        * Rufe Dokumente für alle Varianten ab
    * Aufgabe: Füge Chat-History hinzu
        * Nächster Schritt: Implementiere Context-Reformulation (Frage + History →  Standalone-Frage)
    * Aufgabe: Re-implementiere das RAG System in Langgraph, und erweitere
        das System mit Tool-calling
    * Aufgabe: verwende ein lokales LLM
    * Aufgabe: Implementiere einen "Reranking step"
