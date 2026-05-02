from typing import List, Dict, Any
import numpy as np
from langchain_ollama import OllamaEmbeddings


class RAGStore:
    """
    Almacén vectorial en memoria usando embeddings de OpenAI.
    Se construye una vez al finalizar el workflow y se consulta
    en el chat posterior.
    """

    def __init__(self, agent):
        self.agent = agent        # instancia de Agents para embeddings y chat
        self.chunks: List[str] = []
        self.embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
        self.embeddings: List[np.ndarray] = []
        self.built = False

    # ------------------------------------------------------------------ #
    #  Construcción del índice                                             #
    # ------------------------------------------------------------------ #

    def _embed(self, texts: List[str]) -> List[np.ndarray]:
        """Obtiene embeddings usando LangChain OpenAIEmbeddings."""
        return [np.array(self.embeddings_model.embed_query(t)) for t in texts]

    def _chunk_from_analysis(self, item: Dict[str, Any]) -> str:
        """Convierte un item de análisis en texto plano para vectorizar."""
        return (
            f"Título: {item.get('title', '')}\n"
            f"Resumen: {item.get('summary', '')}\n"
            f"Puntos clave: {item.get('key_points', '')}\n"
            f"Relevancia: {item.get('relevance', '')} | "
            f"Credibilidad: {item.get('credibility', '')}\n"
            f"Fuente: {item.get('source', '')} — {item.get('link', '')}"
        )

    def build(self, analysis: List[Dict[str, Any]], claim: str, final_answer: str) -> None:
        """Vectoriza todo el análisis + el reporte final."""
        if not analysis:
            return

        # Chunk extra con el reporte final para que el agente lo tenga como contexto
        report_chunk = f"REPORTE FINAL sobre '{claim}':\n{final_answer}"

        self.chunks = [self._chunk_from_analysis(item) for item in analysis]
        self.chunks.append(report_chunk)

        self.embeddings = self._embed(self.chunks)
        self.built = True
        print(f"[RAG] Índice construido con {len(self.chunks)} chunks.")

    # ------------------------------------------------------------------ #
    #  Recuperación                                                        #
    # ------------------------------------------------------------------ #

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def retrieve(self, query: str, top_k: int = 4) -> List[str]:
        """Devuelve los top_k chunks más relevantes para la query."""
        if not self.built or not self.chunks:
            return []

        query_emb = self._embed([query])[0]
        scores = [self._cosine_similarity(query_emb, emb) for emb in self.embeddings]
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.chunks[i] for i in top_indices]

    # ------------------------------------------------------------------ #
    #  Chat con contexto                                                   #
    # ------------------------------------------------------------------ #

    def chat(self, history: List[Dict[str, str]], user_message: str, claim: str) -> str:
        """
        Responde al usuario usando los chunks recuperados como contexto.
        history: lista de {'role': ..., 'content': ...} para memoria de conversación.
        """
        context_chunks = self.retrieve(user_message)
        context = "\n\n---\n\n".join(context_chunks) if context_chunks else "Sin contexto disponible."

        system_prompt = (
            f"Eres un asistente experto en verificación de hechos.\n"
            f"Se analizó la siguiente noticia: \"{claim}\"\n\n"
            f"Usa exclusivamente el siguiente contexto para responder. "
            f"Si la respuesta no está en el contexto, dilo claramente.\n\n"
            f"CONTEXTO:\n{context}"
        )

        messages = [{"role": "system", "content": system_prompt}] + history + [
            {"role": "user", "content": user_message}
        ]

        return self.agent.invoke(messages)