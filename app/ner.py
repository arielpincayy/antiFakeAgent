from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import torch
from pathlib import Path
from typing import List, Dict, Any, Set
from itertools import combinations


class NERModel(torch.nn.Module):
    def __init__(self):
        super(NERModel, self).__init__()
        BASE_DIR = Path(__file__).resolve().parent.parent
        model_path = BASE_DIR / "app/modelo_ner_metricas/checkpoint-759"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1,
        )

    def _extract_query_parts(self, texto: str, base: str) -> Dict[str, Set[str]]:
        """Corre el NER y agrupa los términos extraídos por tipo de entidad."""
        allowed_types = (
            {"TECHNOLOGY", "METHOD", "ORGANIZATION"}
            if base == "scientific"
            else {"TECHNOLOGY", "METHOD", "ORGANIZATION", "PERSON", "LOCATION"}
        )

        query_parts: Dict[str, Set[str]] = {t: set() for t in allowed_types}

        for entidad in self.ner_pipeline(texto):
            tipo = entidad["entity_group"]
            palabra = entidad["word"].replace("##", "").replace("Ġ", "").strip()
            if tipo in query_parts and palabra:
                query_parts[tipo].add(palabra)

        # Eliminar tipos sin términos
        return {t: terms for t, terms in query_parts.items() if terms}

    @staticmethod
    def _build_group(tipo: str, terminos: Set[str]) -> str:
        """Construye un bloque OR para un tipo de entidad: (\"a\" OR \"b\")"""
        return "(" + " OR ".join(f'"{t}"' for t in sorted(terminos)) + ")"

    def _generate_queries(self, query_parts: Dict[str, Set[str]], min_groups: int = 1) -> List[str]:
        """
        Genera todas las combinaciones posibles de grupos de entidades.
        Cada combinación produce una query AND entre los grupos seleccionados.

        Ejemplo con 3 grupos [T, M, O]:
          - combinaciones de 1: T | M | O
          - combinaciones de 2: T AND M | T AND O | M AND O
          - combinación  de 3: T AND M AND O
        """
        tipos = list(query_parts.keys())
        queries: List[str] = []

        for r in range(min_groups, len(tipos) + 1):
            for combo in combinations(tipos, r):
                groups = [self._build_group(t, query_parts[t]) for t in combo]
                queries.append(" AND ".join(groups))

        # Eliminar duplicados manteniendo orden
        seen: set = set()
        unique = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique.append(q)

        return unique

    def generar_queries_cientificas(self, texto: str, base: str, min_groups: int = 1) -> tuple[List[str], List[Dict[str, Any]]]:
        """
        Extrae entidades del texto y genera todas las combinaciones de queries posibles.

        Args:
            texto:      Texto de la noticia a analizar.
            base:       'scientific' limita los tipos a TECHNOLOGY, METHOD, ORGANIZATION.
                        Cualquier otro valor incluye también PERSON y LOCATION.
            min_groups: Mínimo de grupos que debe tener cada query (default 1).

        Returns:
            queries:    Lista de queries ordenadas de menor a mayor especificidad.
            resultados: Entidades crudas devueltas por el NER pipeline.
        """
        resultados = self.ner_pipeline(texto)
        query_parts = self._extract_query_parts(texto, base)

        if not query_parts:
            return [], resultados

        queries = self._generate_queries(query_parts, min_groups=min_groups)
        return queries, resultados

    def get_queries_and_entities(self, texto: str, base: str) -> tuple[List[str], List[Dict[str, Any]]]:
        return self.generar_queries_cientificas(texto, base)