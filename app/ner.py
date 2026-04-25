from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import torch
from pathlib import Path

from typing import List, Dict, Any

class NERModel(torch.nn.Module):
    """
    This class defines a Named Entity Recognition (NER) model using Hugging Face's Transformers library.
    It loads a pre-trained model and tokenizer from a specified checkpoint and provides methods to extract entities
    from input text and to generate scientific queries based on the extracted entities.
    """
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

    def generar_query_cientifica(self, texto: str) -> tuple[str, List[Dict[str, Any]]]:
        resultados = self.ner_pipeline(texto)
        
        # Diccionarios para organizar los términos de la query
        query_parts = {
            "TECHNOLOGY": set(),
            "METHOD": set(),
            "ORGANIZATION": set(),
            "PERSON": set(),
            "LOCATION": set()
        }
        
        for entidad in resultados:
            tipo = entidad['entity_group']
            palabra = palabra = (
                        entidad['word']
                        .replace("##", "")
                        .replace("Ġ", "")
                        .strip()
                    )
            
            if tipo in query_parts:
                query_parts[tipo].add(palabra)
                
        sub_queries = []
        for tipo, terminos in query_parts.items():
            if terminos:
                # Unimos términos del mismo tipo con OR
                group = "(" + " OR ".join(f'"{t}"' for t in terminos) + ")"
                sub_queries.append(group)
        
        query_final = " AND ".join(sub_queries)
        
        return query_final, resultados
     
    def get_query_and_entities(self, texto: str) -> tuple[str, List[Dict[str, Any]]]:
        query, entities = self.generar_query_cientifica(texto)

        return query, entities