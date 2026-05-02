from dotenv import load_dotenv
import os
import time
load_dotenv()

from app import ResearchAgent
from app.rag import RAGStore
from rich.console import Console
from rich.markdown import Markdown

if __name__ == "__main__":
    agent = ResearchAgent()
    os.system('cls' if os.name == 'nt' else 'clear')

    console = Console()
    colors = ["red", "yellow", "green", "cyan", "blue", "magenta"]

    print("\n")
    with open("app/ascii_art/name.txt") as f:
        for i, line in enumerate(f):
            console.print(line.rstrip(), style=colors[i % len(colors)])
    with open("app/ascii_art/logo.txt") as f:
        console.print(f.read(), style=colors[3])

    print("Let me check a Fact: \n")

    rag: RAGStore | None = None
    chat_history: list = []
    current_claim: str = ""

    while True:
        topic = input("User: ").strip()
        print("\n" + "=" * 50)

        if topic == "/bye":
            break

        # Comando para limpiar y analizar una nueva noticia
        if topic == "/new":
            rag = None
            chat_history = []
            current_claim = ""
            console.print("Contexto limpiado. Ingresa una nueva noticia.", style="bold yellow")
            continue

        # Si no hay RAG activo, tratar el input como una nueva noticia a verificar
        if rag is None:
            current_claim = topic
            console.print("\n--- Agent Report ---\n", style="bold green")

            init = time.time()
            result, final_analysis = agent.run(topic)   # ver cambio en agent.py abajo
            end = time.time()

            console.print(Markdown(result))
            print("\n" + 50 * "=")
            print(f"Time: {end - init:.2f}s")
            print("\n" + "=" * 50)

            # Construir índice RAG con todo lo recopilado
            rag = RAGStore(agent.agent)
            rag.build(final_analysis, current_claim, result)

            console.print(
                "\n[RAG activo] Ahora puedes hacer preguntas sobre el análisis. "
                "Escribe /new para verificar otra noticia.\n",
                style="bold cyan"
            )

        # Si hay RAG activo, responder en modo chat
        else:
            response = rag.chat(chat_history, topic, current_claim)

            chat_history.append({"role": "user", "content": topic})
            chat_history.append({"role": "assistant", "content": response})

            console.print("\n--- Respuesta ---\n", style="bold green")
            console.print(Markdown(response))
            print("\n" + "=" * 50)