from dotenv import load_dotenv
import os

load_dotenv()  # Carga las variables de entorno desde el archivo .env

from app import ResearchAgent

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

    print("What topic do you want to investigate?: \n")
    while(True):
        topic = input("User: ")
        print("\n" + "="*50)
        if(topic == "/bye"): break
        console.print("\n--- Agent Report ---\n", style="bold green")
        result = agent.run(topic)
        console.print(Markdown(result))
        print("\n" + "="*50)