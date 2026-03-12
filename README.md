# TORTOISE RESEARCH AGENT

## FOR RUN CONTAINER EXECUTES:


```bash
docker run -it --name research-agent --gpus all \
  -e OPENAI=xxx \
  -e GROQ=xxx \
  -e OPENALEX=xxx \
  -e GOOGLESCHOLAR=xxx \
  research-agent_image
```