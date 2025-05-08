#!/bin/bash

echo "Starting Ollama server..."
ollama serve &  # Start Ollama in the background

echo "Ollama is ready, creating the model..."

if [ -z "$LLM_NAME" ]; then
  echo "Error: LLM_NAME environment variable is not set."
  exit 1
fi
ollama create $LLM_NAME -f ./model_files/Modelfile
ollama run $LLM_NAME
