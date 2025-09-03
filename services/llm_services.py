from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """Base class for LLM services."""
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt (str): The formatted prompt with observations and actions
            **kwargs: Model-specific parameters (temperature, max_tokens, etc.)
            
        Returns:
            str: Response in format "Action, Arguments"
        """
        pass

class OpenAILLM(BaseLLM):
    """OpenAI API implementation."""
    
    def __init__(self, model="gpt-4o-mini", temperature=0.7, api_key=None, log_dir="logs", log_file="llm_conversation.txt"):
        try:
            import openai
            import os
        except ImportError as e:
            raise ImportError(f"Required package not found: {e}")
            
        self.model = model
        self.temperature = temperature
        if api_key:
            openai.api_key = api_key
        self.client = openai.OpenAI()
        
        self.log_dir = log_dir
        self.log_file = log_file
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_file)
        
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an AI agent in an economic simulation. You must respond with exactly one action in the format 'Action, Arguments'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                **kwargs
            )
            result = response.choices[0].message.content.strip()
            with open(self.log_path, 'a') as f:
                f.write(f"\n{'='*50}\nPROMPT:\n{prompt}\n\nRESPONSE:\n{result}\n{'='*50}\n")
            return result
        except Exception as e:
            print(f"Error generating response from OpenAI: {e}")
            return "Nothing, Nothing"  # Safe fallback

class OllamaLLM(BaseLLM):
    """Ollama local LLM implementation."""
    
    def __init__(self, model="llama3.2:1b", temperature=0.7, base_url="http://localhost:11434", log_dir="logs", log_file="llm_conversation.txt"):
        try:
            import requests
            import json
            import os
        except ImportError as e:
            raise ImportError(f"Required package not found: {e}")
            
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_file)
        
        try:
            response = requests.get(f"{base_url}/api/tags")
            if response.status_code != 200:
                raise ConnectionError(f"Cannot connect to Ollama at {base_url}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Ollama is not running at {base_url}. Start it with 'ollama serve'")
    
    def generate_response(self, prompt, **kwargs):
        try:
            import requests
            import json
            
            response = requests.post(f"{self.base_url}/api/generate", 
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        **kwargs
                    }
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
                
            result = response.json()["response"].strip()
            
            # Log the conversation
            with open(self.log_path, 'a') as f:
                f.write(f"\n{'='*50}\nMODEL: {self.model}\nPROMPT:\n{prompt}\n\nRESPONSE:\n{result}\n{'='*50}\n")
            
            return result
            
        except Exception as e:
            print(f"Error generating response from Ollama: {e}")
            return "Nothing, Nothing"  # Safe fallback

class AnthropicLLM(BaseLLM):
    """Anthropic API implementation."""
    
    def __init__(self, model="claude-3-opus", api_key=None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Anthropic package not found. Install with 'pip install anthropic'")
            
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using Anthropic API."""
        try:
            response = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Error generating response from Anthropic: {e}")
            return "Nothing, Nothing"  # Safe fallback

def create_llm(config: dict) -> BaseLLM:
    """
    Factory function to create LLM instance based on config.
    
    Args:
        config (dict): Configuration containing LLM settings
            Required keys:
            - type: str, The type of LLM ("openai", "anthropic")
            Optional keys:
            - model: str, Model name
            - temperature: float, Temperature for sampling
            - api_key: str, API key if needed
            
    Returns:
        BaseLLM: An instance of the specified LLM class
    """
    llm_type = config.get("type", "openai").lower()
    api_key = config.get("api_key")
    
    if llm_type == "ollama":
        model = config.get("model", "llama3.2:1b")
        temperature = config.get("temperature", 0.7)
        base_url = config.get("base_url", "http://localhost:11434")
        log_dir = config.get("log_dir", "logs")
        log_file = config.get("log_file", "llm_conversation.txt")
        return OllamaLLM(
            model=model,
            temperature=temperature,
            base_url=base_url,
            log_dir=log_dir,
            log_file=log_file
        )
    elif llm_type == "openai":
        model = config.get("model", "gpt-4")
        temperature = config.get("temperature", 0.7)
        log_dir = config.get("log_dir", "logs")
        log_file = config.get("log_file", "llm_conversation.txt")
        return OpenAILLM(
            model=model,
            temperature=temperature,
            api_key=api_key,
            log_dir=log_dir,
            log_file=log_file
        )
    
    elif llm_type == "anthropic":
        model = config.get("model", "claude-3-opus")
        return AnthropicLLM(model=model, api_key=api_key)
    
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")
