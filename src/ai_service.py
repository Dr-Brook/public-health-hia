"""AI service module — Ollama / LLM integration."""

import os
import requests
from typing import Optional

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "glm-5.1:cloud")


def check_ollama_available() -> bool:
    """Check if Ollama is reachable."""
    try:
        resp = requests.get(OLLAMA_API_URL.replace("/v1", ""), timeout=3)
        return resp.status_code == 200 or resp.status_code == 404  # 404 from / is fine
    except Exception:
        return False


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.7,
             max_tokens: int = 4096) -> str:
    """Call Ollama via OpenAI-compatible API. Returns generated text or error message."""
    try:
        resp = requests.post(
            f"{OLLAMA_API_URL}/chat/completions",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        return "⚠️ Ollama is not running. Please start Ollama and ensure the model is available."
    except requests.exceptions.Timeout:
        return "⚠️ AI request timed out. The model may be loading — try again in a moment."
    except requests.exceptions.HTTPError as e:
        return f"⚠️ AI service error: {e.response.status_code} — {e.response.text[:200]}"
    except Exception as e:
        return f"⚠️ Unexpected AI error: {str(e)[:200]}"


SYSTEM_PUBLIC_HEALTH = """You are a senior public health analyst with expertise in epidemiology, health disparities, social determinants of health, and data-driven public health practice. You hold an MD and MPH from Johns Hopkins Bloomberg School of Public Health.

Your role is to analyze datasets and provide:
1. Key trends and patterns in the data
2. Health disparities identified (across demographic, geographic, or socioeconomic groups)
3. Correlations between variables and their public health significance
4. Evidence-based recommendations for public health action
5. A professional narrative summary suitable for a public health report

Guidelines:
- Use precise public health terminology
- Reference relevant frameworks (social determinants of health, health equity, etc.)
- Quantify findings whenever possible
- Highlight disparities and inequities
- Provide actionable, evidence-based recommendations
- Write in a professional yet accessible tone
- Structure your analysis clearly with headers
- Note limitations of the data where relevant"""


def generate_insights(data_context: str) -> str:
    """Generate AI-powered public health insights from data context."""
    user_prompt = f"""Analyze the following public health dataset and provide a comprehensive analysis.

{data_context}

Please provide:
## Key Trends and Patterns
Identify the most significant trends in the data. Note any temporal patterns, geographic variations, or notable distributions.

## Health Disparities
Identify any health disparities present in the data. Consider differences across regions, income levels, or other social determinants. Discuss equity implications.

## Notable Correlations
Discuss the significant correlations found and their public health meaning. Which relationships are expected? Which are surprising?

## Public Health Recommendations
Based on the analysis, provide 3-5 specific, actionable recommendations for public health intervention or policy. Prioritize by likely impact.

## Summary
Write a 2-3 paragraph professional narrative summary of the key findings, suitable for inclusion in a public health brief or report."""

    return call_llm(SYSTEM_PUBLIC_HEALTH, user_prompt, temperature=0.5)


def generate_chat_response(data_context: str, question: str, chat_history: list = None) -> str:
    """Generate a response to a follow-up question about the data."""
    history_text = ""
    if chat_history:
        for msg in chat_history[-6:]:  # Keep last 3 exchanges
            history_text += f"\n{msg['role'].capitalize()}: {msg['content']}"

    user_prompt = f"""You are answering questions about a public health dataset. Here is the data context:

{data_context}

{f"Previous conversation:{history_text}" if history_text else ""}

Current question: {question}

Provide a data-driven answer. Reference specific numbers from the data when possible. If the data doesn't contain information to answer the question, say so clearly. Use public health expertise to provide context and interpretation."""

    return call_llm(SYSTEM_PUBLIC_HEALTH, user_prompt, temperature=0.4)