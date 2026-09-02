"""Brave Search API Provider for discovery and images."""
from __future__ import annotations

import logging
import os
import requests

from ..utils import env

LOG = logging.getLogger("utube.brave")

class BraveProvider:
    BASE_URL = "https://api.search.brave.com/res/v1"
    MAX_REQUESTS_PER_RUN = 25
    _request_count = 0
    _web_exhausted = False
    _answers_exhausted = False
    
    @classmethod
    def _key(cls) -> str:
        key = env("BRAVE_API_KEY")
        if not key:
            raise RuntimeError("BRAVE_API_KEY is not set.")
        return key
        
    @classmethod
    def _headers(cls) -> dict:
        return {
            "Accept": "application/json",
            "X-Subscription-Token": cls._key()
        }

    @classmethod
    def search_news(cls, query: str, count: int = 10, freshness: str = "pd") -> list[dict]:
        """
        Search Brave News API.
        freshness: 'pd' (past day), 'pw' (past week), 'pm' (past month)
        """
        if cls._web_exhausted or cls._request_count >= cls.MAX_REQUESTS_PER_RUN:
            LOG.warning("Brave Web API exhausted or limit reached. Skipping News request.")
            return []
            
        try:
            cls._request_count += 1
            r = requests.get(
                f"{cls.BASE_URL}/news/search",
                params={"q": query, "count": min(count, 20), "freshness": freshness},
                headers=cls._headers(),
                timeout=10
            )
            if r.status_code in (402, 403, 429):
                LOG.warning("Brave Web API quota exceeded (HTTP %d). Disabling for this run.", r.status_code)
                cls._web_exhausted = True
                return []
            r.raise_for_status()
            data = r.json()
            
            results = data.get("results", [])
            out = []
            for item in results:
                out.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "summary": item.get("description", ""),
                    "age": item.get("age", ""),
                    "source": "brave_news",
                    "source_class": "tier1",
                    "credibility_tier": 1,
                    "score": 100, # Base score
                })
            return out
        except Exception as e:
            LOG.error("Brave News search failed for query '%s': %s", query, e)
            return []

    @classmethod
    def search_images(cls, query: str, count: int = 5) -> list[dict]:
        """Search Brave Images API."""
        if cls._web_exhausted or cls._request_count >= cls.MAX_REQUESTS_PER_RUN:
            LOG.warning("Brave Web API exhausted or limit reached. Skipping Image request.")
            return []
            
        try:
            cls._request_count += 1
            r = requests.get(
                f"{cls.BASE_URL}/images/search",
                params={"q": query, "count": min(count, 50), "safesearch": "moderate"},
                headers=cls._headers(),
                timeout=10
            )
            if r.status_code in (402, 403, 429):
                LOG.warning("Brave Web API quota exceeded (HTTP %d). Disabling for this run.", r.status_code)
                cls._web_exhausted = True
                return []
            r.raise_for_status()
            data = r.json()
            
            results = data.get("results", [])
            out = []
            for item in results:
                out.append({
                    "title": item.get("title", ""),
                    "url": item.get("properties", {}).get("url", ""), # Original image URL
                    "thumbnail": item.get("thumbnail", {}).get("src", ""),
                    "source": item.get("source", ""),
                })
            return out
        except Exception as e:
            LOG.error("Brave Image search failed for query '%s': %s", query, e)
            return []

    @classmethod
    def search_web(cls, query: str, count: int = 5) -> list[dict]:
        """Search Brave Web API for snippets and context."""
        if cls._web_exhausted or cls._request_count >= cls.MAX_REQUESTS_PER_RUN:
            LOG.warning("Brave Web API exhausted or limit reached. Skipping Web request.")
            return []
            
        try:
            cls._request_count += 1
            r = requests.get(
                f"{cls.BASE_URL}/web/search",
                params={"q": query, "count": min(count, 10), "safesearch": "moderate"},
                headers=cls._headers(),
                timeout=10
            )
            if r.status_code in (402, 403, 429):
                LOG.warning("Brave Web API quota exceeded (HTTP %d). Disabling for this run.", r.status_code)
                cls._web_exhausted = True
                return []
            r.raise_for_status()
            data = r.json()
            
            results = data.get("web", {}).get("results", [])
            out = []
            for item in results:
                out.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("description", ""),
                    "extra_snippets": item.get("extra_snippets", [])
                })
            return out
        except Exception as e:
            LOG.error("Brave Web search failed for query '%s': %s", query, e)
            return []

    @classmethod
    def get_answer(cls, query: str) -> str:
        """Get an AI-generated answer from Brave Search Answers API."""
        if cls._answers_exhausted or cls._request_count >= cls.MAX_REQUESTS_PER_RUN:
            LOG.warning("Brave Answers API exhausted or limit reached. Skipping request.")
            return ""
            
        try:
            cls._request_count += 1
            r = requests.post(
                f"{cls.BASE_URL}/chat/completions",
                headers=cls._headers(),
                json={
                    "messages": [{"role": "user", "content": f"Give me a highly detailed summary and fascinating facts about: {query}"}],
                    "model": "brave",
                    "stream": False,
                },
                timeout=30
            )
            if r.status_code in (402, 403, 429):
                LOG.warning("Brave Answers API quota exceeded (HTTP %d). Disabling for this run.", r.status_code)
                cls._answers_exhausted = True
                return ""
            r.raise_for_status()
            data = r.json()
            
            choices = data.get("choices", [])
            if not choices:
                return ""
                
            return choices[0].get("message", {}).get("content", "")
        except Exception as e:
            LOG.error("Brave Answers request failed for query '%s': %s", query, e)
            return ""

