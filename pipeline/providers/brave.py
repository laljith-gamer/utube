"""Brave Search API Provider for discovery and images."""
from __future__ import annotations

import logging
import os
import requests

from ..utils import env

LOG = logging.getLogger("utube.brave")

class BraveProvider:
    BASE_URL = "https://api.search.brave.com/res/v1"
    
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
        try:
            r = requests.get(
                f"{cls.BASE_URL}/news/search",
                params={"q": query, "count": min(count, 20), "freshness": freshness},
                headers=cls._headers(),
                timeout=10
            )
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
        try:
            r = requests.get(
                f"{cls.BASE_URL}/images/search",
                params={"q": query, "count": min(count, 50), "safesearch": "moderate"},
                headers=cls._headers(),
                timeout=10
            )
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
