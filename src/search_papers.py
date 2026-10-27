
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

OPENALEX_URL = "https://api.openalex.org/works"

# Optional: OpenAlex key, separate from Gemini
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")


def search_papers(
    topic,
    limit=5,
    year_from=None,
    year_to=None
):
    if not 1 <= limit <= 100:
        raise ValueError("Paper limit must be between 1 and 100.")

    if (
        year_from is not None
        and year_to is not None
        and year_from > year_to
    ):
        raise ValueError("Invalid publication year range.")

    params = {
        "search": topic,
        "per_page": limit
    }

    # Build publication-date filters
    filters = []

    if year_from is not None:
        filters.append(
            f"from_publication_date:{year_from}-01-01"
        )

    if year_to is not None:
        filters.append(
            f"to_publication_date:{year_to}-12-31"
        )

    if filters:
        params["filter"] = ",".join(filters)


    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY

    response = requests.get(
        OPENALEX_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    results = response.json().get("results", [])

    papers = []

    for paper in results:
        oa_location = paper.get("best_oa_location") or {}

        pdf_url = oa_location.get("pdf_url")

        if not pdf_url:
            for location in paper.get("locations", []):
                if location.get("is_oa") and location.get("pdf_url"):
                    pdf_url = location["pdf_url"]
                    break

        papers.append({
            "paper_id": paper.get("id"),
            "title": paper.get("title"),
            "year": paper.get("publication_year"),
            "doi": paper.get("doi"),
            "pdf_url": pdf_url,
        })

    return papers


if __name__ == "__main__":

    topic = input("Enter research topic: ").strip()

    if not topic:
        raise ValueError("Please enter a research topic.")

    try:
        papers = search_papers(topic)

        print("\n" + "=" * 60)
        print("RESEARCHGAP AI - PAPER DISCOVERY")
        print("=" * 60)

        for i, paper in enumerate(papers, start=1):
            print(f"\nPaper {i}")
            print("Title:", paper["title"])
            print("Year:", paper["year"])
            print("OpenAlex ID:", paper["paper_id"])
            print("PDF:", paper["pdf_url"] or "Not found")

        print("\nTotal papers found:", len(papers))

    except requests.RequestException as e:
        print("OpenAlex request failed:", e)
