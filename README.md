<img src="https://raw.githubusercontent.com/oliviahealth/ichild-rag/refs/heads/main/assets/ichild-banner.png?token=GHSAT0AAAAAACF66JDCLKEKN5TTZWWSXYHEZYK4B3Q" />

IntelligentCHILD (Community Health Information Local Database) is an semantic search engine indexing curated resources and content for mothers, children and families.

This repository contains the unified search implementation of the backend server. We define unified search to be a natural language based search of physical locations and general knowledge content.

Physical locations encompass brick-and-mortar destinations including doctor's offices, clinics, food pantries, shelters etc..
The OliviaHealth team has curated a list of resources from several counties in Texas that is indexed by iChild to respond to location based queries.

Knowledge Content encompasses the content (including articles, infographics, videos, etc..) in OliviaHealth.org. This content includes general information put out by the OliviaHealth team. This application parses and indexes this information to respond to user queries.

This implementation of functionality follows the RAG (retrieval-augmented-generation) model of development in which text-based information (such as locations, articles and blogs) are compiled into vectors via an embedding model, stored in a vector database and are retrieved through relevant user queries.