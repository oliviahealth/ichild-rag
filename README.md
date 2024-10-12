<img src="https://raw.githubusercontent.com/oliviahealth/ichild-rag/refs/heads/main/assets/ichild-banner.png?token=GHSAT0AAAAAACF66JDCLKEKN5TTZWWSXYHEZYK4B3Q" />

IntelligentCHILD (Community Health Information Local Database) is an semantic search engine indexing curated resources and content for mothers, children and families.

This repository contains the unified search implementation of the backend server. We define unified search to be a natural language based search of physical locations and general knowledge content.

Physical locations encompass brick-and-mortar destinations including doctor's offices, clinics, food pantries, shelters etc..
The OliviaHealth team has curated a list of resources from several counties in Texas that is indexed by iChild to respond to location based queries. Examples of location based queries include:
<ul>
  <li>Dental Services in Corpus Christi.</li>
  <li>Where can I get mental health support in Bryan?</li>
</ul>

Knowledge Content encompasses the content (including articles, infographics, videos, etc..) in OliviaHealth.org. This content includes general information put out by the OliviaHealth team. This application parses and indexes this information to respond to user queries. Examples of direct queries include:
<ul>
  <li>Newborn nutritonal advice.</li>
  <li>How do hormonal IUDs prevent pregnancy?</li>
  <li>What is mastitis treated with?</li>
</ul>

This implementation of functionality follows the RAG (retrieval-augmented-generation) model of development in which text-based information (such as locations, articles and blogs) are compiled into vectors via an embedding model, stored in a vector database and are retrieved through relevant user queries.

<img src="https://raw.githubusercontent.com/oliviahealth/ichild-rag/refs/heads/main/assets/rag-sequence-diagram.jpg?token=GHSAT0AAAAAACF66JDD5ZZRSMRZE3DH3XUUZYK4TOA" />

<small>In our implementation, 'Retrieved Documents' comprise location records from a SQL table and documents (including articles, blogs, infographics, videos) from OliviaHealth.org.</small>  

<small>Embedding Model is the OpenAI Embedding Model</small>

<small>Vector DB is PostgreSQL (with the PGVector extension)</small>

<small>LLM is gpt-4o-mini</small>

By unified search, we expose a single endpoint, `/search/<id>` for both location based and direct queries. The server then classifies the query as either location based or direct and invokes the proper handler to generate a response. Once the response has been handled properly, the memory is updated in the database.

In addition to providing relevant answers to queries, our model must implement memory to remember conversation history and use it as context for future interactions. This means users can revisit and continue past conversations.