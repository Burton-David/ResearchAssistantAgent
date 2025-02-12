📚 AI-Powered Research Paper Analysis

🚀 Accelerating Scientific Discovery with AI, LangChain, and Vector Databases

 

 

 

🔍 Finding high-quality research papers shouldn’t be hard. This project leverages LLMs, LangChain, FAISS vector search, and scientific databases to build an AI-powered research assistant that helps scientists find high-quality citations for research grant applications, literature reviews, and academic exploration.

	💡 Why This Matters?
Researchers spend hours searching for credible citations. This tool automates and enhances the process by ranking papers based on:
		•	Relevance (semantic search via embeddings)
	•	Citation Impact (highly cited papers)
	•	Methodology (identifying strong research techniques)
	•	Recency (fresh papers)
	•	Venue Quality (high-impact journals/conferences)

🚀 Features

✅ Multi-Source Search: Queries arXiv and Semantic Scholar
✅ Vector Search: Uses FAISS (with Pinecone support coming soon)
✅ LLM Analysis: OpenAI embeddings help rank research papers
✅ Citations & Quality Scoring: Finds the best sources for grant proposals
✅ Asynchronous Processing: Fast and scalable
✅ Extensible: Modular design for adding new data sources
✅ Docker Support: Easy setup with docker-compose

📦 Installation

1️⃣ Clone the Repository

git clone https://github.com/Burton-David/ResearchAssistantAgent
cd ResearchAssistantAgent

2️⃣ Set Up the Environment

Ensure you have Python 3.8+ installed.

python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt

3️⃣ API Key Setup

You’ll need API keys for Semantic Scholar and OpenAI. Create a .env file in the config/ directory:

mkdir config && touch config/.env

Add the following to .env:

OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key

4️⃣ Run the Research Collector

python main.py

🐳 Docker Deployment

For an isolated, ready-to-use setup, run:

docker-compose up --build

This ensures all dependencies, including FAISS, are set up inside a container.

🛠️ How It Works

🔹 1. Collect Papers

papers = await paper_collector.fetch_papers("climate change AI", max_results=50)

	•	Fetches from Semantic Scholar & arXiv
	•	Uses pagination to get full results
	•	Filters for high-quality research

🔹 2. Store in Vector Database

await paper_collector._store_in_vector_db(papers)

	•	Converts papers to embeddings using OpenAI
	•	Stores them in FAISS (local, fast, scalable)

🔹 3. Perform AI-Driven Analysis

results = await analyzer.analyze_research_topic("climate change AI")

	•	Ranks papers based on quality metrics
	•	Uses similarity search to find related papers

📍 Roadmap

✔️ MVP: Fetch papers + basic FAISS search ✅
🔜 LangChain Summarization: AI-generated research summaries 🔄
🔜 Pinecone Support: Scalable cloud-based vector search
🔜 Full-Text Embeddings: Extract knowledge beyond abstracts
🔜 Web UI: Interactive dashboard for exploring research

	Got feature ideas? Open an issue or submit a pull request! 🚀

🧑‍💻 Contributing

Want to help make research easier? Contributions are welcome!
	1.	Fork this repo
	2.	Create a feature branch (git checkout -b feature-name)
	3.	Commit your changes (git commit -m "Added new feature")
	4.	Push to GitHub (git push origin feature-name)
	5.	Open a pull request

📜 License

This project is MIT Licensed – free to use, modify, and contribute.

🚀 Join us in making research faster, smarter, and better.
