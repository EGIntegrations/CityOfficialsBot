
⸻

# 📖 Ozark City Ordinance Chatbot

This chatbot assists citizens by providing direct quotations from Ozark City ordinances. It is explicitly restricted to quote ordinance text to ensure accuracy and legality.

## 🚀 Quick Overview

- **Strictly ordinance-specific responses** (quotes ONLY, no opinions).
- Built with Python, LangChain, OpenAI, FAISS, and Streamlit.
- Easy integration into the City of Ozark website.

---

## 🛠 Local Setup (Step-by-Step)

**Clone this repo locally:**
```
bash
git clone <your-repo-link>
cd ozark-ordinance-chatbot
```
Create virtual environment (recommended):
```
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```
Install dependencies:
```
pip install -r requirements.txt
```
API Setup (Important!)

Create a .env file in your root directory:
```
OPENAI_API_KEY=your_openai_api_key_here
```


⸻

📖 Preparing Embeddings (Important first-time setup)

Place your PDF (ordinances.pdf) into the root directory.

Run:
```
python prepare_pdf.py
```
This generates embeddings in faiss_index/ (local use only, don’t push to GitHub).

⸻

💻 Run Chatbot Locally (CLI)
```
python chatbot.py
```
	•	Type your questions directly into the terminal.
	•	Responses strictly quote from ordinance documents.

⸻

🌐 Run Web Interface (Streamlit)
```
streamlit run app.py
```
Open your browser:
```
http://localhost:8501
```

⸻

🚨 Important Legal Notice
	•	The chatbot ONLY quotes directly from ordinances.
	•	It does NOT provide interpretations or opinions.
	•	Always consult city officials for clarification.

⸻

🌎 Deploying to Streamlit Cloud (For Easy Testing)
	1.	Push this repo to GitHub.
	2.	Create a free Streamlit Cloud account.
	3.	Click “New App”, select your GitHub repository.
	4.	Enter app.py as the entry point.
	5.	Deploy to get a publicly shareable URL.

⸻

🚦 Embedding on a City's Website

City IT administrators can embed the chatbot directly into the city’s official website using an iframe:
```
<iframe src="https://your-streamlit-app.streamlit.app" width="100%" height="800px"></iframe>
```
Or use DNS management to create a subdomain pointing directly to the Streamlit app URL.

⸻

🔍 Updating Ordinances

Whenever ordinances are updated:
	1.	Replace ordinances.pdf with the updated PDF.
	2.	Re-run the embedding script:
```
python prepare_pdf.py
```
	3.	Restart or redeploy your app.

⸻

⚠️ Troubleshooting Common Errors
	•	“Pickle deserialization error”:
Update load method explicitly:
```
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
```
	•	Missing Streamlit command:
Activate your virtual environment first:
```
source venv/bin/activate
```


⸻

📌 Dependency List (requirements.txt)
```
streamlit
langchain
langchain-openai
openai
faiss-cpu
python-dotenv
pypdf
```


⸻

🗂 Best Practices
	•	Keep .env and sensitive files off GitHub.
	•	Update dependencies regularly with pip freeze > requirements.txt.

⸻

🤝 Contributing & Maintenance

Contact City IT or egintegrations@gmail.com for support or contributions.

⸻

📜 License

Egintegrations © 2024. All rights reserved.

---

# 🚩 **Commands to create GitHub Repo & Push Code:**
```
bash
git init
git add .
git commit -m "Initial chatbot commit"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```


⸻
