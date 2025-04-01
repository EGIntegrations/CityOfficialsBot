
# 🏛️ City Officials Ordinance Reference Bot

This chatbot assists city officials in quickly and accurately referencing and comparing ordinances from multiple municipalities. It strictly provides direct quotations along with useful metadata, such as the ordinance timestamp, category, and surrounding context.

---

## 🚩 **Key Features:**

- **Direct Quotations Only:** No opinions, interpretations, or paraphrasing.
- **Metadata Enhanced Responses:** Timestamp, category, and surrounding context provided clearly.
- **Multiple City Ordinances:** Easily compare ordinances across different cities.

---

## 🛠️ **Local Setup (Clearly Step-by-Step):**

**1. Clone the repository:**
```
bash
git clone <your-repo-url>
cd CityOfficialsBot
```
2. Set up the Python environment:
```
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```
3. Install dependencies:
```
pip install -r requirements.txt
```
4. Add your OpenAI API key:
	•	Create a file named .env in the project root clearly containing:
```
OPENAI_API_KEY=your_openai_api_key_here
```


⸻

📚 Generate Ordinance Embeddings:
	•	Place all PDFs you want to reference clearly into the ordinances/ folder.

Run embedding script explicitly:
```
python prepare_pdf.py
```
(This generates the faiss_index/ directory used for ordinance lookups.)

⸻

🚀 Run the Chatbot Locally (Streamlit):
```
streamlit run app.py
```
	•	Visit the application at: http://localhost:8501

⸻

🌐 Deploying to Streamlit Cloud (Recommended):
	1.	Push your repo clearly to GitHub:
```
git add .
git commit -m "Prepared chatbot for deployment"
git push
```
	2.	Sign into Streamlit Cloud
	3.	Deploy directly by selecting your GitHub repository and setting the main file as app.py.

⸻

⚠️ Important Usage Notes (Strict Legal Guidelines):
	•	This chatbot provides only exact ordinance quotations.
	•	It never offers interpretation or legal advice.
	•	Always confirm with official city resources when making decisions.

⸻

📌 Adding or Updating Ordinances:

To add or update ordinances clearly:
	1.	Put new or updated PDFs into the ordinances/ directory.
	2.	Run the embeddings script again:

python prepare_pdf.py

	3.	Restart your Streamlit application or redeploy.

⸻

📦 Dependencies and Versions:
	•	Python: 3.11.7
	•	Streamlit: 1.33.0
	•	LangChain: 0.1.17
	•	LangChain-OpenAI: 0.0.8
	•	OpenAI: 1.6.1
	•	FAISS-CPU: 1.8.0
	•	Python-dotenv: 1.0.1
	•	PyPDF: 4.0.2
	•	Tiktoken: 0.5.2

⸻

📬 Support and Contact Information:
	•	Maintained by City of Ozark IT Department.
	•	For technical assistance, contact: your-email@example.com

⸻

📝 License:

City of Ozark © 2025. All Rights Reserved.

---
