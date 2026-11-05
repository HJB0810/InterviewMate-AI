# InterviewMate AI – Interview Trainer Agent

An AI-powered interview preparation platform that helps students and job seekers practise technical and HR interviews, receive personalized feedback, and improve their interview performance using IBM Granite.

## 1. Project Overview

InterviewMate AI is an intelligent interview preparation system designed to simulate an interview environment. It allows candidates to select their preferred interview type, enter their skills and experience, answer interview questions, and receive AI-generated feedback.

The application uses **IBM Granite through IBM watsonx.ai** to generate interview questions and evaluate candidate answers. It also uses a document-based **Retrieval-Augmented Generation (RAG)** approach to provide relevant interview knowledge to the AI model.

The main objective is to make interview preparation more accessible, structured, and personalized for students and job seekers.

## 2. Problem Statement

Many students and fresh graduates face difficulties preparing for technical and HR interviews because:

- They do not have regular access to interview practice.
- They may not know which questions are important for their selected job role.
- They often lack immediate feedback on their answers.
- They find it difficult to identify their weak technical concepts.
- They may not have access to personalized interview guidance.

InterviewMate AI addresses these challenges by providing an AI-based interview practice environment with question generation, answer evaluation, and performance analysis.

## 3. Key Features

- **Personalized Interview Setup:** Enter candidate details, job role, experience, and technical skills.
- **Technical Interview Practice:** Practise questions related to Python, Django, SQL, DSA, and System Design.
- **HR Interview Practice:** Prepare for common HR and behavioural interview questions.
- **AI Question Generation:** Generate interview questions using IBM Granite.
- **RAG-Based Knowledge Retrieval:** Retrieve relevant information from the project's interview knowledge-base documents.
- **AI Answer Evaluation:** Evaluate candidate answers and provide a score out of 10.
- **Detailed Feedback:** Identify strengths, missing concepts, areas for improvement, and sample answers.
- **Performance Report:** View interview scores, answered questions, and overall performance.
- **Interview History:** Maintain interview-related records using SQLite, where implemented.

## 4. Interview Categories

The project supports the following interview preparation areas:

| Category | Topics Covered |
|---|---|
| Python | Fundamentals, data types, functions, OOP concepts, exceptions |
| Django | MVT architecture, models, views, URLs, ORM, middleware |
| SQL | Queries, joins, grouping, subqueries, database concepts |
| DSA | Arrays, strings, searching, sorting, linked lists, stacks, queues, trees |
| System Design | Requirements, architecture, scalability, databases, system components |
| HR | Introduction, strengths, weaknesses, teamwork, career goals, behavioural questions |

## 5. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | User interface |
| IBM Granite | AI question generation and answer evaluation |
| IBM watsonx.ai | Access to IBM Granite models |
| RAG | Retrieval of relevant interview knowledge |
| SQLite | Local interview data storage |
| Python Libraries | Document processing and application functionality |

## 6. System Architecture

The application follows a modular architecture in which the user interface, knowledge retrieval, AI services, and data storage work together.

```text
             Candidate
                 |
                 v
          Streamlit Interface
                 |
                 v
          Candidate Profile
                 |
                 v
           Interview Setup
                 |
                 v
            RAG Engine
                 |
                 v
         Knowledge Base Files
                 |
                 v
          Retrieved Context
                 |
                 v
          IBM Granite Model
            (watsonx.ai)
                 |
          +------+------+
          |             |
          v             v
    Question        Answer
    Generation      Evaluation
          |             |
          +------+------+
                 |
                 v
          Feedback & Scores
                 |
                 v
         Performance Report
                 |
                 v
             SQLite
```

## 7. Retrieval-Augmented Generation (RAG)

InterviewMate AI uses a document-based RAG approach to provide relevant interview knowledge to IBM Granite.

Instead of relying only on the model's general knowledge, the application retrieves relevant information from its local interview knowledge base.

### RAG Workflow

1. The candidate selects an interview category and provides their skills and experience.
2. The RAG engine searches the available knowledge-base documents.
3. Relevant interview questions and answers are retrieved.
4. The retrieved information is passed as context to IBM Granite.
5. Granite generates relevant questions or evaluates the candidate's answer using the available context.

### Knowledge Base

The project contains interview preparation material organized into separate documents:

```text
knowledge_base/
│
├── Python_Interview_QA.txt
├── Django_Interview_QA.txt
├── SQL_Interview_QA.txt
├── HR_Interview_QA.txt
├── DSA_Interview_QA.txt
└── System_Design_Interview_QA.txt
```

Each document is intended to contain interview questions, sample answers, topics, and relevant explanations.

## 8. Project Structure

```text
InterviewMate-AI/
│
├── app.py
│
├── granite_service.py
├── rag_engine.py
├── kb_loader.py
├── guardrails.py
├── interview_history.py
│
├── knowledge_base/
│   ├── Python_Interview_QA.txt
│   ├── Django_Interview_QA.txt
│   ├── SQL_Interview_QA.txt
│   ├── HR_Interview_QA.txt
│   ├── DSA_Interview_QA.txt
│   └── System_Design_Interview_QA.txt
│
├── test_guardrails.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

### Main Files

| File | Description |
|---|---|
| `app.py` | Main Streamlit application and interview workflow |
| `granite_service.py` | Handles communication with IBM Granite |
| `rag_engine.py` | Retrieves relevant knowledge-base content |
| `kb_loader.py` | Loads and processes knowledge-base documents |
| `guardrails.py` | Provides input/output validation and safety checks |
| `interview_history.py` | Handles interview history-related functionality |
| `test_guardrails.py` | Contains tests for guardrail functionality |
| `requirements.txt` | Lists the required Python packages |
| `.env.example` | Provides the environment variable template |

## 9. Installation and Setup

Follow these steps to run the project locally.

### Prerequisites

- Python 3.10 or later
- pip
- IBM Cloud account
- IBM watsonx.ai access
- IBM watsonx.ai API key and Project ID

### Step 1: Clone the Repository

```bash
git clone <your-github-repository-url>
```

Move into the project directory:

```bash
cd InterviewMate-AI
```

### Step 2: Create a Virtual Environment

```bash
py -m venv .venv
```

Activate it using PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create your `.env` file from the example:

```powershell
Copy-Item .env.example .env
```

Open `.env` and add your IBM watsonx.ai credentials using the variable names specified in `.env.example`.

Example:

```env
WATSONX_API_KEY=your_ibm_api_key
WATSONX_PROJECT_ID=your_watsonx_project_id
```


### Step 5: Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser.

## 10. How to Use the Application

1. Open InterviewMate AI.
2. Enter your name, job role, experience, and technical skills.
3. Select Technical Interview or HR Interview.
4. Choose the number of interview questions.
5. Start the interview.
6. Read each question and submit your answer.
7. Review the AI-generated feedback and score.
8. Complete the interview to view your performance report.

## 11. AI Evaluation

IBM Granite evaluates candidate answers and provides structured feedback.

The evaluation is designed to include:

- **Score:** A rating out of 10.
- **Strengths:** Correct points and concepts in the answer.
- **Missing Concepts:** Important information that the candidate did not include.
- **Suggestions:** Practical ways to improve the answer.
- **Sample Answer:** An example of a stronger answer.

The purpose of the evaluation is to help candidates understand their mistakes and improve their technical communication.

## 12. Responsible AI and Guardrails

The project includes guardrail functionality intended to make interview interactions safer and more reliable.

Key considerations include:

- Restricting the AI assistant to interview preparation tasks.
- Validating generated responses.
- Checking candidate inputs for suspicious prompt-injection patterns.
- Maintaining structured evaluation outputs.
- Preventing unsupported or fabricated feedback where possible.
- Keeping API credentials outside the source code.

Guardrails reduce risks but do not guarantee that every unsafe or incorrect response will be detected.

## 13. Future Enhancements

Possible future improvements include:

- Voice-based mock interviews.
- Speech-to-text answer submission.
- Resume-based personalized interview questions.
- More programming languages and job roles.
- Advanced question similarity detection.
- Candidate-wise interview analytics.
- Adaptive difficulty based on previous performance.
- More detailed performance dashboards.

## 14. Project Objective

The objective of InterviewMate AI is to provide an accessible AI-powered interview preparation platform that combines document-based knowledge retrieval with IBM Granite to help candidates practise interviews, evaluate their answers, identify weaknesses, and build confidence.

## 15. License

This project is developed for educational and academic purposes. A suitable open-source license can be added if the project is distributed publicly.
