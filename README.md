# **Project Feasibility Analyzer (Version 1.0)**

An AI-powered solution to analyze PDF documents and assess the feasibility of projects without needing to manually go through lengthy files. This tool is designed to provide instant insights, reduce manual effort, and help businesses make quick decisions about potential projects.

**Note**: This is the **version 1.0** of the project, which is still a work in progress. The core functionality is implemented, but future versions will feature an improved front-end for a more polished user experience.

## **Overview**

This project was developed during the **KI Hackathon 2024 in Bremen**, where the challenge was to create an efficient tool that helps companies quickly assess whether a project is feasible from their perspective.

Given the complexity of lengthy client-side documents or legal paperwork, this tool analyzes the content, extracts key information, and provides a quick feasibility summary—saving time, improving accuracy, and ensuring better decision-making.

## **Features**

- **PDF Upload and Analysis**:  
  Upload PDF documents for instant analysis. The tool processes and extracts key metrics such as **budget**, **timeline**, **resource requirements**, and **risks**.

- **Automated Feasibility Score**:  
  Get an automatic **feasibility score** based on key factors such as budget, timeline, and risks. A simple visual indicator (green = feasible, yellow = review needed, red = not feasible) shows whether the project is likely to succeed.

- **Dynamic Custom Queries**:  
  Ask specific questions about the document (e.g., "What is the project timeline?", "What are the risks involved?") using an LLM-powered chatbot. The AI will respond with context-specific answers from the document.

- **Key Metrics Extraction**:  
  Automatically extract **important project highlights** like budget, stakeholders, technical requirements, and potential risks.  
  Results can be exported in **CSV** or **Excel** format for further analysis or reporting.

- **Risk and Compliance Analysis**:  
  Identifies and flags **risks**, **legal considerations**, and **compliance issues** based on the document content.  
  The risk level is displayed as a **progress bar** to help assess the project's challenges.

- **Interactive Doughnut Chart for Feasibility**:  
  Visual representation of the project’s feasibility based on extracted factors. Key metrics such as budget, timeline, and resources are displayed in an **interactive doughnut chart**.

- **Operational Cost Calculation**:  
  Provides real-time **cost calculations** for using various AI models (e.g., GPT-3.5, GPT-4). This ensures cost transparency and helps users manage their budgets when using advanced models.

- **Downloadable Reports**:  
  Export the analysis and key findings as CSV or Excel files for internal review, presentations, or further action.

## **Tech Stack**

- **Streamlit**: For building the web-based UI.
- **LangChain**: For connecting to language models and handling text-based queries.
- **OpenAI GPT**: For answering dynamic custom queries and providing intelligent responses.
- **PyPDF2**: For extracting text from PDF documents.
- **Plotly**: For creating interactive visualizations like the doughnut chart.
- **Pandas & XlsxWriter**: For handling data extraction and providing downloadable CSV/Excel reports.
- **Tesseract OCR**: For text extraction from images within PDFs.
- **Tabula**: For extracting tabular data from PDFs.

## **Problem Statement**

**Encoway** and the organizers of the **KI Hackathon 2024 in Bremen** posed the challenge of creating a solution that allows companies to quickly assess the feasibility of potential projects without needing to manually read through long documents. The goal was to automate the process of extracting crucial project information and presenting it in a way that is both efficient and insightful.

The solution helps decision-makers at companies save time, offering a clear feasibility analysis, and providing actionable recommendations for project success.

## **How to Run the Project Locally**

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/project-feasibility-analyzer.git
    cd project-feasibility-analyzer
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up your OpenAI API key by creating a `.env` file in the project root and adding:
    ```
    OPENAI_API_KEY=your-openai-api-key-here
    ```

5. Run the application:
    ```bash
    streamlit run app.py
    ```

6. Navigate to `http://localhost:8501` in your browser to view the app.

## **Future Improvements**

- **Enhanced Front-End**: A more polished front-end design is planned for future versions, improving the user interface and experience.
- **Feasibility Simulation**: Add sliders for adjusting project parameters (e.g., budget, timeline) to see how changes affect the overall feasibility score.
- **Improved Risk Analysis**: Incorporate machine learning models to provide deeper insights into potential project risks.
- **User Feedback Integration**: Allow users to rate the feasibility analysis and provide feedback for improving future recommendations.

## **Contributing**

Contributions are welcome! If you have any ideas or improvements, feel free to fork the repo, make your changes, and submit a pull request.

## **License**

This project is licensed under the MIT License. See the `LICENSE` file for details.

