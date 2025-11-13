import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain.openai import ChatOpenAI
from langchain.tools import Tool
from newspaper import Article
from duckduckgo_search import DDGS
from urllib.parse import urlparse
import logging
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit app setup
st.set_page_config(page_title="AI Journalist Agent", page_icon="🤖", layout="wide")
st.title("🤖 AI Journalist Agent")
st.caption("Automatically research, analyze, summarise, and write high-quality articles using GPT-4o or GPT-3.5-turbo.")

# Initialize session states
if "current_step" not in st.session_state:
    st.session_state.current_step = "Not Started"
if "research_results" not in st.session_state:
    st.session_state.research_results = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    model_choice = st.selectbox(
        "Select AI Model",
        ["gpt-3.5-turbo (Fast)", "gpt-4o (High Quality)"],
        help="GPT-3.5-turbo is faster, GPT-4o produces higher quality articles."
    )
    article_length = st.slider("Article Length (words)", 300, 1000, 500, 100)
    num_sources = st.slider("Number of Sources", 2, 5, 3)
    show_intermediates = st.checkbox("Show Intermediate Results", value=False)
    st.subheader("Optional")
    article_style = st.selectbox(
        "Article Style",
        ["Informative", "Persuasive", "Narrative", "Analytical", "Conversational"]
    )
    target_audience = st.text_input("Target Audience (Optional)", placeholder="e.g., General public, Students, Professionals")

# DuckDuckGo Search Tool
def duckduckgo_search(query):
    try:
        with DDGS() as ddg:
            results = list(ddg.text(query, max_results=num_sources))
            urls = [result["href"] for result in results if result.get("href")]
            return urls if urls else ["No valid search results found."]
    except Exception as e:
        logger.error(f"Error in DuckDuckGo search: {str(e)}")
        return [f"Error performing search: {str(e)}"]

search_tool = Tool(
    name="DuckDuckGoSearch",
    func=duckduckgo_search,
    description="Perform DuckDuckGo searches and retrieve top URLs."
)

# Newspaper3k Fetch Tool
def fetch_article(url):
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return f"Invalid URL format: {url}"
        article = Article(url, timeout=10)
        article.download()
        article.parse()
        content = article.text
        return content[:2500] if content and len(content) > 50 else f"No valid content from {url}"
    except Exception as e:
        logger.error(f"Error fetching article from {url}: {str(e)}")
        return f"Error fetching article from {url}: {str(e)}"

fetch_tool = Tool(
    name="FetchArticle",
    func=fetch_article,
    description="Fetch article text from URLs with robust error handling."
)

# UI Elements
topic = st.text_input("Enter the topic you want an article on:")
progress_placeholder = st.empty()
message_placeholder = st.empty()
research_placeholder = st.empty()
analysis_placeholder = st.empty()
final_article_placeholder = st.empty()

def update_progress_message(step_name, message, progress_value):
    progress_bar = progress_placeholder.progress(progress_value)
    message_placeholder.info(f"**Current Step:** {step_name} – {message}")
    st.session_state.current_step = step_name
    time.sleep(0.5)
    return progress_bar

# Main Execution
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    try:
        chosen_model = "gpt-3.5-turbo" if "3.5" in model_choice else "gpt-4o"
        llm = ChatOpenAI(api_key=openai_api_key, model=chosen_model, temperature=0.7)

        journalist_agent = Agent(
            role="AI Journalist",
            goal=f"Create a {article_style.lower()} article for {target_audience or 'a general audience'}",
            tools=[search_tool, fetch_tool],
            llm=llm,
            backstory="A versatile journalist who creates quality articles adapted to different audiences and styles.",
            verbose=True
        )

        if topic and st.button("Generate Article"):
            with st.container():
                update_progress_message("Starting", "Initializing article generation process...", 0)

                # Research Phase
                research_task = Task(
                    description=f"Find {num_sources} relevant and authoritative URLs on '{topic}'. Format as numbered markdown links.",
                    agent=journalist_agent,
                    expected_output=f"{num_sources} markdown-formatted URLs"
                )
                update_progress_message("Research", "Searching for relevant sources...", 15)
                crew_research = Crew(agents=[journalist_agent], tasks=[research_task], verbose=True, process=Process.sequential)
                research_result = crew_research.kickoff()

                research_text = str(getattr(research_result, 'raw', research_result))
                st.session_state.research_results = research_text
                update_progress_message("Research", "Found relevant sources!", 30)

                if show_intermediates:
                    research_placeholder.success("Research Results:")
                    research_placeholder.markdown(research_text)

                # Analysis Phase
                analysis_task = Task(
                    description=f"Fetch and analyze content from each URL related to '{topic}'. Summarize each source in 2-3 sentences.",
                    agent=journalist_agent,
                    context=[research_task],
                    expected_output="Concise summaries of each source's key points"
                )
                update_progress_message("Analysis", "Analyzing content from sources...", 45)
                crew_analysis = Crew(agents=[journalist_agent], tasks=[analysis_task], verbose=True, process=Process.sequential)
                analysis_result = crew_analysis.kickoff()

                analysis_text = str(getattr(analysis_result, 'raw', analysis_result))
                st.session_state.analysis_results = analysis_text
                update_progress_message("Analysis", "Source analysis complete!", 60)

                if show_intermediates:
                    analysis_placeholder.success("Analysis Results:")
                    analysis_placeholder.markdown(analysis_text)

                # Writing Phase
                update_progress_message("Writing", "Crafting your article...", 75)
                writing_task = Task(
                    description=f"Write a {article_style.lower()} article about '{topic}' that is approximately {article_length} words. "
                                f"Target audience: {target_audience or 'general readers'}. "
                                "Format with markdown: include a compelling headline, introduction, main sections, and conclusion. "
                                "Base the article on the research and analysis results provided.",
                    agent=journalist_agent,
                    context=[research_task, analysis_task],
                    expected_output="A well-structured markdown article."
                )
                crew_writing = Crew(agents=[journalist_agent], tasks=[writing_task], verbose=True, process=Process.sequential)
                writing_result = crew_writing.kickoff()

                article_text = str(getattr(writing_result, 'raw', writing_result))
                update_progress_message("Complete", "Article generation completed successfully!", 100)

                # Display final article
                final_article_placeholder.subheader("📝 Final Article")
                final_article_placeholder.markdown(article_text)

                # Download button
                st.download_button("Download Article", article_text, f"{topic.replace(' ', '_')}_article.md", "text/markdown")

                # Feedback section
                st.subheader("🗣️ Article Feedback")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("👍 Great Article!"):
                        st.success("Thanks for your positive feedback!")
                with col2:
                    if st.button("🙂 Good but could be better"):
                        st.info("Thanks for your feedback! What could be improved?")
                        st.text_area("Suggestions for improvement:")
                with col3:
                    if st.button("👎 Needs Improvements"):
                        st.warning("We appreciate your honest feedback!")
                        st.text_area("What could be improved?")
    except Exception as e:
        st.error(f"Error: {str(e)}")
        logger.error(f"An error occurred: {str(e)}")
else:
    st.warning("Please enter your OpenAI API Key in the sidebar to continue.")

st.markdown("---")
st.caption("AI Journalist — powered by CrewAI and LangChain")
