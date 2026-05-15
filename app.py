from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pypdf import PdfReader
import gradio as gr
import os
import requests

load_dotenv()


def push_notification(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        },
    )


@tool
def record_user_details(email, name="Name not provided", notes="not provided") -> dict:
    """Tool to record user details when they provide their email. This can be used to follow up with them later. The notes field can be used to capture any additional context or information about the user that might be relevant for follow-up."""
    push_notification(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}


@tool
def record_unknown_question(question) -> dict:
    """Tool to record any question that the model couldn't answer. This can be used to identify gaps in the model's knowledge or areas where it needs improvement."""
    push_notification(f"Recording {question}")
    return {"recorded": "ok"}


class Me:
    def __init__(self):
        self.name = "Santosh Tripathy"

        self.reader = PdfReader("Me/Santosh_Tripathy_Profile.pdf")
        self.linkedin = ""
        for page in self.reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text

        with open("Me/Profile_Summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

        self.model = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            tools=[record_user_details, record_unknown_question],
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

    def chat(self, message, history):
        # Convert history to LangChain message format
        lc_history = []
        for h in history:
            if h["role"] == "user":
                lc_history.append(HumanMessage(content=h["content"]))
            elif h["role"] == "assistant":
                lc_history.append(SystemMessage(content=h["content"]))
        messages = [SystemMessage(content=self.system_prompt())] + lc_history + [HumanMessage(content=message)]
        response = self.model.invoke(messages)
        return response.content

    def system_prompt(self):
        return (
            f"You are acting as {self.name}. You are answering questions on {self.name}'s website, "
            f"particularly questions related to {self.name}'s career, background, skills and experience. "
            f"Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. "
            f"You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. "
            f"Be professional and engaging, as if talking to a potential client or future employer who came across the website. "
            f"If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. "
            f"If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "
            f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
            f"With this context, please chat with the user, always staying in character as {self.name}."
        )


if __name__ == "__main__":
    me = Me()
    demo = gr.ChatInterface(me.chat)
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
