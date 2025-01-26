from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import datetime


# Google Calendar API helper functions
def authenticate_google_calendar():
    """Authenticate and return a service object for Google Calendar API."""
    creds = Credentials.from_authorized_user_file(
        "token.json", ["https://www.googleapis.com/auth/calendar"]
    )
    return build("calendar", "v3", credentials=creds)


def book_appointment(service, summary, start_time, end_time):
    """Book an appointment in Google Calendar."""
    event = {
        "summary": summary,
        "start": {"dateTime": start_time, "timeZone": "Asia/Dhaka"},
        "end": {"dateTime": end_time, "timeZone": "Asia/Dhaka"},
    }
    event_result = service.events().insert(calendarId="primary", body=event).execute()
    return f"Event created: {event_result.get('htmlLink')}"


# Define the tool
def book_calendar_tool(input_text):
    """Parse input and book an appointment."""
    try:
        service = authenticate_google_calendar()

        # Example: Parse input text (you can use NLP models to improve this)
        summary = "Meeting with John"  # Replace with parsed input
        start_time = "2025-01-27T10:00:00"  # Replace with parsed input
        end_time = "2025-01-27T11:00:00"  # Replace with parsed input

        # Book the event
        result = book_appointment(service, summary, start_time, end_time)
        return result
    except Exception as e:
        return str(e)


tools = [
    Tool(
        name="Google Calendar",
        func=book_calendar_tool,
        description="Use this tool to book appointments in Google Calendar. Provide details like meeting name, start time, and end time.",
    )
]

# Initialize LangChain agent
llm = ChatOpenAI(temperature=0.2, model="gpt-4")
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# Run the agent
response = agent.run(
    "Book a meeting with John on January 27th, 2025, from 10:00 AM to 11:00 AM."
)
print(response)
