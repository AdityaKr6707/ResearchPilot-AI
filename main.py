
from typing import TypedDict, Literal

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class ResearchState(TypedDict):
    question: str
    history: list
    answer: str
    route: str


llm = ChatOllama(model="llama3.2:3b", temperature=0.2)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are ResearchPilot AI. Explain clearly and never invent citations."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])


def route_question(state: ResearchState) -> dict:
    question = state["question"].lower()

    research_words = ["research", "paper", "study", "evidence", "sources"]

    if any(word in question for word in research_words):
        return {"route": "research"}

    return {"route": "general"}


def generate_answer(state: ResearchState) -> dict:
    response = (prompt | llm).invoke({
        "question": state["question"],
        "history": state["history"]
    })
    return {"answer": response.content}


def choose_path(state: ResearchState) -> Literal["research", "general"]:
    return state["route"]


builder = StateGraph(ResearchState)

builder.add_node("router", route_question)
builder.add_node("research", generate_answer)
builder.add_node("general", generate_answer)

builder.add_edge(START, "router")

builder.add_conditional_edges(
    "router",
    choose_path,
    {
        "research": "research",
        "general": "general"
    }
)

builder.add_edge("research", END)
builder.add_edge("general", END)

research_graph = builder.compile()


def main():
    history = []

    print("\n=== ResearchPilot AI ===")
    print("LangChain + LangGraph | Type 'exit' to quit\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        try:
            result = research_graph.invoke({
                "question": question,
                "history": history,
                "answer": "",
                "route": ""
            })

            answer = result["answer"]
            print("\nResearchPilot:\n", answer, "\n")

            history.extend([
                HumanMessage(content=question),
                AIMessage(content=answer)
            ])

        except Exception as error:
            print("Error:", error)


if __name__ == "__main__":
    main()

print(research_graph.get_graph().draw_ascii())

