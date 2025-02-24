from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq


class ChatBot:
    def __init__(self, api_key, model='llama3-8b-8192'):

        # Get Groq API key
        self.model = model
        # Initialize Groq Langchain chat object and conversation
        self.groq_chat = ChatGroq(
            groq_api_key=api_key,
            model_name=model
        )

        self.general_data = ''
        self.analysis_data = ''
        self.system_prompt = 'You are a friendly conversational chatbot made for CashGPT, a service for helping with ' \
                             'insurance claims processes. '
        self.conversational_memory_length = 15
        self.memory = ConversationBufferWindowMemory(k=self.conversational_memory_length, memory_key="chat_history",
                                                     return_messages=True)
        self.chat_history = []

    def respond(self, user_question, custom_instructions=""):
        if not user_question:
            return None

        # Construct a chat prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}"),
            ]
        )

        # Create a conversation chain
        conversation = LLMChain(
            llm=self.groq_chat,
            prompt=prompt,
            verbose=False,
            memory=self.memory,
        )

        # Generate response
        response = conversation.predict(human_input=user_question)
        return response

    def chat_loop(self):
        print("Hello! I am Cash GPT's chatbot.")
        while True:
            user_question = input("Ask a question: ")
            response = self.respond(user_question)
            print("Chatbot:", response)
