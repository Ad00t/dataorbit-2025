from chatbot import ChatBot


def main():
    """
    This function is the main entry point of the application. It sets up the Groq client, the Streamlit interface, and handles the chat interaction.
    """

    groq_api_key = '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'
    cash_gpt = ChatBot(groq_api_key)
    data = ""

    # Summary of analytics
    # print(cash_gpt.respond(custom_instructions=f"Summarize these analytics: {data}"))

    cash_gpt.chat_loop()


if __name__ == "__main__":
    main()