import streamlit as st
import random
import time
import requests
import dotenv
import os
import json
import re  # Import regex module for parsing <think> tags
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

dotenv.load_dotenv()


# Initialize Streamlit session state
def initialize_session_state():
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = ""
    if "conversation_id_buttons" not in st.session_state:
        st.session_state.conversation_id_buttons = ""
    if "latest_message" not in st.session_state:
        st.session_state.latest_message = ""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "customer_name" not in st.session_state:
        st.session_state.customer_name = ""
    if "customer_email" not in st.session_state:
        st.session_state.customer_email = ""
    if "firstpage" not in st.session_state:
        st.session_state.firstpage = True
    if "show_thank_you" not in st.session_state:
        st.session_state.show_thank_you = False

# Initialize LangChain's Groq LLM
chat_model = ChatGroq(
            temperature=0, 
            groq_api_key=os.getenv("GROQ_API_KEY"), 
            model_name="qwen-qwq-32b"
        )

# Function to get chatbot response using LangChain
def get_chatbot_response(user_input):
    try:
        # Use LangChain to generate a response
        response = chat_model([HumanMessage(content=user_input)])
        return response.content
    except Exception as e:
        return f"Error: {e}"

# Function to get chatbot response using LangChain (streaming)
def get_chatbot_response_stream(user_input):
    try:
        # Simulate streaming response from the LLM
        response_generator = chat_model.stream([HumanMessage(content=user_input)])
        for chunk in response_generator:
            yield chunk.content  # Yield each chunk of the response
    except Exception as e:
        yield f"Error: {e}"  # Yield the error message

# Function to fetch dynamic buttons using LangChain
def get_dynamic_buttons(latest_message):
    try:
        # Use LangChain to generate a response for dynamic buttons
        response = {'single': {'options': ['Option 1', 'Option 2']}}
        # Assuming the response is a JSON string with button details
        return json.dumps(response)
    except Exception as e:
        return {}


# Function to display chat messages
def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar", None)):
            st.markdown(message["content"])

# Function to handle dynamic buttons
# @st.fragment()
def handle_dynamic_buttons():
    # Check if the latest message has changed
    if "dynamic_buttons" not in st.session_state or st.session_state.latest_message != st.session_state.messages[-1]["content"]:
        st.session_state.latest_message = st.session_state.messages[-1]["content"]
        # Fetch dynamic buttons only when the latest message changes
        dynamic_buttons = get_dynamic_buttons(st.session_state.latest_message)
        dynamic_buttons = json.loads(dynamic_buttons)
        st.session_state.dynamic_buttons = dynamic_buttons
    else:
        # Use cached dynamic buttons
        dynamic_buttons = st.session_state.dynamic_buttons

    # Handle single or multi-select buttons
    if dynamic_buttons.get("type") == "single":
        options = dynamic_buttons.get("options", [])
        num_buttons = len(options)
        cols = st.columns(max(10,num_buttons))
        start_index = (10 - num_buttons) // 2
        for i, button_label in enumerate(options):
            with cols[start_index + i]:
                if st.button(button_label, key=f"dynamic_button_{i}"):
                    return button_label
    elif dynamic_buttons.get("type") == "multi":
        options = dynamic_buttons.get("options", [])
        selected_options = []
        num_buttons = len(options)
        cols = st.columns(max(10,num_buttons))
        start_index = (10 - num_buttons) // 2
        # Dynamically render checkboxes for each option
        for i, option in enumerate(options):
            with cols[start_index + i]:
                if st.checkbox(option, key=f"checkbox_{i}"):
                    selected_options.append(option)

        # Add a submit button
        cols = st.columns(10)
        with cols[4]:
            if st.button("Submit Selection"):
                if selected_options:
                    # Return the selected options as a comma-separated string
                    return ", ".join(selected_options)
                else:
                    st.warning("Please select at least one option before submitting.")
    return None

# Function to handle user input
def handle_user_input():
    selected_button = handle_dynamic_buttons()
    if selected_button:
        return selected_button
    return st.chat_input("Type your responses here...")

# Function to handle thank-you message
def handle_thank_you():
    st.markdown(
        """
        <div style="text-align: center; margin-top: 50px;">
            <h2>Thank you for chatting with me.</h2>
            <p>We will send you a confirmation email shortly.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()



# Main function
def main():
    st.set_page_config(layout="wide", page_title="Chatbot UI", page_icon=":robot:")
    initialize_session_state()
    if st.session_state.show_thank_you:
        handle_thank_you()

    st.title("Chatbot UI")

    if not st.session_state.customer_email:
        st.subheader("Welcome! Please provide your details to start the conversation.")
        with st.form("user_details_form"):
            st.session_state.customer_name = st.text_input("Customer Name", placeholder="Enter your name")
            st.session_state.customer_email = st.text_input("Customer Email", placeholder="Enter your email")
            submit_button = st.form_submit_button("Submit")

        if submit_button:
            if st.session_state.customer_email:
                st.success("Thank you! You can now start the conversation.")
                time.sleep(1)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Hello! {st.session_state.customer_name}, I am your AI Assistant."
                })
                st.session_state.firstpage = False
                st.rerun()

    display_chat_messages()

    if not st.session_state.firstpage:
        prompt = handle_user_input()
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Thinking block
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                st.session_state.thinking = True  # Set thinking state
                with st.spinner("Thinking..."):
                    for chunk in get_chatbot_response_stream(prompt):
                        full_response += chunk

                # Parse the response to extract <think> content
                think_match = re.search(r"<think>(.*?)</think>", full_response, re.DOTALL)
                if think_match:
                    think_content = think_match.group(1).strip()  # Extract content inside <think> tags
                    actual_response = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()  # Remove <think> tags
                else:
                    think_content = None
                    actual_response = full_response.strip()

                # Store the full response and <think> content in session state
                st.session_state.latest_response = actual_response
                st.session_state.think_content = think_content

                # Display the "thinking" block with buttons
                if st.session_state.thinking:
                    if st.button("Reveal Thinking"):
                        if st.session_state.think_content:
                            st.markdown(f"**Thinking Process:** {st.session_state.think_content}")  # Show the <think> content
                        else:
                            st.warning("No thinking process available.")
            
                    st.session_state.thinking = False  # Update state to stop thinking
                    response_placeholder.markdown(actual_response)  # Show the actual response
                    st.session_state.messages.append({"role": "assistant", "content": actual_response})
                    st.rerun()
                else:
                    response_placeholder.markdown("_The assistant is thinking..._")  # Placeholder text

if __name__ == "__main__":
    main()


