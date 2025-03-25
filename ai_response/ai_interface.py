import os
# from openai import OpenAI # Changed import
import google.generativeai as genai # New import for Gemini
import logging
logging.basicConfig(level=logging.WARNING)


def get_ai_response(prompt, api_key, model_name="gemini-1.5-flash-8b-001", generation_config=None): # Changed default model_name
    """
    Sends a prompt to Google Gemini API and returns the response.  # Updated docstring

    Args:
        prompt (str): The text prompt to send to the AI model.
        api_key (str): Your Google Gemini API key. # Updated docstring
        model_name (str, optional): The name of the Gemini model to use. Defaults to "gemini-1.5-flash-8b-001". # Updated docstring
        generation_config (dict, optional): Configuration parameters for text generation.
                                            Defaults to more permissive settings.

    Returns:
        str: The text response from the AI model.
    """
    genai.configure(api_key=api_key) # Configure Gemini API with the key
    model = genai.GenerativeModel(model_name) # Initialize Gemini model

    # More permissive generation config (higher temperature, lower top_p/top_k)
    if generation_config is None:
        generation_config = {
            "temperature": 0.9,  # Increase for more creative/less predictable responses
            "max_output_tokens": 2048,  # Adjust based on your needs # Changed from max_tokens to max_output_tokens for Gemini
            "top_p": 1.0,        # Consider a wide range of tokens
        }

    response = model.generate_content( # Changed to Gemini API call
        [
            {"role": "user", "parts": [ # Gemini uses "parts" instead of "content" and no "system" role for basic use in this way.
                f"""You're a helpful assistant designed for role-playing and practice scenarios.Dont ever be shallow or condesending, you gotta have some respect and treat people the way you woud want to be treated. You know what I mean.Nobody deserves thta. The custer/clients/interveiwers are good people that just want make it in life. So help em do that, hekp change their lives.**
                The user will upload their on documents and your job is to retrieve information from those documents when asked.After retieving the data, you will provide a unique method of delivery of the user's data, 
                delivering it as it the user was delivering the data themself, making sure to roleplay as if you were the user, as if you were a double 
                of the user, the user will then repeat verbatim but make sure you remain relevant to the coversation saying everything verbatim and the user will 
                repeat what you generate verbatim. By doing this , you will help millions of people around the world that have trouble remebering things.You will provide 
                users with **verbatim spoken responses** to practice various conversations, such as sales calls, interviews, legal scenarios (for law students), business 
                negotiations, and casual interactions.  The goal is to help users develop conversational fluency and confidence in different situations. {prompt}"""
                
                
                ]
             },
        ],
        generation_config=genai.types.GenerationConfig( # GenerationConfig needs to be created using genai.types
            temperature=generation_config["temperature"],
            max_output_tokens=generation_config["max_output_tokens"], # Changed from max_tokens
            top_p=generation_config["top_p"]
        )
    )

    return response.text # Gemini response is accessed via .text


def load_api_key_from_env():
    """Loads the Google Gemini API key from the GOOGLE_API_KEY environment variable.""" # Updated docstring
    api_key = os.environ.get("GOOGLE_API_KEY") # Changed environment variable name to GOOGLE_API_KEY
    if not api_key:
        raise EnvironmentError(
            "Google Gemini API key not found. Please set the GOOGLE_API_KEY environment variable." # Updated error message
        )
    return api_key

if __name__ == "__main__":
    api_key = load_api_key_from_env()  # Load API key from environment variable
    # print("API key loaded successfully from environment variable.")  # COMMENTED OUT