# IQ_LLM
Intelligent Question LLM Generation


# Example usage for SpecificRewardFunction class
```bash
prompt_criteria = """
Hello, {user_name}! Based on your current state of {state}, and considering {consideration}, how can I assist you today?

Please ensure the output is sanitized for JSON by removing any URLs, fixing typos,
and adjusting spaces and punctuation as necessary.

Here is the additional info you requested:
{additional_info}
"""

reward_function = SpecificRewardFunction(llm_model, output_logits, score_dict, prompt_criteria)

# Example parameters
params = {
    "user_name": "John Doe",
    "state": "California",
    "consideration": "weather conditions",
    "additional_info": "It will be sunny tomorrow."
}

prompt = reward_function.generate_prompt(**params)
print(prompt)
```