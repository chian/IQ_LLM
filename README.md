# IQ_LLM
Intelligent Question LLM Generation


# Example usage for SpecificRewardFunction class
```bash

# Assuming llm_model is an instance of a language model class
llm_model = SomeLanguageModel()

# Define the prompt template with placeholders for dynamic content
prompt_criteria = """
Hello, {user_name}! Based on your current state of {state}, and considering {consideration}, how can I assist you today?
Please ensure the output is sanitized for JSON by removing any URLs, fixing typos, and adjusting spaces and punctuation as necessary.
Here is the additional info you requested:
{additional_info}
"""

# Define possible outputs from the language model
output_logits = ['Option1', 'Option2', 'Option3']

# Define scores for each possible output
score_dict = {
    'Option1': 10,
    'Option2': -5,
    'Option3': 0
}

# Initialize the SpecificRewardFunction
reward_function = SpecificRewardFunction(llm_model, prompt_criteria, output_logits, score_dict)

# Example parameters
params = {
    "user_name": "John Doe",
    "state": "California",
    "consideration": "weather conditions",
    "additional_info": "It will be sunny tomorrow."
}

prompt = reward_function.generate_prompt(**params)
print(prompt)

# Simulate a model response (in practice, this would come from the llm_model)
model_response = {
    'Option1': 0.2,  # Probability of Option1 being the response
    'Option2': 0.5,  # Probability of Option2 being the response
    'Option3': 0.3   # Probability of Option3 being the response
}

# Compute the rewards based on the model response
scores = reward_function.compute_reward(model_response)
print("Computed Rewards:", scores)
```