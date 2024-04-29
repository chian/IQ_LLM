from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from langchain_core.prompts import PromptTemplate


class SpecificRewardFunction(RewardFunction):
    def __init__(self, llm, prompt_criteria, output_logits, score_dict):
        """
        Initialize common properties for all reward functions.

        :param llm: Language model or similar large model used.
        :param prompt_criteria: Score judging criteria.
        :param output_logits: Logits from the model output to evaluate.
        :param score_dict: Dictionary from logits --> scores.
        """
        super().__init__()
        self.llm = llm
        self.prompt_template = PromptTemplate.from_template(prompt_criteria)
        self.output_logits = output_logits
        self.score_dict = score_dict

    def generate_prompt(self, **params):
        """
        Generate a formatted prompt using the provided parameters.

        :param params: A dictionary of parameters where keys match the placeholders in the template.
        :return: A formatted prompt string.
        """
        return self.prompt_template.invoke(**params)

    def compute_reward(self, prompt_dict):
        """
        Compute the reward for a given state transition.

        :param state: The current state.
        :return: The computed reward.
        """
        prompt = self.generate_prompt(**prompt_dict)
        model_output = self.llm(prompt)
        scores = {}
        for logit in self.output_logits:
            probability = model_output.get(logit, 0)  # MAGIC: NEED TO FIX: Get the probability of the logit from the model output
            score_value = self.score_dict.get(logit, 0)  # Get the score value from the score dictionary
            weighted_score = probability * score_value  # Calculate the weighted score
            scores[logit] = weighted_score  # Assign the weighted score to the logit in the scores dictionary
        #Or the python unreadable version:
        #scores = {logit: model_output.get(logit, 0) * self.score_dict.get(logit, 0) for logit in self.output_logits}
        return scores
    
import unittest
from rewards import SpecificRewardFunction  # Assuming rewards.py contains the SpecificRewardFunction class
from ARGO import ArgoWrapper
from CustomLLM import ARGO_LLM

class TestSpecificRewardFunction(unittest.TestCase):
    def setUp(self):
        # Define the prompt criteria and output logits
        prompt_criteria = "What is the capital of France? Options: A) Paris, B) London, C) Berlin, D) Madrid"
        self.output_logits = ['A', 'B', 'C', 'D']
        
        # Define the score dictionary
        self.score_dict = {'A': 1, 'B': -1, 'C': -1, 'D': -1}
        
        # Initialize the SpecificRewardFunction with the actual LLM from ARGO
        self.llm = self.initialize_argo_llm() # Placeholder for actual LLM initialization
        self.reward_function = SpecificRewardFunction(
            llm=self.llm,
            prompt_criteria=prompt_criteria,
            output_logits=self.output_logits,
            score_dict=self.score_dict
        )
        
        # Setup the prompt template
        self.reward_function.prompt_template = MagicMock()
        self.reward_function.prompt_template.invoke = MagicMock(return_value="What is the capital of France? Options: A) Paris, B) London, C) Berlin, D) Madrid")

    def initialize_argo_llm(self):
        # Initialize and return the ARGO LLM
        argo_wrapper_instance = ArgoWrapper()
        llm = ARGO_LLM(argo=argo_wrapper_instance, model_type='gpt4', temperature=1.0)
        return llm

    def test_compute_reward(self):
        # Generate prompt dictionary for testing
        prompt_dict = {}
        
        # Call compute_reward
        scores = self.reward_function.compute_reward(prompt_dict)
        
        # Print the scores and the logits
        print("Computed Rewards:")
        for logit, score in scores.items():
            print(f"Logit: {logit}, Score: {score}")

# Run the tests
if __name__ == '__main__':
    unittest.main()
