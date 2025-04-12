import requests
import os
import json

# Get API key directly from environment (set by Docker Compose)
VENICE_API_KEY = os.getenv("VENICE_API_KEY")
VENICE_API_URL = "https://api.venice.ai/api/v1/chat/completions"

# Debug output to check if the environment variable is loaded
print(f"DEBUG: VENICE_API_KEY is {'set' if VENICE_API_KEY else 'not set'}")
print(f"DEBUG: VENICE_API_KEY exact value: '{VENICE_API_KEY}'")
if VENICE_API_KEY:
    print(f"DEBUG: VENICE_API_KEY value starts with: {VENICE_API_KEY[:4]}{'*' * (len(VENICE_API_KEY) - 4)}")
else:
    print(f"DEBUG: VENICE_API_KEY is None or empty")

def get_venice_response(prompt):
    if not VENICE_API_KEY or VENICE_API_KEY == "your-venice-api-key":
        # Return a more helpful mock response if no valid API key is provided
        print(f"No valid Venice API key found. Using mock response mode.")
        
        # Create specific mock responses for common queries
        mock_responses = {
            "explain rl in simple terms": "Reinforcement Learning (RL) is like teaching a dog tricks with treats. The dog (agent) performs actions, gets rewards (treats) for good behavior, and learns which actions lead to more rewards. Over time, it figures out the best strategy to get the most treats. In RL, a computer program learns by interacting with an environment, receiving feedback through rewards, and adjusting its behavior to maximize those rewards.",
            "what is rl": "Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative rewards. Unlike supervised learning, RL doesn't require labeled data but learns through trial and error.",
            "how does ppo work": "Proximal Policy Optimization (PPO) is a reinforcement learning algorithm that improves upon policy gradient methods. It works by collecting experiences, calculating advantages, and then updating the policy in small steps while staying close to the old policy (using a 'trust region'). This conservative update approach helps maintain stable learning.",
            "what is cartpole": "CartPole is a classic reinforcement learning environment where a pole is attached to a cart that moves along a track. The goal is to balance the pole by moving the cart left or right. It's a simple but effective environment for testing RL algorithms as it has a clear objective and relatively simple physics."
        }
        
        # Check if we have a pre-defined response for this prompt
        for key, response in mock_responses.items():
            if key.lower() in prompt.lower():
                return {"response": response, "status": "mock", "mock": True}
        
        # For other queries, generate a generic response
        return {
            "response": f"[MOCK RESPONSE] This is a simulated response to your query about: '{prompt}'. To get real responses, please configure a valid Venice AI API key in the .env file.",
            "status": "mock",
            "mock": True
        }
    
    # Format the request exactly as in the working example
    payload = {
        "venice_parameters": {
            "include_venice_system_prompt": True,
            "enable_web_search": "on"
        },
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "n": 1,
        "temperature": 0.15,
        "top_p": 0.9,
        "parallel_tool_calls": True,
        "model": "llama-3.3-70b",
        "repetition_penalty": 1,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    headers = {
        "Authorization": f"Bearer {VENICE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        # Print debugging info (remove in production)
        print(f"Sending request to {VENICE_API_URL}")
        print(f"Using authorization header: Bearer {VENICE_API_KEY[:4]}{'*' * (len(VENICE_API_KEY) - 4)}")
        
        # Make the request exactly as in the example
        response = requests.request("POST", VENICE_API_URL, json=payload, headers=headers)
        
        # Print response status for debugging
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text[:200]}...")  # Print first 200 chars of response
        
        if response.status_code == 200:
            response_json = response.json()
            # Check if the expected response format is present
            if "choices" in response_json and len(response_json["choices"]) > 0:
                message = response_json["choices"][0].get("message", {})
                content = message.get("content", "")
                return {"response": content}
            else:
                return {"response": str(response_json), "note": "Response format unexpected"}
        else:
            return {
                "response": f"API Error: Status {response.status_code}",
                "status": "error",
                "error_details": response.text
            }
            
    except Exception as e:
        # Handle any exceptions and provide descriptive error messages
        error_message = str(e)
        
        if "401" in error_message:
            return {
                "response": "Authentication failed. Please check your Venice AI API key.",
                "status": "error",
                "mock": True,
                "error_details": error_message
            }
        else:
            return {
                "response": f"API Error: {error_message}",
                "status": "error",
                "mock": True,
                "error_details": error_message
            }