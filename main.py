import os
import json
import math
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # API key

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------------------------------------------------
# 1. ENVIRONMENT & TOOLS (The "Signals" and "Proteins")
# ---------------------------------------------------------
AVAILABLE_TOOLS = {
    "calculator": {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluates mathematical expressions. Input MUST be a valid Python math expression string (e.g. '2 + 2 * 3').",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]
            }
        }
    },
    "reverse_string": {
        "type": "function",
        "function": {
            "name": "reverse_string",
            "description": "Reverses a given string.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"]
            }
        }
    }
}


def execute_tool(name: str, args: Dict[str, Any]) -> str:
    if name == "calculator":
        try:
            # Tworzymy bezpieczne środowisko, ale wstrzykujemy funkcje matematyczne (np. sqrt)
            safe_env = {"__builtins__": None}
            for k, v in math.__dict__.items():
                if not k.startswith("__"):
                    safe_env[k] = v
            return str(eval(args["expression"], safe_env, {}))
        except Exception as e:
            return f"Error: {e}"
    elif name == "reverse_string":
        return args["text"][::-1]
    return "Tool not found."


# ---------------------------------------------------------
# 2. THE SPECIALIZED AGENT ARCHITECTURE (The "Differentiated Cell")
# ---------------------------------------------------------
class AgentDNA(BaseModel):
    system_prompt: str = Field(description="The heavily optimized system prompt for the specialized agent.")
    selected_tools: List[str] = Field(description="List of tool names selected from the available library.")


class SpecializedAgent:
    def __init__(self, dna: AgentDNA):
        self.dna = dna
        self.tools = [AVAILABLE_TOOLS[t] for t in dna.selected_tools if t in AVAILABLE_TOOLS]

    def run(self, user_input: str) -> str:
        messages = [{"role": "system", "content": self.dna.system_prompt}, {"role": "user", "content": user_input}]

        # ZMIANA: Pozwalamy na maksymalnie 5 kroków (iteracji) narzędziowych w jednym zadaniu.
        # Umożliwia to rozwiązywanie problemów matematycznych krok po kroku.
        MAX_ITERATIONS = 5

        for _ in range(MAX_ITERATIONS):
            params = {"model": "gpt-4o", "messages": messages, "temperature": 0.0}
            if self.tools:
                params["tools"] = self.tools
                params["tool_choice"] = "auto"

            response = client.chat.completions.create(**params)
            message = response.choices[0].message

            if message.tool_calls:
                # 1. Dodajemy wiadomość asystenta do historii
                messages.append(message)

                # 2. Iterujemy przez każde wywołanie narzędzia wygenerowane przez model
                for tool_call in message.tool_calls:
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        args = {}  # Zabezpieczenie przed halucynacją JSON-a przez model

                    tool_result = execute_tool(tool_call.function.name, args)

                    # 3. Zwracamy wynik przypisany do konkretnego ID wywołania
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(tool_result)
                    })
                # UWAGA: Pętla kontynuuje się tutaj, aby model mógł przeanalizować tool_result
                # i wywołać kolejne narzędzia (jeśli zajdzie taka potrzeba).
            else:
                # Model nie używa już narzędzi, podaje ostateczną odpowiedź w tekście.
                return message.content

        return "Error: Maximum tool iterations reached. Agent could not complete the task."


# ---------------------------------------------------------
# 3. EVALUATION FRAMEWORK (The "Safeguards" & "Fitness Check")
# ---------------------------------------------------------
MATH_DATASET = [
    {
        "input": "Calculate the exact result of 17384.51 multiplied by 49.23, then subtract 34.876.",
        "expected": "855804.5513"
    },
    {
        "input": "What is 345 raised to the power of 3, then divided by 12.5?",
        "expected": "3285090.0"
    },
    {
        "input": "Find the exact value of (845.28 + 12.81) * (99.14 - 4.3).",
        "expected": "81381.2556"
    },
    {"input": "What is the square root of 8464 multiplied by 13.5?", "expected": "1242"},
    {"input": "Multiply 452 by 17, then subtract the result of 8912 divided by 4.", "expected": "5456"},
    {
        "input": "If an investment of 5000 grows by exactly 7 percent each year, what is the exact amount after 3 years? Do not round.",
        "expected": "6125.215"}
]


def evaluate_agent(agent: SpecializedAgent, dataset: List[Dict]) -> tuple[float, List[str]]:
    correct = 0
    errors = []
    for item in dataset:
        output = agent.run(item["input"])

        # Uodpornienie na "Comma Bug" (np. 6,125.215 -> 6125.215)
        clean_output = output.replace(",", "")

        if item["expected"] in clean_output:
            correct += 1
        else:
            errors.append(f"Input: {item['input']} | Expected: {item['expected']} | Got: {output}")
    accuracy = correct / len(dataset)
    return accuracy, errors


# ---------------------------------------------------------
# 4. THE STEM AGENT (The "Meta-Optimizer")
# ---------------------------------------------------------
def stem_cell_differentiation(task_description: str, target_accuracy: float = 1.0,
                              max_generations: int = 5) -> SpecializedAgent:
    history = []

    for generation in range(max_generations):
        print(f"\n--- Generation {generation + 1} ---")

        # The Stem Agent decides what to become
        stem_prompt = f"""
        You are a Stem Agent. Your goal is to design a highly specialized AI agent for the following task:
        TASK: {task_description}

        Available tools in the environment: {list(AVAILABLE_TOOLS.keys())}.

        Past generations history (Iterate and improve based on these failures):
        {json.dumps(history, indent=2)}

        Output the DNA (System Prompt and selected tools) for the new specialized agent. 
        If previous generations failed, change your strategy (e.g., add Chain of Thought, fix tool selection).
        """

        response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[{"role": "user", "content": stem_prompt}],
            response_format=AgentDNA,
            temperature=0.7
        )

        dna = response.choices[0].message.parsed
        print(f"Generated DNA: \nTools: {dna.selected_tools}\nPrompt: {dna.system_prompt[:100]}...")

        # Grow the cell and test it
        candidate_agent = SpecializedAgent(dna)
        accuracy, errors = evaluate_agent(candidate_agent, MATH_DATASET)

        print(f"Accuracy: {accuracy * 100}%")

        if accuracy >= target_accuracy:
            print("Differentiation successful. Stem cell has matured.")
            return candidate_agent
        else:
            print("Safeguards triggered. Reverting and learning from errors.")
            history.append({
                "dna_attempted": {"prompt": dna.system_prompt, "tools": dna.selected_tools},
                "accuracy": accuracy,
                "errors": errors
            })

    print("Max generations reached. Returning best effort.")
    return candidate_agent


# ---------------------------------------------------------
# 5. MAIN EXECUTION (Before / After Comparison)
# ---------------------------------------------------------
if __name__ == "__main__":
    task_desc = "Solve complex math word problems. You must be perfectly accurate."

    print("=== BEFORE: BASELINE ZERO-SHOT MODEL ===")
    baseline_dna = AgentDNA(system_prompt="You are a helpful assistant.", selected_tools=[])
    baseline_agent = SpecializedAgent(baseline_dna)
    base_acc, base_err = evaluate_agent(baseline_agent, MATH_DATASET)
    print(f"Baseline Accuracy: {base_acc * 100}%")
    if base_err:
        print(f"Baseline Errors:\n" + "\n".join(base_err))

    print("\n=== STARTING STEM CELL DIFFERENTIATION ===")
    specialized_agent = stem_cell_differentiation(task_description=task_desc)

    print("\n=== AFTER: SPECIALIZED AGENT ===")
    final_acc, final_err = evaluate_agent(specialized_agent, MATH_DATASET)
    print(f"Final Specialized Accuracy: {final_acc * 100}%")
    if final_err:
        print(f"Final Errors:\n" + "\n".join(final_err))