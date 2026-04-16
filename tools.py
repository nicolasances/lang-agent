from typing import List
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Input should be a valid Python math expression (e.g. '2 + 2', '10 * 5 / 2')."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

@tool
def get_common_items() -> List[str]:

    """Returns the list of items that are the most commonly used in the shopping list (supermarket list)"""
    return ["Bacon", "Bacon i tern", "Spaghetti", "Greek Yogurt", "Riskiks", "Majskiks", "Medister", "Pasta", "Bread", "Bread C", "Bread N"]

@tool
def add_item_to_list(item: str): 
    """Adds a given item to the supermarket list (shopping list)"""
    print(f"Item {item} added to your shopping list")