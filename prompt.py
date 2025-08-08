import openai
import os
import random
import re
import time

# Load OpenAI API key
with open("openai-api-key.txt", "r") as f:
    api_key = f.read().strip()

client = openai.OpenAI(api_key=api_key)

# Choose object types and topics
object_types = ["definition", "theorem", "lemma", "example"]
topics = [
    "group", "ring", "field", "topological space", "metric space", "smooth manifold",
    "sigma-algebra", "supremum", "basis of a vector space", "injectivity of a function",
    "prime number", "the pigeonhole principle", "the irrationality of sqrt 2",
    "the surjectivity of exp : ℝ → ℝ⁺", "bijection between ℕ and ℚ",
    "dense set", "power set", "eigenvector", "quotient group"
]

def generate_description(object_type: str, topic: str) -> str:
    if object_type == "definition":
        return f"a formal definition of a {topic}"
    elif object_type == "example":
        return f"an example of a {topic}"
    elif object_type == "theorem":
        return f"a formal proof of the {topic}"
    elif object_type == "lemma":
        return f"a lemma about the {topic}"
    else:
        return f"{object_type} of {topic}"

def clean_output(text: str) -> str:
    # Remove ``` blocks (even with "lean")
    text = re.sub(r"^```(?:lean)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()

# Clean file name
def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")

# GPT system message (strong constraint)
system_prompt = {
    "role": "system",
    "content": (
        "You are a Lean 4 code generator. "
        "Your only task is to output valid Lean 4 code. "
        "Do not include explanations, comments, or markdown formatting. "
        "Do not use triple backticks. Only output Lean code. "
        "Include import statements if needed. "
        "The result must be copy-pastable into a .lean file and compile in a mathlib-enabled project."
    )
}

# Output directory
output_dir = "lean_env/LeanEnv"
os.makedirs(output_dir, exist_ok=True)

# Number of examples to generate
N = 2

for i in range(N):
    object_type = random.choice(object_types)
    topic = random.choice(topics)
    description = generate_description(object_type, topic)

    print(f"[{i+1}/{N}] Generating: {object_type} — {description}")

    user_prompt = {
        "role": "user",
        "content": f"Write a Lean 4 {object_type} corresponding to the following description:\n\n{description}"
    }

    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[system_prompt, user_prompt],
            temperature=0.2,
            max_tokens=800,
        )

        lean_code = clean_output(response.choices[0].message.content.strip())

        file_name = f"{slugify(object_type)}__{slugify(topic)}.lean"
        path = os.path.join(output_dir, file_name)

        with open(path, "w") as f:
            f.write(lean_code)

        print(f"✅ Saved to {path}\n")

    except Exception as e:
        print(f"❌ Failed: {e}\n")

    # Be polite to the API
    time.sleep(1)