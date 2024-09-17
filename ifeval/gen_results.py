from openai import OpenAI
import concurrent
import json
import tqdm

client = OpenAI(
    base_url="http://0.0.0.0:5050/v1",
    api_key="test",
    )



def generate_one(row):
    prompt = row["prompt"]
    system = '''You are a world-class AI system capable of complex reasoning and reflection. You respond to all questions in the following way-
<thinking>
In this section you understand the problem and develop a plan to solve the problem.

For easy problems-
Make a simple plan and use COT

For moderate to hard problems-
1. Devise a step-by-step plan to solve the problem. (don't actually start solving yet, just make a plan)
2. Use Chain of Thought  reasoning to work through the plan and write the full solution within thinking.

You can use <reflection> </reflection> tags whenever you execute a complex step to verify if your reasoning is correct and if not correct it.


</thinking>

<output>
In this section, provide the complete answer for the user based on your thinking process. Do not refer to the thinking tag. Include all relevant information and keep the response somewhat verbose, the user will not see what is in the thinking tag.
</output>'''
    messages = [{"role":"system","content":system},{"role": "user", "content":prompt}]
    response = client.chat.completions.create(
        model="sahil2801/test_reflect",
        messages=messages,
        temperature=0.0,
        max_tokens=6000,
        stream=False,
        extra_body={"skip_special_tokens": False},
    )
    try:
        return {"prompt":prompt, "response":response.choices[0].message.content.split("<output>")[1].replace("</output>", "").strip()}
    except:
        return {"prompt":prompt, "response":response.choices[0].message.content}

def load_data():
    with open("data/ifeval_input_data.jsonl") as f:
        data = [json.loads(line) for line in f]

    return data
def write_to_jsonl(data, filename):
    with open(filename, 'w') as file:
        for item in data:
            json_line = json.dumps(item)
            file.write(json_line + '\n')


responses = []

with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
    tasks = [executor.submit(generate_one,row) for row in load_data()]
    for future in tqdm.tqdm(concurrent.futures.as_completed(tasks),total=len(tasks),desc=f'''Generating reflection responses'''):
        result = future.result()
        if result is not None:
            responses.append(result)


write_to_jsonl(responses, 'data/reflection_output.jsonl')