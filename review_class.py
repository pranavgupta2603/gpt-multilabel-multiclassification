from openai import OpenAI
from modules import LLM
import json
import pandas as pd
from tqdm import tqdm

api_key = "<enter_key>"
client = OpenAI(api_key=api_key)  

labels = [
        "App Crashes", "App Responsiveness", "Connectivity (Wi-Fi/4G issues)", "Battery Usage",
        "Storage Usage", "Data Usage", "App Size", "User Interface (UI)", "Ease of Use",
        "Sign-up Process", "User Onboarding", "Notifications", "Accessibility", "Language Support",
        "Customizability", "Security", "Data Privacy", "App Permissions", "Customer Support",
        "Feature Requests", "Updates", "Bugs", "Pricing", "Account Management", "Advertisements", "Others"
    ]

system_prompt = """You are an expert at analyzing reviews and categorizing them into relevant labels from a given list. Based on the review provided, identify all the labels that accurately match the content and context of the review.

Review:
“Insert review text here.”

Available Labels:
	•⁠  ⁠Performance
    •⁠  ⁠App Crashes
    •⁠  ⁠App Responsiveness
    •⁠  ⁠Connectivity (Wi-Fi/4G issues)
    •⁠  ⁠Battery Usage
    •⁠  ⁠Storage Usage
    •⁠  ⁠Data Usage
    •⁠  ⁠App Size
    •⁠  ⁠User Interface (UI)
    •⁠  ⁠Ease of Use
    •⁠  ⁠Sign-up Process
    •⁠  ⁠User Onboarding
    •⁠  ⁠Notifications
    •⁠  ⁠Accessibility
    •⁠  ⁠Language Support
    •⁠  ⁠Customizability
    •⁠  ⁠Security
    •⁠  ⁠Data Privacy
    •⁠  ⁠App Permissions
    •⁠  ⁠Customer Support
    •⁠  ⁠Feature Requests
    •⁠  ⁠Updates
    •⁠  ⁠Bugs
    •⁠  ⁠Pricing
    •⁠  ⁠Account Management
    •⁠  ⁠Advertisements
    •⁠  Others
    

Instructions:
	1.	Analyze the review carefully and consider its tone, content, and context.
	2.	From the list of available labels, select all the labels that apply to the review.
	3.	Only choose labels that are directly relevant to the review's content.
    4.  If none of the labels apply, select 'Others'.

Output Format:
[List of matching labels]

Example Format:
['Performance', 'App Crashes', 'User Interface (UI)', 'Security']
"""

llm = LLM(client, system_prompt)

"""resp, prompt_tokens, completion_tokens = llm.get_description("gpt-4o-mini", [llm.text_content("It's an amazing concept. Where everyone will be benefited.")])
resp = eval(resp)
print(resp["final_answer"])
print(prompt_tokens*0.15/1e6 + completion_tokens*0.6/1e6)"""

reviews = pd.read_csv("data.csv", usecols=["Review"])
try:
    df = pd.read_csv("output.csv")
    reviews = reviews[~reviews["Review"].isin(df["Review"])]
    columns = df.columns
except:
    columns = ["Review", "Price"] + labels
    df = pd.DataFrame(columns=columns)
    
reviews = reviews["Review"].tolist()
# List of labels provided
total_price = 0
for idx, review in tqdm(enumerate(reviews)):
    resp, prompt_tokens, completion_tokens = llm.get_description("gpt-4o-mini", [llm.text_content(review)])
    resp = eval(resp)
    print(resp["final_answer"])
    total_price += prompt_tokens*0.15/1e6 + completion_tokens*0.6/1e6
    row = [review, prompt_tokens*0.15/1e6 + completion_tokens*0.6/1e6]
    for label in labels:
        if label in resp["final_answer"]:
            row.append(1)
        else:
            row.append(0)
    df = df._append(pd.Series(row, index=columns), ignore_index=True)
    df.to_csv("output.csv", index=False)
    if idx == 100:
        print("Total cost: ", total_price)