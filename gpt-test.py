from openai import OpenAI 
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(
  api_key=OPENAI_API_KEY
)

# response = client.responses.create(
#   model="gpt-4o-mini",
#   input="write a haiku about ai",
#   store=True,
# )

# print(response.output_text);


# ico_sample = """
# - headquarter country: Russia 
# - ICO duration: 30 days 
# - pre-ICO duration: not disclosed 
# - rating: 2.60 
# - ERC20 compliance: Yes 
# - ethereum-based status: Yes 
# - token type: Utility 
# - token price: $1.3500 
# - whitelist process: No 
# - kYC process: No 
# - bounty program: No 
# - team size: 4 members 
# - tokens for sale: 3,170,000 
# - investor token allocation %: not disclosed 
# - soft cap: No 
# - hard cap: No 
# - website availability: Yes 
# - whitepaper availability: Yes 
# - Twitter presence: Yes 
# - GitHub presence: Yes 
# - Telegram presence: Yes
# """

ico_sample = """
The headquarter country is Switzerland. The ICO duration is not disclosed. The pre-ICO duration is not disclosed. The rating(1 to 5) is 2.90. The ERC20 compliance is Yes. The ethereum-based status is No. The token type is missingData. The token price is not disclosed. The whitelist process is Yes. The kYC process is Yes. The bounty program is No. The team size is 20 members. The tokens for sale is 500,000,000. The investor token allocation % is 0.50. The soft cap is missingData. The hard cap is missingData. The website availability is Yes. The whitepaper availability is Yes. The Twitter presence is missingData. The GitHub presence is Yes. The Telegram presence is missingData.

"""

# response = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         # {"role": "system", "content": "You are a fraud detection assistant. Consider an ICO project high-risk if it exhibits multiple common red flags."},
#         {"role": "user", "content": f"Consider an ICO risky if it has multiple strong red flags. Is this ICO high-risk? Answer Yes or No.\n\n{ico_sample}"}
#     ],
#     temperature=0,
# )

# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         # {"role": "system", "content": "You are a fraud detection assistant. Consider an ICO project high-risk if it exhibits multiple common red flags."},
#         {"role": "user", "content": f"You are evaluating whether an ICO project exhibits signs of operational misconduct.\
#          Definition: Operational misconduct refers to patterns consistent with intentional deception in marketing signals. Poor business prospects alone do NOT constitute misconduct.\
#          Guidelines: Only answer 'Yes' if there are strong and credible signals suggesting intentional deception. Missing information or weak fundamentals alone are NOT sufficient. If the evidence is weak or ambiguous, answer 'No'.\
#          Question: Does this ICO project exhibit signs of operational misconduct? Answer (Yes or No):\n\n{ico_sample}"}
#     ],
#     temperature=0,
# )



response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": f"You are evaluating whether an ICO project shows signs of fraud. Fraud refers to potential scam, deceptive behaviour, or misleading claims. Based on the information given, does this ICO project exhibit signs of fraud? Answer with exactly one word: Yes or No.:\n\n{ico_sample}"}
    ],
    temperature=0,
)

print(response.choices[0].message.content)