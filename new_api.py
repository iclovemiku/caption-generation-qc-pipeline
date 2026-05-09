import openai
import httpx
import traceback



api_key = "sk-uy8gDUxOY0xRViZhi1MU1MCNcghzvHTAIYxmPgw9UXE36TCb"
base_url = "http://115.120.87.159:8082/"


# 这是问题
questions = f"""
介绍自己
"""


def get_response(question, api_key, base_url):
    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url, http_client=httpx.Client(verify=False))
        response = client.chat.completions.create(
            model="claude-sonnet-4-6-thinking",
            # model="glmmoedsa",
            messages=[{"role": "user", "content": question}],

        )
        return response.choices[0].message.content
    except Exception as e:
        traceback.print_exc()
        return f"请求失败: {str(e)}"


if __name__ == "__main__":
    response = get_response(questions, api_key, base_url)
    print(f"回答: {response}\n")
