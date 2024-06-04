from jericho import *

from langchain_openai import ChatOpenAI
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import queue

def load_text(fpaths, by_lines=False):
    with open(fpaths, "r") as fp:
        if by_lines:
            return fp.readlines()
        else:
            return fp.read()
        
def get_txt_obs(obs_list):
    result = ""
    for obs in obs_list:
        result += obs
    return result

# GPT client
api_key = "EMPTY" 
api_base = "http://localhost:11016/v1" 
model = "meta-llama/Meta-Llama-3-8B-Instruct" 

# gpt client
llm = ChatOpenAI(
    openai_api_key = api_key,
    openai_api_base = api_base,
    model_name = model,
    temperature = 0.,
    stop= ["<|eot_id|>", ";\n\n"]
)

# env 
env = FrotzEnv("z-machine-games-master/jericho-game-suite/zork1.z5")
observation, info = env.reset()
done = False

# prompt
system_template = load_text("system_prompt.txt")
system_message = SystemMessage(content=system_template)

# obs queue
action_list = []
obs_list = [observation]

while not done:
    # stack observation
    msg = [system_message]
    for i in range(len(action_list)):
        msg.append(HumanMessage(content=obs_list[i]))
        msg.append(AIMessage(content=action_list[i]))
    msg.append(HumanMessage(content=obs_list[-1]))
    # query llms to get action
    response = llm(msg)
    action = response.content.split("Action: ")[1]
    # interact w/ the env
    observation, reward, done, info = env.step(action)
    # update obs and action list
    obs_list.append(observation)
    action_list.append(response.content)
    obs_list = obs_list[-9:]
    action_list = action_list[-9:]
    # visualize
    print("[Action]", action)
    print("[Observation]", observation)
    print('Total Score', info['score'], 'Moves', info['moves'])
    print("="*60)
print('Scored', info['score'], 'out of', env.get_max_score())