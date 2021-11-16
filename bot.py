from re import search
import discord
from qa_agent import QA_Agent, QA_AgentMultiple
from es_agent import ElasticSearchAgent
import os
from dotenv import load_dotenv

load_dotenv()

client = discord.Client()

es = ElasticSearchAgent('localhost', '9200', 'genshin')
print("Initializing QA Agent")
# qa_agent = QA_Agent()
qa_agent = QA_AgentMultiple()
print("QA Agent initialized")

@client.event
async def on_ready():
    print("Logged in")

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    
    if message.content.startswith('$q '):
        question = message.content[3:]
        es_result = es.search(question, 3)
        if len(es_result) > 0:
            for search_result in es_result:
                passage = search_result['_source']['content']
                ar_score = search_result['_score']
                title = search_result['_source']['title']
                section = search_result['_source']['section']
                multi_res = qa_agent.predict(context=passage, question=question)
                await message.channel.send("**Taken from {} - {} with article confidence of {}**".format(title, section, ar_score))
                for res, score, model in multi_res:
                    await message.channel.send("**{}**".format(res) + " | Confidence : " + str(score) + " | Model : " + model)
            
        else:
            await message.channel.send("No data found on that question. Sorry, Traveler")

client.run(os.environ.get("BOT_TOKEN"))
