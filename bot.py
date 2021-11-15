import discord
from qa_agent import QA_Agent
from es_agent import ElasticSearchAgent

client = discord.Client()

es = ElasticSearchAgent('localhost', '9200', 'genshin')
print("Initializing QA Agent")
qa_agent = QA_Agent()
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
            passage = es_result[0]['_source']['content']
            res = qa_agent.predict(context=passage, question=question)
            await message.channel.send(res)
            
        else:
            await message.channel.send("No data found on that question. Sorry, Traveler")

client.run('Njk2Nzg0MDU1OTExNDQ4NjQ3.XotwuA.62zB4p6YByUPrdraMTr8FbWRrjc')