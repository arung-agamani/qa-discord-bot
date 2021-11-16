""" 
    13518005 Arung Agamani Budi Putera
    13518036 Fikra Hadi Ramadhan
    13518140 Hansel Grady Daniel Thamrin
"""

from re import search
import discord
from qa_agent import QA_Agent, QA_AgentMultiple
from es_agent import ElasticSearchAgent
import os
from dotenv import load_dotenv

load_dotenv()

""" Bot Discord Question and Answering
    Menggunakan ElasticSearchAgent untuk melakukan pencaharian dokumen yang berkaitan dengan pertanyaan.
    Menggunakan QA Agent yakni agent Question and Answering yang dibangun menggunakan Transformers.
        Adapun QA Agent menggunakan model-model seperti BERT, XLM-RoBERTa, dan distilBERT
"""

client = discord.Client()

es = ElasticSearchAgent('localhost', '9200', 'genshin')
print("Initializing QA Agent")
# qa_agent = QA_Agent()
qa_agent = QA_AgentMultiple()
print("QA Agent initialized")

@client.event
async def on_ready():
    print("Logged in")

# Definisi dari handler untuk message yang datang pada bot.
@client.event
async def on_message(message):
    if message.author == client.user:
        return
    
    # Apabila bot menerima message dengan awalan prefix tertentu, maka akan dianggap sebagai
    # suatu query untuk melakukan pertanyaan. Pertanyaan akan diekstraksi lalu diberikan ke
    # ElasticSearchAgent untuk melakukan pencaharian dokumen.
    # Hasil pencaharian lalu dilakukan prediksi pada model yang ada untuk mendapatkan jawaban
    # Jawaban lalu dikembalikan ke pengguna sebagai suatu message baru pada Discord.
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

# Jalankan bot
client.run(os.environ.get("BOT_TOKEN"))
