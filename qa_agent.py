""" 
    13518005 Arung Agamani Budi Putera
    13518036 Fikra Hadi Ramadhan
    13518140 Hansel Grady Daniel Thamrin
"""
from re import L
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from transformers import XLNetForQuestionAnswering, XLNetTokenizerFast
from transformers.pipelines import pipeline

# Contoh pertanyaan dan konteks
question = '''Why was the student group called "the Methodists?"'''
paragraph = ''' The movement which would become The United Methodist Church began in the mid-18th century within the Church of England.
            A small group of students, including John Wesley, Charles Wesley and George Whitefield, met on the Oxford University campus.
            They focused on Bible study, methodical study of scripture and living a holy life.
            Other students mocked them, saying they were the "Holy Club" and "the Methodists", being methodical and exceptionally detailed in their Bible study, opinions and disciplined lifestyle.
            Eventually, the so-called Methodists started individual societies or classes for members of the Church of England who wanted to live a more religious life. '''

# Contoh lain pertanyaan dan konteks
ques2 = '''Who is barbasitos?'''
para2 = ''' Genshin Impact is a free-to-play action RPG developed and published by miHoYo.
            The game features a fantasy open-world environment and action based combat system using elemental magic, character switching, and gacha monetization system for players to obtain new characters, weapons, and other resources.
            The game can only be played with an internet connection and features a limited multiplayer mode allowing up to four players in a world.
            The story of Genshin Impact takes place in the world of Teyvat which is home to seven major nations: Mondstadt, Liyue, Inazuma, Sumeru, Fontaine, Natlan, and Snezhnaya. These nations are ruled over by deities known as Archons, part of a group of gods called The Seven. Each Archon is associated with one of the seven elements of Teyvat, and is also reflected in their nation's landscape and culture. For example, the nation of Mondstadt is ruled over by Barbatos the Anemo Archon, and so the element of wind is an important symbol known by its people. Apart from the seven nations, there was the nation of Khaenri'ah that lived independently from the authority of gods.
            Celestia is the main heavenly body that presides over Teyvat and is in direct connection with The Seven. A floating island seen in the sky is believed to be it, allegedly the home to gods and also mortals who have ascended to godhood. Humans that are granted Visions, magical gems that grant bearers the ability to command elemental power, are called "allogenes" ??? ones with the potential to reach godhood.
            While gods rule, the nations also have their own human governing bodies: the Knights of Favonius to Mondstadt, the Qixing to Liyue, and the Shogunate to Inazuma. '''

# QA Agent dengan menggunakan BERT
# Pembuatan model dengan menggunakan pre-trained model yang telah
# di fine-tuned dengan menggunakan SQuAD v1.1
def bert_qa():
    MODEL = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    model = BertForQuestionAnswering.from_pretrained(MODEL)
    tokenizer = BertTokenizer.from_pretrained(MODEL)
    encoding = tokenizer.encode_plus(text=ques2, text_pair=para2, add_special_tokens=True)
    # print(encoding)

    inputs = encoding['input_ids']
    sentence_embedding = encoding['token_type_ids']
    tokens = tokenizer.convert_ids_to_tokens(inputs)

    output = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))
    # print(start_scores, end_scores)
    start_index = torch.argmax(output.start_logits)
    end_index = torch.argmax(output.end_logits)

    answer = ' '.join(tokens[start_index:end_index+1])
    # print("Question: ", ques2)
    corrected_answer = ''
    for word in answer.split():
        if word[0:2] == '##':
            corrected_answer += word[2:]
        else:
            corrected_answer += ' ' + word
    print("Answer: ", corrected_answer)

# QA Agent dengan menggunakan BERT dengan simplifikasi menggunakan
# Pipeline model dari library Transformers
def bert_qa_pipeline():
    MODEL = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    nlp = pipeline('question-answering', model=MODEL, tokenizer=MODEL)
    qa_input = {
        'question': ques2,
        'context': para2
    }
    res = nlp(qa_input)
    print("Answer: ", res['answer'])

# QA Agent dengan menggunakan XLM-RoBERTa-Large
# di-finetuned dengan menggunakan SQuAD v2.0
def xlm_qa():
    MODEL = 'deepset/xlm-roberta-large-squad2'
    # model = AutoModelForQuestionAnswering.from_pretrained(MODEL)
    # tokenizer = AutoTokenizer.from_pretrained(MODEL)
    nlp = pipeline('question-answering', model=MODEL, tokenizer=MODEL)
    qa_input = {
        'question': ques2,
        'context': para2
    }
    res = nlp(qa_input)
    print("Answer: ", res['answer'])

# QA Agent dengan menggunakan distilBERT-Base
# di-finetuned dengan menggunakan SQuAD v2.0
def distilbert_qa():
    MODEL = 'twmkn9/distilbert-base-uncased-squad2'
    nlp = pipeline('question-answering', model=MODEL, tokenizer=MODEL)
    qa_input = {
        'question': ques2,
        'context': para2
    }
    res = nlp(qa_input)
    print("Answer: ", res['answer'])

# Definisi dari class QA_Agent, di mana class ini menggunakan model BERT sebagi
# basis model untuk melakukan prediksi jawaban dari konteks dan pertanyaan
class QA_Agent():
    def __init__(self, model="bert-large-uncased-whole-word-masking-finetuned-squad") -> None:
        self.question = ""
        self.context = ""
        self.answer = ""
        self.pipeline = pipeline('question-answering', model=model, tokenizer=model)
    
    def predict(self, question, context):
        qa_input = {
            'question': question,
            'context': context
        }
        result = self.pipeline(qa_input)
        return (result['answer'], result['score'])

# Definisi dari QA Agent namun menggunakan 3 model yang berbeda secara sekaligus
# Prediksi akan menghasilkan 3 jawaban yang berasal dari hasil prediksi setiap model yang telah
# dijelaskan di atas
class QA_AgentMultiple():
    def __init__(self) -> None:
        MODEL1 = 'bert-large-uncased-whole-word-masking-finetuned-squad'
        MODEL2 = 'deepset/xlm-roberta-large-squad2'
        MODEL3 = 'twmkn9/distilbert-base-uncased-squad2'
        self.question = ""
        self.context = ""
        self.answer = ""
        self.pipeline1 = pipeline('question-answering', model=MODEL1, tokenizer=MODEL1)
        self.pipeline2 = pipeline('question-answering', model=MODEL2, tokenizer=MODEL2)
        self.pipeline3 = pipeline('question-answering', model=MODEL3, tokenizer=MODEL3)
    
    def predict(self, question, context):
        res = []
        qa_input = {
            'question': question,
            'context': context
        }
        result1 = self.pipeline1(qa_input)
        result2 = self.pipeline2(qa_input)
        result3 = self.pipeline3(qa_input)
        res.append((result1['answer'], result1['score'], 'BERT-Large-Uncased'))
        res.append((result2['answer'], result2['score'], 'XLM-RoBERTa-Large'))
        res.append((result3['answer'], result3['score'], 'distilBERT-Base-Uncased'))
        return res

if __name__ == '__main__':
    print("Begin QA task")
    print("Question: ", ques2)
    print("===XLM-RoBERTa-Large")
    xlm_qa()
    print("===BERT-large-uncased")
    bert_qa()
    print("===distilBERT-base-uncased")
    distilbert_qa()
    print("End QA task")