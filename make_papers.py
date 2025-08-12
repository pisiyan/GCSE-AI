import pickle
from openai import OpenAI
import numpy as np
import random
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import json
import os
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = api_key
debug = False


class GcseAssistantMakeExams:

    def __init__(self, subject, examiner):
        print("Initializing...")
        self.EXAMPLE_QUESTIONS = 8
        self.subject = subject
        self.examiner = examiner
        self.vectorDatabase = self.subject + "-" + self.examiner + "-vectorDatabase"
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = FAISS.load_local("data/rs/aqa/ReligiousStudies-AQA-vectorDatabase", self.embedding_model)
        self.llm_model = "gpt-3.5-turbo"
        self.llm = ChatOpenAI(model_name=self.llm_model, temperature=0)
        self.question_retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        self.question_qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=self.question_retriever)
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=self.question_retriever)
        with open("questions.pkl", "rb") as file:
            self.questions = pickle.load(file)
        self.prompts = self.load_inputs("prompts")
        self.queries = self.load_inputs("queries")
        self.client = OpenAI()
        if debug:
            print("Initializing complete\n")
            print(self.prompts)
            print(self.queries)

    def semantic_similarity_score_list(self, unused, used, model="text-embedding-3-small"):
        embeddings_unused_resp = self.client.embeddings.create(model=model, input=unused).data
        embeddings_used_resp = self.client.embeddings.create(model=model, input=used).data

        embeddings_unused = [np.array(item.embedding) for item in embeddings_unused_resp]
        embeddings_used = [np.array(item.embedding) for item in embeddings_used_resp]

        scores = []
        for i, emb_a in enumerate(embeddings_unused):
            sims = []
            for emb_b in embeddings_used:
                sim = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
                sims.append(sim)
            score = max(s for s in sims)
            scores.append((unused[i], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def pick_random_from_lower_half(self, scores):
        if len(scores) == 1:
            return scores[0]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        half_index = len(scores) // 2
        lower_half = scores[half_index:]

        if not lower_half:
            return None
        return random.choice(lower_half)[0]

    def load_inputs(self, folder_path):
        inputs = {}
        for filename in os.listdir(folder_path):
            full_path = os.path.join(folder_path, filename)
            with open(full_path, "r") as file:
                content = file.read()
            inputs[filename.replace(".txt", "")] = content
        return inputs

    def test_qa(self, query):
        docs = self.question_retriever.invoke(query)

        print("\n--- Retrieved Chunks ---")
        for i, doc in enumerate(docs):
            print(f"\nChunk {i + 1}:\n{doc.page_content}\n")

        result = self.question_qa_chain.invoke({"query": query})

        print("\n--- LLM Answer ---")
        print(result["result"])

    def convert_exam_structure(self, ai_exam_structure, user_input):
        topics = self.get_topics(user_input)
        prompt = self.prompts["format_exam_structure"].format(
            ai_exam_structure=ai_exam_structure,
            user_input=user_input,
            topics=topics
        )
        result = json.loads(self.llm.invoke(prompt).content)
        return result

    def get_exam_structure(self, topic):
        query = self.queries["get_exam_structure"].format(
            topic=topic,
            subject=self.subject,
            examiner=self.examiner
        )
        result = self.qa_chain.invoke(query)["result"]
        return result

    def get_subtopics(self, question_topic):
        spec_point_raw = self.get_spec_point_raw(question_topic)
        spec_point = self.format_spec_point(spec_point_raw)
        specification_result = self.get_specification_result(question_topic, spec_point)
        list_spec_subtopics = self.format_spec_subtopics(specification_result)

        if debug:
            print("SPEC POINT")
            print(spec_point)
            print("\nSPEC TOPIC INFO")
            print(specification_result)
            print("\nSPEC TOPIC LIST -FINAL")
            print(list_spec_subtopics)

        return list_spec_subtopics

    def get_spec_point_raw(self, question_topic):
        query = self.queries["get_spec_point_for_topic"].format(
            question_topic=question_topic
        )
        return self.qa_chain.invoke(query)["result"]

    def format_spec_point(self, spec_point_raw):
        prompt = self.prompts["retrieve_subtopic_spec_point"].format(
            spec_point_raw=spec_point_raw
        )
        return self.llm.invoke(prompt).content

    def get_specification_result(self, question_topic, spec_point):
        query = self.queries["get_topic_info"].format(
            question_topic=question_topic,
            spec_point=spec_point
        )
        return self.qa_chain.invoke(query)["result"]

    def format_spec_subtopics(self, specification_result):
        prompt = self.prompts["format_subtopics_from_spec"].format(
            specification_result=specification_result
        )
        return json.loads(self.llm.invoke(prompt).content)

    def process_questions_for_subtopics(self, topic_of_exam, specification_result):
        filtered_questions = []
        subtopics_from_questions = []

        for question in self.questions:
            if question["topic"] == topic_of_exam and question["examiner"] == self.examiner:
                filtered_questions.append(question["content"])

                if len(filtered_questions) == self.EXAMPLE_QUESTIONS:
                    subtopics_from_questions += self.process_question_batch(
                        topic_of_exam, filtered_questions, specification_result
                    )
                    filtered_questions.clear()

        return subtopics_from_questions

    def process_question_batch(self, topic_of_exam, filtered_questions, specification_result):
        prompt = self.prompts["get_subtopics_from_questions"].format(
            topic_of_exam=topic_of_exam,
            subject=self.subject,
            examiner=self.examiner,
            filtered_questions=filtered_questions
        )
        result = self.llm.invoke(prompt).content
        topics_of_questions = json.loads(result)

        prompt = self.prompts["filter_irrelevant_topics_from_questions"].format(
            topic_of_exam=topic_of_exam,
            subject=self.subject,
            examiner=self.examiner,
            specification_result=specification_result,
            topics_of_questions=topics_of_questions
        )
        result = self.llm.invoke(prompt).content
        filtered_topics = json.loads(result)

        if debug:
            print(topics_of_questions)
            print(filtered_topics)

        return filtered_topics

    def get_common_subtopics(self, subtopics_from_questions, list_spec_subtopics):
        prompt = self.prompts["get_common_subtopics"].format(
            subject=self.subject,
            examiner=self.examiner,
            subtopics_from_questions=subtopics_from_questions,
            list_spec_subtopics=list_spec_subtopics
        )
        return self.llm.invoke(prompt).content

    def get_subtopic_info(self, subtopic, topic):
        query = self.queries["get_subtopic_info"].format(
            topic=topic,
            subtopic=subtopic
        )
        result = self.qa_chain.invoke(query)["result"]
        return result

    def get_random_questions(self, marks, topic):
        filtered_questions = []
        for question in self.questions:
            if question["topic"] == topic and question["marks"] == marks:
                filtered_questions.append(question)
        random_questions = []
        for i in range(self.EXAMPLE_QUESTIONS):
            rq = random.choice(filtered_questions)
            random_questions.append(rq["content"])
            filtered_questions.remove(rq)
        return random_questions

    def make_question(self, marks, exam_topic, question_topic, subtopic):

        random_questions = self.get_random_questions(marks, exam_topic)
        topic_info = self.get_subtopic_info(subtopic, question_topic)
        prompt = self.prompts["make_question"].format(
            subject=self.subject,
            examiner=self.examiner,
            marks=marks,
            random_questions=random_questions,
            topic_info=topic_info
        )
        if debug:
            print(random_questions)
            print(topic_info)

        return self.llm.invoke(prompt).content

    def get_topics(self, user_input):
        prompt = self.prompts["get_topics_from_user"].format(
            user_input=user_input
        )
        return self.llm.invoke(prompt).content

    def make_exam(self, topic, user_input):
        exam_structure = self.convert_exam_structure(self.get_exam_structure(topic), user_input)
        if debug: print(f"EXAM STRUCTURE: {exam_structure}")
        for exam_topic in exam_structure.keys():
            subtopics = rs_aqa_assistant.get_subtopics(exam_topic)
            used = []
            if debug: print(f"POSSIBLE SUBTOPICS: {subtopics}")
            i = 0
            for mark in exam_structure[exam_topic]:
                i += 1
                if debug: print(f"\nQUESTION {i}: {mark} MARKS")
                if len(used) == 0:
                    subtopic = random.choice(subtopics)
                else:
                    scores = self.semantic_similarity_score_list(subtopics, used)
                    subtopic = self.pick_random_from_lower_half(scores)
                if debug: print(f"Subtopic: {subtopic}")
                subtopics.remove(subtopic)
                used.append(subtopic)
                question = self.make_question(mark, topic, exam_topic, subtopic)
                print()
                l = len(question.split("\n")[0])
                if debug: print("*" * l)
                print(question)
                if debug: print("*" * l)
                print()


rs_aqa_assistant = GcseAssistantMakeExams("ReligiousStudies", "AQA")

user_input = "make a 24 mark RS AQA GCSE Themes paper which has 1 topic: (The existence of God and revelation)"

rs_aqa_assistant.make_exam("Themes", user_input)
