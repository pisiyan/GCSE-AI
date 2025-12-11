import pickle
from langchain.schema import HumanMessage
from load_and_store import DatabaseManager
from openai import OpenAI
import numpy as np
import random
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import json
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = api_key
debug = False


class GcseAssistantMakeExams:

    def __init__(self, subject, examiner):
        print("Initializing...")
        self.subject = subject
        self.examiner = examiner
        self.subject_info = self.get_subject_info()
        self.EXAMPLE_QUESTIONS = int(self.subject_info["example_questions"])
        self.vectorDatabase = self.subject + "-" + self.examiner + "-vectorDatabase"
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = FAISS.load_local(
            f"data/{self.subject}/{self.examiner}/{self.subject}-{self.examiner}-vectorDatabase",
            self.embedding_model
        )
        self.llm_model = "gpt-4o"
        self.spec_llm_model = "gpt-4o"
        self.llm = ChatOpenAI(model_name=self.llm_model, temperature=0)
        self.spec_llm = ChatOpenAI(model_name=self.spec_llm_model, temperature=0)
        self.image_llm = ChatOpenAI(model_name=self.llm_model, temperature=0)
        self.spec_retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={
            "k": int(self.subject_info["spec_search_kwargs_k"]),
            "filter": {"type": "Specification"}
        })
        self.ms_retriever = self.vectorstore.as_retriever(search_type="mmr", search_kwargs={
            "k": int(self.subject_info["ms_search_kwargs_k"]),
            "filter": {"type": "MarkScheme"}
        })
        self.ms_qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=self.ms_retriever)
        self.spec_qa_chain = RetrievalQA.from_chain_type(llm=self.spec_llm, retriever=self.spec_retriever)
        with open(f"data/{self.subject}/{self.examiner}/questions.pkl", "rb") as file:
            self.questions = pickle.load(file)
        with open(f"data/{self.subject}/{self.examiner}/msquestions.pkl", "rb") as file:
            self.mark_schemes = pickle.load(file)
        self.prompts = self.load_inputs("prompts")
        self.queries = self.load_inputs("queries")
        self.client = OpenAI()
        if debug:
            print("Initializing complete\n")
            print(self.prompts)
            print(self.queries)

    def get_subject_info(self):
        with open("subject_info.txt", "r") as file:
            subject_info_file = file.read().split("\n")

        subject = {}
        subject_info = {}
        for line in subject_info_file:
            info = line.split(" ")
            if len(info) == 1:
                try:
                    subject_info[subject_name] = subject
                    subject = {}
                    subject_name = info[0]
                except:
                    subject_name = info[0]
            else:
                subject[info[0]] = info[1]

        return subject_info[f"{self.subject}-{self.examiner}"]

    def choose_relevant_parent_description(self, subtopic_info):
        best = [0, ""]
        for question in self.questions:
            if question["type"] == "parent_question":
                description = question["parent_question_description"]
                similarity = self.semantic_similarity(description, subtopic_info)
                if similarity > best[0]:
                    best = [similarity, description]
        return best[1]

    def semantic_similarity(self, x, y, model="text-embedding-3-small"):
        embeddings_resp = self.client.embeddings.create(
            model=model,
            input=[x, y]
        ).data

        emb_x = np.array(embeddings_resp[0].embedding)
        emb_y = np.array(embeddings_resp[1].embedding)
        sim = np.dot(emb_x, emb_y) / (np.linalg.norm(emb_x) * np.linalg.norm(emb_y))
        return sim

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
        a = random.choice(lower_half)
        if isinstance(a[0], tuple):
            return a[0][0]
        else:
            return a[0]

    def load_inputs(self, folder_path):
        inputs = {}
        for filename in os.listdir(folder_path):
            full_path = os.path.join(folder_path, filename)
            with open(full_path, "r") as file:
                content = file.read()
            inputs[filename.replace(".txt", "")] = content
        return inputs

    def test_qa(self, type, query):
        if type == "ms":
            retriever = self.ms_retriever
            qa_chain = self.ms_qa_chain
        else:
            retriever = self.spec_retriever
            qa_chain = self.spec_qa_chain
        # docs = retriever.invoke(query)

        # print("\n--- Retrieved Chunks ---")
        # for i, doc in enumerate(docs):
        #     print(f"\nChunk {i + 1}:\n{doc.page_content}\n")

        result = qa_chain.invoke({"query": query})

        # print("\n--- LLM Answer ---")
        # print(result["result"])
        return result["result"]

    def test_llm(self, input):
        content = self.llm.invoke(input).content
        return content

    def convert_exam_structure(self, base_exam_structure, topics):
        n = len(base_exam_structure)
        k = len(topics)
        chunk_size = n // k
        remainder = n % k

        result = {}
        start = 0

        for i, key in enumerate(topics):
            end = start + chunk_size + (1 if i < remainder else 0)
            result[key] = base_exam_structure[start:end]
            start = end


        # prompt = self.prompts["format_exam_structure"].format(
        #     base_exam_structure=result,
        #     user_input=user_input,
        #     marks=marks
        # )
        # result = json.loads(self.llm.invoke(prompt).content)
        return result

    def flatten(self, lst):
        try:
            result = 0
            for item in lst:
                if isinstance(item, list):
                    for a in item:
                        result += a
                else:
                    result += item
            return result
        except:
            return lst

    def get_past_exam_structures(self, topic):
        current_exam = ""
        exam_marks = []
        possible_marks_exams = []
        for question in self.questions:
            if question["topic"] == topic and (question["type"] == "parent_question" or question["type"] == "basic_question"):
                if question["exam"] != current_exam:
                    if len(exam_marks) > 0:
                        possible_marks_exams.append(exam_marks)
                    exam_marks = []
                    current_exam = question["exam"]
                if question["type"] == "parent_question":
                    marks = [self.flatten(question["parent_question_structure"]), "parent"]
                else:
                    marks = [question["marks"], "basic"]
                exam_marks.append(marks)
        return possible_marks_exams

    def make_exam_structure(self, marks_of_exam, topic):
        past_exam_structures = self.get_past_exam_structures(topic)
        average_marks_per_question = sum(mark[0] for mark in past_exam_structures[0]) / len(past_exam_structures[0])
        n_questions = round(marks_of_exam / average_marks_per_question)
        while True:
            exam_structure = []
            last_type = ""
            marks_left = marks_of_exam
            for i in range(n_questions):
                possible_marks_for_question = []
                for exam in past_exam_structures:
                    try:
                        possible_marks_for_question.append(exam[i])
                    except:
                        if debug: print(len(exam), past_exam_structures.index(exam))
                possible_marks = [mark for mark in possible_marks_for_question if mark[0] <= marks_left]
                try:
                    best_mark = random.choice(possible_marks)
                except:
                    best_mark = [0, last_type]
                exam_structure.append(best_mark)
                marks_left -= best_mark[0]
                last_type = best_mark[1]
            if marks_left != 0:
                exam_structure[-1][0] += marks_left

            try:
                final_exam_structure = []
                if debug: print(exam_structure)
                i = 0
                q_no = 0
                last_exam = ""
                for question in exam_structure:
                    structure = question[0]
                    if question[1] == "parent":
                        possible_structures = []
                        for q in self.questions:
                            if q["type"] == "parent_question" or q["type"] == "basic_question":
                                q_no += 1
                                if q["exam"] != last_exam:
                                    last_exam = q["exam"]
                                    q_no = 0
                                if self.flatten(q["parent_question_structure"]) == question[0] and q["topic"] == topic and (q_no == exam_structure.index(question) or not eval(self.subject_info["question_no_importance"])):
                                    possible_structures.append(q["parent_question_structure"])
                        structure = random.choice(possible_structures)
                        q_no += 1
                    final_exam_structure.append(structure)
                    i += 1
                print(final_exam_structure)
                return final_exam_structure
            except Exception as error:
                if debug:
                    print("An error occurred:", type(error).__name__, "–", error)

    def get_subtopics(self, question_topic):
        specification_result = self.get_specification_result(question_topic)
        list_spec_subtopics = self.format_spec_subtopics(specification_result)

        if debug:
            print("\nSPEC TOPIC INFO")
            print(specification_result)
            print("\nSPEC TOPIC LIST -FINAL")
            print(list_spec_subtopics)

        return list_spec_subtopics

    def get_spec_point_raw(self, question_topic):
        query = self.queries["get_spec_point_for_topic"].format(
            question_topic=question_topic
        )
        return self.test_qa("spec", query)

    def format_spec_point(self, spec_point_raw):
        prompt = self.prompts["retrieve_subtopic_spec_point"].format(
            spec_point_raw=spec_point_raw,
            spec_point_length=1
        )
        return self.llm.invoke(prompt).content

    def get_specification_result(self, question_topic):
        query = self.queries["get_topic_info"].format(
            question_topic=question_topic,
        )
        return self.test_qa("spec", query)

    def format_spec_subtopics(self, specification_result):
        prompt = self.prompts["format_subtopics_from_spec"].format(
            specification_result=specification_result
        )
        subtopics = self.test_llm(prompt)
        print(subtopics)
        return json.loads(subtopics)

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
        result = self.test_qa("spec", query)
        return result

    def get_subsubtopic_info(self, subsubtopic, subtopic, topic):
        query = self.queries["get_subsubtopic_info"].format(
            topic=topic,
            subtopic=subtopic,
            subsubtopic=subsubtopic
        )
        result = self.spec_qa_chain.invoke(query)["result"]
        return result

    def get_random_objects(self, marks, topic, object_list, n, comparison):
        filtered_objects = []
        random.shuffle(object_list)
        for obj in object_list[:n*2]:
            if obj["topic"] == topic and obj["marks"] == marks:
                score = self.semantic_similarity(comparison, obj["question_content"])
                filtered_objects.append([score, obj["question_content"]])
        objects = []
        filtered_objects.sort()
        for obj in filtered_objects[:n]:
            objects.append(obj[1])
        return objects

    def make_question(self, marks, exam_topic, question_content, subtopic_info, parent_description, subtopic):

        random_questions = self.get_random_objects(marks, exam_topic, self.questions, self.EXAMPLE_QUESTIONS, subtopic)
        prompt_extention = ""
        if question_content != "":
            prompt_extention = self.prompts["question_prompt_extention"].format(
                question_content=question_content,
                parent_description=parent_description
            )
        prompt = self.prompts["make_question"].format(
            subject=self.subject,
            examiner=self.examiner,
            marks=marks,
            random_questions=random_questions,
            topic_info=subtopic_info,
            subtopic=subtopic
        ) + "\n" + prompt_extention


        return self.llm.invoke(prompt).content

    def get_topics(self, user_input):
        prompt = self.prompts["get_topics_from_user"].format(
            user_input=user_input
        )
        return self.llm.invoke(prompt).content

    def make_parent_description(self, subtopic, exam_topic, question_content, subtopic_info, parent_description):
        valid_descriptions = []
        for question in self.questions:
            if question["type"] == "parent_question" and question["topic"] == exam_topic:
                valid_descriptions.append(question)
        n = int(self.subject_info["example_descriptions"])
        random_descriptions = []
        random.shuffle(valid_descriptions)
        for description in valid_descriptions[:n*2]:
            score = self.semantic_similarity(subtopic, description["parent_question_description"])
            random_descriptions.append([score, description["question_content"]])
        close_descriptions = []
        random_descriptions.sort()
        for description in random_descriptions[:n]:
            close_descriptions.append(description[1])
        if debug:
            for a in close_descriptions: print(a)
        prompt_extention = ""
        if question_content != "":
            prompt_extention = self.prompts["question_prompt_extention"].format(
                question_content=question_content,
                parent_description=parent_description
            )
        prompt = self.prompts["extract_relevant_text"].format(
            random_questions=random_descriptions,
        ) + "\n" + prompt_extention
        random_descriptions = self.llm.invoke(prompt).content
        if debug:
            print(subtopic_info)
            print(random_descriptions)
        prompt = self.prompts["make_parent_description"].format(
            random_questions=random_descriptions,
            subtopic=subtopic,
            subtopic_info=subtopic_info
        )
        return self.llm.invoke(prompt).content

    def adapt_question(self, marks, exam_topic, subtopic_info, subtopic, question_content, parent_description):
        prompt_extention = ""
        if question_content != "":
            prompt_extention = self.prompts["question_prompt_extention"].format(
                question_content=question_content,
                parent_description=parent_description
            )
        example_question = self.get_random_objects(marks, exam_topic, self.questions, self.EXAMPLE_QUESTIONS, subtopic)
        prompt = self.prompts["adapt_question"].format(
            subject=self.subject,
            subtopic_info=subtopic_info,
            example_question=example_question,
            subtopic=subtopic
        ) + "\n\n"+ prompt_extention
        return self.llm.invoke(prompt).content

    def improve_question_coherance(self, question_content):
        prompt = self.prompts["improve_question_coherance"].format(
            subject=self.subject,
            question_content=question_content
        )
        return self.llm.invoke(prompt).content

    def make_exam(self, exam_topic, user_input, marks, topics):
        exam_structure = self.convert_exam_structure(self.make_exam_structure(marks, exam_topic), topics)
        print(f"EXAM STRUCTURE: {exam_structure}")
        letters = "abcdefghijklmno"
        romans = ["i", "ii", "iii", "iv", "v"]
        for topic in exam_structure.keys():
            subtopics = self.get_subtopics(exam_topic + " " + topic)
            print(subtopics)
            used = []
            if debug: print(f"POSSIBLE SUBTOPICS: {subtopics}")
            i = 0
            for mark in exam_structure[topic]:
                i += 1
                print(f"\nQUESTION {i}: {self.flatten(mark)} MARKS")
                if len(used) == 0:
                    subtopic = random.choice(subtopics)
                else:
                    scores = self.semantic_similarity_score_list(subtopics, used)
                    subtopic = self.pick_random_from_lower_half(scores)
                if debug: print(f"Subtopic: {subtopic}")
                subtopics.remove(subtopic)
                used.append(subtopic)
                subtopic_info = self.get_subtopic_info(subtopic, topic)
                print(subtopic.upper())
                if isinstance(mark, int):
                    question_content = f"{i}) " + self.make_question(mark, exam_topic, "", subtopic_info, "", subtopic) + f" [{mark} marks]\n\n"
                    print(question_content)
                else:
                    parent_description = self.choose_relevant_parent_description(subtopic)

                    question_content = parent_description + f"\n\n"
                    print(f"{i}) " + question_content)
                    subsubtopics = self.format_spec_subtopics(subtopic_info)
                    print("Subsubtopics:", subsubtopics)
                    used_subsubtopics = []
                    z = -1
                    for x in mark:
                        z += 1
                        if len(used_subsubtopics) < 2:
                            subsubtopic = random.choice(subsubtopics)
                        else:
                            scores = self.semantic_similarity_score_list(subsubtopics, used_subsubtopics)
                            subsubtopic = self.pick_random_from_lower_half(scores)
                        subsubtopics.remove(subsubtopic)
                        used_subsubtopics.append(subsubtopic)
                        subsubtopic_info = self.get_subsubtopic_info(subsubtopic, subtopic, topic)
                        print("Subsubtopic: ", subsubtopic)
                        if isinstance(x, int):
                            question = self.make_question(x, exam_topic, subsubtopic_info, subsubtopic, question_content, parent_description)
                            new_question_content = question + f" [{x} marks]\n\n"
                            question_content += new_question_content
                            print(f"{letters[z]}) " + new_question_content)
                        else:
                            if z != 0:
                                child_parent_description = self.make_parent_description(subsubtopic, exam_topic, question_content, subsubtopic_info, parent_description)
                                new_question_content = child_parent_description + "\n\n"
                                subsubtopic_info = subtopic_info
                            else:
                                child_parent_description = parent_description
                                new_question_content = "\n\n"
                            question_content += new_question_content
                            print(f"{letters[z]}) " + new_question_content)
                            y = -1
                            for a in x:
                                y += 1
                                description = parent_description + "\n" + child_parent_description
                                question = self.make_question(a, exam_topic, subsubtopic_info, subsubtopic, question_content, description)
                                new_question_content = question + f" [{a} marks]\n\n"
                                question_content += new_question_content
                                print(f"{romans[y]}) " + new_question_content)

    def create_ms(self, question, marks, topic):
        random_mark_schemes = self.get_random_objects(marks, topic, self.mark_schemes, int(self.subject_info["example_ms"]), question)
        structure = self.test_llm(self.prompts["create_ms_structure"].format(subject=self.subject, random_mark_schemes=random_mark_schemes))
        info = self.test_qa("spec", self.queries["get_question_related_info"].format(question=question))
        print("INFO")
        print(info)
        ms = self.test_llm(self.prompts["create_new_markscheme"].format(
            question=question,
            subject=self.subject,
            structure=structure,
            info=info,
            marks=marks
        ))
        return ms


    def mark_answer(self, answer, mark_scheme, question, marks):
        command_word = self.test_llm(self.prompts["get_command_word"].format(question=question))
        advice = self.test_qa("ms", f"""Give marking advice for {marks} mark '{command_word}' questions""")
        prompt = self.prompts["format_mark_scheme"].format(
            mark_scheme=mark_scheme
        )
        mark_scheme = self.test_llm(prompt)
        prompt = self.prompts["mark_answer"].format(
            subject=self.subject,
            mark_scheme=mark_scheme,
            answer=answer,
            marks=marks,
        )
        return self.test_llm(prompt)



    def load_ms(self):
        dbm = DatabaseManager(self.subject, self.examiner)
        dbm.add_folder_database(f"user_data/user_ms", f"user_data/ms_vdb")

    def get_marks(self, question):
        prompt = self.prompts["extract_marks"].format(
            question=question
        )
        return int(self.test_llm(prompt))

    def exam_type_of_question(self, question):
        exam_types = []
        for example_question in self.questions:
            if example_question["topic"] not in exam_types:
                exam_types.append(example_question["topic"])
        exam_type = self.test_qa("spec", self.prompts["exam_type_of_question"].format(exam_types=exam_types, question=question))
        return exam_type

    def mark_question(self):
        question = {"question": "", "answer": "", "ms": ""}
        for type in question.keys():
            directory = f"user_data/{type}"
            for f in os.listdir(directory):
                ext = os.path.splitext(f)[1].lower()
                if ext in [".jpg", ".png", ".jpeg"]:
                    text = self.image_to_text(directory+"/"+f)
                elif ext in [".txt", ".md"]:
                    with open(directory+"/"+f, "r") as file:
                        text = file.read()
                else:
                    print(f"Invalid {type} file format")
                question[type] += text
        if question["question"] == "":
            print("No question")
        elif question["answer"] == "":
            print("No answer")
        else:
            exam_type = self.exam_type_of_question(question["question"])
            marks = self.get_marks(question["question"])
            if question["ms"] == "":
                question["ms"] = self.create_ms(question["question"], marks, exam_type)
            model_answer = self.test_llm(self.prompts["model_answer"].format(
                ms=question["ms"],
                question=question["question"]
            ))
            print("MARK SCHEME")
            print(question["ms"])
            print("\n\nMODEL ANSWER")
            print(model_answer)
            print("\n\n")
            return self.mark_answer(question["answer"], question["ms"], question["question"], marks)


    def mark_exam(self, questions, topic):
        directory = "user_data/ms"
        file_count = len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
        print(file_count)
        if file_count != 0:
            ms_vdb = FAISS.load_local(f"user_data/ms_vdb", self.embedding_model)
            ms_retriever = ms_vdb.as_retriever(search_type="similarity", search_kwargs={"k": int(self.subject_info["ms_search_kwargs_k"]), "filter": {"type": "MarkScheme"}})
            ms_qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=ms_retriever)
        else:
            ms_qa_chain = self.ms_qa_chain
        for question in questions:
            ms = self.create_ms(question["parent_description"] + "\n" + question["question"], question["marks"], topic)
            mark = self.mark_answer(question["answer"], ms, question["marks"], question["question"])
            print("\n\n\n\nMARK SCHEME")
            print("-------------")
            print(ms)
            print("\nSTUDENT ANSWER")
            print("-------------")
            print(question["answer"])
            print("\nRESULT")
            print("-------------")
            print(mark)

    def image_to_text(self, img_path):
        import base64
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        prompt = self.prompts["read_user_exam_page"]

        msg = HumanMessage(content=[
            {"type": "text", "text": prompt },
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
        ])
        content = self.image_llm.invoke([msg]).content
        print("Content:", content)
        return content


rs_aqa_assistant = GcseAssistantMakeExams("Biology", "Edexcel")

marks = 48
exam_topic = "Higher"
topics = ["Topic 6 – Plant structures and their functions", "Topic 5 – Health, disease and the development of medicines "]

user_input = ""
rs_aqa_assistant.make_exam(exam_topic, user_input, marks, topics)
# print(rs_aqa_assistant.mark_question())



