import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import re
from langchain.schema import Document


class PdfFile:
    def __init__(self, name, subject, examiner, type):
        self.subject = subject
        self.examiner = examiner
        self.subject_info = self.get_subject_info()
        self.name = name
        self.meta_data = self.get_metadata()
        self.info = self.meta_data
        self.questions = self.load_questions("questions")
        self.msquestions = self.load_questions("msquestions")
        keys = {
            "Specification": "spec",
            "MarkSchemes": "ms",
            "QuestionPapers": "spec"
        }
        type = keys[type]
        self.splitter = CharacterTextSplitter(
            chunk_size=int(self.subject_info[f"{type}_chunk_size"]),
            chunk_overlap=int(self.subject_info[f"{type}_chunk_overlap"]),
            separator=""
        )
        self.marks_pattern = rf"{self.subject_info['mark_pattern']}"
        self.letter_pattern = rf"{self.subject_info['letter_pattern']}"
        self.roman_pattern = rf"{self.subject_info['roman_pattern']}"
        self.question_pattern = rf"{self.subject_info['question_pattern']}"

    def get_subject_info(self):
        with open("subject_info.txt", "r") as file:
            subject_info_file = file.read().split("\n")

        subject = {}
        subject_info = {}
        for line in subject_info_file:
            info = line.split(" ")
            print(info)
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

    def first_marker_type(self, text):
        letter_match = re.search(self.letter_pattern, text)
        roman_match = re.search(self.roman_pattern, text)
        letter_pos = letter_match.start() if letter_match else None
        roman_pos = roman_match.start() if roman_match else None

        if letter_pos is None and roman_pos is None:
            return None
        elif letter_pos is None:
            return "roman"
        elif roman_pos is None:
            return "letter"
        else:
            return "letter" if letter_pos < roman_pos else "roman"

    def load_questions(self, type):
        try:
            with open(f"data/{self.subject}/{self.examiner}/{type}.pkl", "rb") as file: return pickle.load(file)
        except: return []

    def extract_mark(self, text, pattern):
        regex = re.compile(pattern)
        matches = list(regex.finditer(text))
        if matches:
            last_match = matches[-1]
            return int(last_match.group(1)) if last_match.groups() else int(last_match.group())
        return None

    def add_metadata(self, chunks):
        for doc in chunks:
            doc.metadata.update(self.meta_data)
        return chunks

    def load_pdf(self):
        loader = PyPDFLoader(self.name)
        document = loader.load()
        print(f"{self.name} loaded. \n")
        return document

    def pdf_to_text(self, pages):
        return " ".join(pages[i+1].page_content for i in range(len(pages)-1))

    def flatten(self, lst):
        result = []
        for item in lst:
            if isinstance(item, list):
                for a in item:
                    result.append(a)
            else:
                result.append(item)
        return result

    def is_parent_question_valid(self, parent_question, questions):
        structure = self.flatten(parent_question["parent_question_structure"])
        x = []
        for question in questions:
            if question["marks"] is not None and question["parent_question_description"] == parent_question["parent_question_description"]:
                x.append(question["marks"])
        if structure == x:
            return True
        else:
            return False

    def store_questions(self, content):
        topic = self.meta_data["topic"]
        exam = self.meta_data["time"]
        questions = re.split(self.question_pattern, content)
        question_info = self.process_questions(questions, topic, exam)
        for question in question_info:
            parent_valid = False
            if question["type"] == "parent_question":
                parent_valid = self.is_parent_question_valid(question, question_info)
            if question["marks"] is not None or parent_valid:
                self.questions.append(question)
        with open(f"data/{self.subject}/{self.examiner}/questions.pkl", 'wb') as file: pickle.dump(self.questions, file)

    def store_msquestions(self, content):
        mark_schemes = re.split(self.subject_info["ms_pattern"], content)
        mark_schemes_info = []
        for mark_scheme in mark_schemes:
            mark_schemes_info.append({
                "topic": self.meta_data["topic"],
                "question_content": mark_scheme,
                "marks": self.extract_mark(mark_scheme, self.subject_info["ms_mark_pattern"]),
                "exam": self.meta_data["time"]
            })
        self.msquestions += mark_schemes_info
        with open(f"data/{self.subject}/{self.examiner}/msquestions.pkl", 'wb') as file: pickle.dump(self.msquestions, file)

    def split_document(self, document):
        if self.meta_data["type"] == "QuestionPaper":
            full_pdf_text = self.pdf_to_text(document)
            self.store_questions(full_pdf_text)
        elif self.meta_data["type"] == "MarkScheme":
            full_pdf_text = self.pdf_to_text(document)
            self.store_msquestions(full_pdf_text)
        else:
            merged_text = "\n".join([str(doc.page_content) for doc in document])
            merged_doc = Document(page_content=merged_text)

            chunks = self.splitter.split_documents([merged_doc])
            print(f"Document split into chunks.")
            print(f"Chunk one:\n{chunks[0]}\n")
            return chunks

    def get_metadata(self):
        keys = ["subject", "examiner", "type", "topic", "time"]
        details = self.name.split("/")[-1].replace(".PDF", "").replace(".pdf", "").split("-")
        print(details)
        meta_data = {}
        for value in details:
            meta_data[keys[details.index(value)]] = value
        print(meta_data)
        return meta_data

    def process_questions(self, questions, topic, exam):
        questions_info = []
        for question in questions:
            first_marker_type_question = self.first_marker_type(question)
            if first_marker_type_question is not None:
                if first_marker_type_question == "letter":
                    sub_questions = re.split(self.letter_pattern, question)
                else:
                    sub_questions = re.split(self.roman_pattern, question)
                parent_question_description = sub_questions[0]
                sub_questions.pop(0)
                question_structure = []
                for sub_question in sub_questions:
                    first_marker_type_question = self.first_marker_type(sub_question)
                    if first_marker_type_question is None:
                        marks = self.extract_mark(sub_question, self.marks_pattern)
                        question_structure.append(marks)
                        questions_info.append({
                            "type": "child_question",
                            "topic": topic,
                            "marks": marks,
                            "question_content": sub_question,
                            "parent_question_structure": None,
                            "parent_question_description": parent_question_description,
                        })
                    else:
                        if first_marker_type_question == "letter":
                            sub_sub_questions = re.split(self.letter_pattern, sub_question)
                        else:
                            sub_sub_questions = re.split(self.roman_pattern, sub_question)
                        parent_child_question_description = sub_sub_questions[0]
                        sub_sub_questions.pop(0)
                        sub_question_structure = []
                        for sub_sub_question in sub_sub_questions:
                            marks = self.extract_mark(sub_sub_question, self.marks_pattern)
                            sub_question_structure.append(marks)
                            questions_info.append({
                                "type": "grandchild_question",
                                "topic": topic,
                                "marks": marks,
                                "question_content": sub_sub_question,
                                "parent_question_structure": None,
                                "parent_question_description": parent_question_description
                            })
                        questions_info.append({
                            "type": "parent_child_question",
                            "topic": topic,
                            "marks": None,
                            "question_content": None,
                            "parent_question_structure": sub_question_structure,
                            "parent_question_description": parent_question_description
                        })
                        question_structure.append(sub_question_structure)
                questions_info.append({
                    "type": "parent_question",
                    "topic": topic,
                    "marks": None,
                    "question_content": question,
                    "parent_question_structure": question_structure,
                    "parent_question_description": parent_question_description
                })

            else:
                questions_info.append({
                    "type": "basic_question",
                    "topic": topic,
                    "marks": self.extract_mark(question, self.marks_pattern),
                    "question_content": question,
                    "parent_question_structure": None,
                    "parent_question_description": None
                })
        for i in range(len(questions_info)):
            questions_info[i]["exam"] = exam
        return questions_info



class VectorStore:

    def __init__(self, name):
        self.vector_database_name = name
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def embed_and_store_chunks_new_database(self, chunks):
        vectorstore = FAISS.from_documents(chunks, self.embedding_model)
        vectorstore.save_local(self.vector_database_name)
        print(f"Chunks saved to new vector database, {self.vector_database_name}\n")

    def embed_and_store_chunks_old_database(self, chunks):
        db = FAISS.load_local(self.vector_database_name, self.embedding_model)
        db.add_documents(chunks)
        db.save_local(self.vector_database_name)
        print(f"Chunks added to {self.vector_database_name}\n")


class DatabaseManager:
    def __init__(self, subject, examiner):
        self.subject = subject
        self.examiner = examiner
    def store_to_database(self, pdf, database):
        document = pdf.load_pdf()
        chunks = pdf.split_document(document)
        if chunks is not None:
            chunks = pdf.add_metadata(chunks)
            try:
                database.embed_and_store_chunks_old_database(chunks)
            except:
                database.embed_and_store_chunks_new_database(chunks)

    def add_folder_database(self, folder, database):
        vdb = VectorStore(database)
        folder_path = folder
        type = folder.split("/")[-1]
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):

                print("Now working on:", file_path)
                pdf_file = PdfFile(file_path, self.subject, self.examiner, type)
                self.store_to_database(pdf_file, vdb)

# subject = "Biology"
# examiner = "Edexcel"
# dbm = DatabaseManager(subject, examiner)
# dbm.add_folder_database(f"data/Biology/Edexcel/Specification", f"data/{subject}/{examiner}/{subject}-{examiner}-vectorDatabase")
#
#
