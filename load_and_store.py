import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import re
from langchain.schema import Document


class PdfFile:
    def __init__(self, name):
        self.name = name
        self.meta_data = self.get_metadata()
        self.info = self.meta_data
        self.questions = self.load_questions()
        self.splitter = CharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=2000,
            separator=""
        )

    def load_questions(self):
        try:
            with open("questions.pkl", "rb") as file: return pickle.load(file)
        except: return []

    def extract_marks(self, text: str) -> int | None:
        pattern = r"""
            [\[\(]          # opening [ or (
            \s*             # optional spaces
            (\d{1,2})       # 1–2 digit number (capture group 1)
            \s*             # optional spaces
            (marks?)?       # optional word "mark"/"marks"
            \s*             # optional spaces
            [\]\)]?         # optional closing ] or )
        """
        m = re.search(pattern, text, flags=re.I | re.X)
        return int(m.group(1)) if m else None

    def add_metadata(self, chunks):
        for doc in chunks:
            doc.metadata.update(self.meta_data)
        return chunks

    def load_pdf(self):
        loader = PyPDFLoader(self.name)
        document = loader.load()
        print(f"{self.name} loaded. \n")
        return document

    def store_questions(self, content):
        for question_content in re.split(r'(?=(?:\s*\d){2}\s*\.\s*\d\b)', content):
            marks = self.extract_marks(question_content)
            question_info = self.info.copy()
            question_info["marks"] = marks
            question_info["content"] = question_content
            print(question_info)
            if marks is not None:
                self.questions.append(question_info)
        with open('questions.pkl', 'wb') as file: pickle.dump(self.questions, file)

    def split_document(self, document):
        if self.meta_data["type"] == "QuestionPaper":
            full_pdf_text = "\n\n".join([doc.page_content for doc in document])
            self.store_questions(full_pdf_text)
        else:
            merged_text = "\n".join([doc.page_content for doc in document])
            merged_doc = Document(page_content=merged_text)

            chunks = self.splitter.split_documents([merged_doc])
            print(f"Document split into chunks.")
            print(f"Chunk one:\n{chunks[0]}\n")
            return chunks

    def get_metadata(self):
        keys = ["subject", "examiner", "type", "topic", "time"]
        details = self.name.split("/")[-1].replace(".PDF", "").split("-")
        print(details)
        meta_data = {}
        for value in details:
            meta_data[keys[details.index(value)]] = value
        return meta_data



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

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):

                print("Now working on:", file_path)
                pdf_file = PdfFile(file_path)
                self.store_to_database(pdf_file, vdb)


dbm = DatabaseManager()
dbm.add_folder_database("data/rs/aqa/themes/questionPapers", "data/rs/aqa/ReligiousStudies-AQA-vectorDatabase")


