import os

from huggingface_hub import InferenceClient
from file_loader import load_pdf
from generate_qa import generate_qa
from environment import OUTPUT_FOLDER


def create_output_folder(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


def get_filename_from_path(file_path, extension):
    return os.path.basename(file_path.split(extension)[0])


def main():
    file_path = "/Users/raphaelfeigl/Documents/Studium Master/Netzwerk Sicherheit/script/03_Firewall_Middleboxes_ho.pdf"
    file_name = get_filename_from_path(file_path, extension=".pdf")
    docs = load_pdf(file_path)
    qa_couples = generate_qa(docs, False)
    create_output_folder(OUTPUT_FOLDER)
    qa_couples.to_csv(f"output/{file_name}_qa.csv")


if __name__ == '__main__':
    main()
