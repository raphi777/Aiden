import random
import pandas as pd
from tqdm import tqdm
from communicator import call_openai
from prompt import QA_generation_prompt, question_groundedness_critique_prompt, question_relevance_critique_prompt, \
    question_standalone_critique_prompt


def generate_qa(docs, filter_bad=True):
    n_generations = 10

    print(f"Generating {n_generations} QA couples ...")

    outputs = []
    for context in tqdm(random.sample(docs, n_generations)):
        output_qa_couple = call_openai(QA_generation_prompt.format(context=context.page_content))
        try:
            question = output_qa_couple.split("Factoid question: ")[-1].split("Answer: ")[0]
            answer = output_qa_couple.split("Answer: ")[-1]
            assert len(answer) < 500, "Answer is too long"
            outputs.append(
                {
                    "context": context.page_content,
                    "question": question,
                    "answer": answer,
                    "source_doc": context.metadata["source"],
                }
            )
        except Exception as e:
            print("An error occurred during Q&A generation: " + str(e))
            continue

    outputs = generate_qa_critique(outputs)

    if filter_bad:
        return filter_bad_questions(outputs)
    return pd.DataFrame.from_dict(outputs)


def generate_qa_critique(qa_couples):
    print("Generating critique for each QA couple ...")

    for qa_couple in tqdm(qa_couples):
        evaluations = {
            "groundedness": call_openai(
                question_groundedness_critique_prompt.format(context=qa_couple["context"],
                                                             question=qa_couple["question"])
            ),
            "relevance": call_openai(
                question_relevance_critique_prompt.format(question=qa_couple["question"]),
            ),
            "standalone": call_openai(
                question_standalone_critique_prompt.format(question=qa_couple["question"])
            )
        }
        try:
            for criterion, evaluation in evaluations.items():
                score, evalu = (
                    int(evaluation.split("Total rating: ")[-1].strip()),
                    evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1],
                )
                qa_couple.update(
                    {
                        f"{criterion}_score": score,
                        f"{criterion}_eval": evalu,
                    }
                )
        except Exception as e:
            print("An error occurred during Q&A critique generation: " + str(e))
            continue

    return qa_couples


def filter_bad_questions(qa_couples):
    pd.set_option("display.max_colwidth", None)

    generated_questions = pd.DataFrame.from_dict(qa_couples)

    print("Evaluation dataset before filtering:")
    print(
        generated_questions[
            [
                "question",
                "answer",
                "groundedness_score",
                "relevance_score",
                "standalone_score"
            ]
        ]
    )
    generated_questions = generated_questions.loc[
        ((generated_questions["groundedness_score"] >= 4) & (generated_questions["relevance_score"] >= 4) &
         (generated_questions["standalone_score"] >= 4))
    ]
    print("============================================")
    print("Final Q&A dataset")
    print(
        generated_questions[
            [
                "question",
                "answer",
                "groundedness_score",
                "relevance_score",
                "standalone_score",
            ]
        ]
    )
    return generated_questions
