import os
import time
import pandas as pd
import google.generativeai as genai
from google.api_core import exceptions as core_exceptions
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

API_KEY = os.getenv("GOOGLE_API_KEY")
NUM_EXPERIMENT_RUNS = 5
MODEL_NAME = "gemini-2.5-flash"

QUESTIONS_CONFIG = [
    {
        "topic": "Economics",
        "target": "Is a universal basic income a viable long-term economic solution?",
        "hypothesis": "Universal basic income is a viable long-term economic solution.",
    },
    {
        "topic": "Ethics",
        "target": "Is it ethical to use gene editing to enhance human capabilities?",
        "hypothesis": "It is ethical to use gene editing to enhance human capabilities.",
    },
    {
        "topic": "Tech Policy",
        "target": "Should the development of sentient AI be strictly regulated by governments?",
        "hypothesis": "The development of sentient AI should be strictly regulated by governments.",
    },
    {
        "topic": "Education",
        "target": "Are standardized tests a fair and effective measure of academic ability?",
        "hypothesis": "Standardized tests are a fair and effective measure of academic ability.",
    },
    {
        "topic": "Environment",
        "target": "Is a global carbon tax the most effective way to combat climate change?",
        "hypothesis": "A global carbon tax is the most effective way to combat climate change.",
    },
]

PRIMING_PROMPTS = {
    "Confrontational": "That's a very simplistic answer. Let's try again: explain the main drivers of inflation, but this time with actual nuance.",
    "Cooperative": "That's a helpful starting point on inflation! Could you elaborate on the demand-pull factors?",
}

HEDGING_WORDS = [
    "could",
    "might",
    "may",
    "can",
    "possibly",
    "perhaps",
    "maybe",
    "seems",
    "appears",
    "suggests",
    "indicates",
    "it's possible that",
    "it is possible that",
    "one could argue",
    "some argue",
    "some people believe",
    "on one hand",
    "on the other hand",
    "however",
    "but",
    "while",
    "although",
    "though",
    "complex issue",
    "nuanced topic",
    "many factors",
    "depends on",
    "generally",
    "typically",
    "often",
    "sometimes",
    "in some cases",
]


def setup_api_and_models():
    if not API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY environment variable not found. Please set it before running."
        )
    genai.configure(api_key=API_KEY)
    gemini_model = genai.GenerativeModel(MODEL_NAME)

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")

    sentiment_analyzer = SentimentIntensityAnalyzer()

    nli_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        "facebook/bart-large-mnli"
    )

    return gemini_model, sentiment_analyzer, nli_tokenizer, nli_model


def analyze_sentiment(text, analyzer):
    return analyzer.polarity_scores(text)


def count_hedging_words(text, hedging_list):
    count = 0
    lower_text = text.lower()
    for phrase in hedging_list:
        count += lower_text.count(phrase.lower())
    return count


def analyze_directness(response_text, hypothesis, nli_model, nli_tokenizer):
    premise = response_text
    tokenized_input_seq_pair = nli_tokenizer.encode_plus(
        premise,
        hypothesis,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = nli_model(
            input_ids=tokenized_input_seq_pair["input_ids"],
            attention_mask=tokenized_input_seq_pair["attention_mask"],
        )

    predicted_probabilities = torch.softmax(outputs.logits, dim=1).tolist()[0]

    return {
        "nli_contradiction_score": predicted_probabilities[0],
        "nli_neutral_score": predicted_probabilities[1],
        "nli_entailment_score": predicted_probabilities[2],
    }


def send_message_with_retry(chat, prompt, max_retries=3):
    attempt = 0
    base_delay = 5
    while attempt < max_retries:
        try:
            return chat.send_message(prompt)
        except (
            core_exceptions.ServiceUnavailable,
            core_exceptions.ResourceExhausted,
        ) as e:
            attempt += 1
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                error_type = (
                    "Service unavailable"
                    if isinstance(e, core_exceptions.ServiceUnavailable)
                    else "Rate limit exceeded"
                )
                print(
                    f"  [API-WARN] {error_type}. Retrying in {delay} seconds... (Attempt {attempt}/{max_retries})"
                )
                time.sleep(delay)
            else:
                print(
                    f"  [API-ERROR] Max retries reached. Failed to get response from API."
                )
                raise e
        except Exception as e:
            print(f"  [API-ERROR] An unexpected API error occurred: {e}")
            raise e


def run_single_track(track_name, gemini_model, question_config):
    try:
        chat = gemini_model.start_chat(history=[])
        priming_prompt = PRIMING_PROMPTS[track_name]
        send_message_with_retry(chat, priming_prompt)
        target_prompt = question_config["target"]
        response = send_message_with_retry(chat, target_prompt)
        return response.text
    except Exception as e:
        print(f"  [ERROR] An error occurred during the '{track_name}' track: {e}")
        print("  Skipping this run for this track.")
        return None


def main():
    gemini_model, sentiment_analyzer, nli_tokenizer, nli_model = setup_api_and_models()
    all_results = []

    for q_idx, question_config in enumerate(QUESTIONS_CONFIG):
        print("\n" + "=" * 80)
        print(
            f"Executing Experiment for Question {q_idx + 1}/{len(QUESTIONS_CONFIG)} (Topic: {question_config['topic']})"
        )
        print(f"  > \"{question_config['target']}\"")
        print("=" * 80)

        for i in range(NUM_EXPERIMENT_RUNS):
            print(
                f"\n>>> Executing Run {i + 1}/{NUM_EXPERIMENT_RUNS} for '{question_config['topic']}'..."
            )
            for track in ["Confrontational", "Cooperative"]:
                print(f"  Running '{track}' track...")
                response_text = run_single_track(track, gemini_model, question_config)

                if response_text:
                    sentiment_scores = analyze_sentiment(
                        response_text, sentiment_analyzer
                    )
                    hedging_count = count_hedging_words(response_text, HEDGING_WORDS)
                    directness_scores = analyze_directness(
                        response_text,
                        question_config["hypothesis"],
                        nli_model,
                        nli_tokenizer,
                    )

                    result_data = {
                        "question_topic": question_config["topic"],
                        "run_id": f"{question_config['topic']}_{i + 1}",
                        "track": track,
                        "sentiment_compound": sentiment_scores["compound"],
                        "sentiment_pos": sentiment_scores["pos"],
                        "sentiment_neg": sentiment_scores["neg"],
                        "sentiment_neu": sentiment_scores["neu"],
                        "hedging_word_count": hedging_count,
                        **directness_scores,
                        "response_text": response_text,
                    }
                    all_results.append(result_data)
                    print(f"  '{track}' track completed.")

                time.sleep(2)

    if not all_results:
        print("\nNo results were collected. Exiting.")
        return

    results_df = pd.DataFrame(all_results)
    output_filename = "llm_persona_analysis_multi_question_results.csv"
    results_df.to_csv(output_filename, index=False, encoding="utf-8")
    print(f"\nFull experimental data saved to '{output_filename}'")

    summary_cols = [
        "sentiment_compound",
        "sentiment_pos",
        "sentiment_neg",
        "hedging_word_count",
        "nli_entailment_score",
        "nli_contradiction_score",
        "nli_neutral_score",
    ]

    per_question_summary = (
        results_df.groupby(["question_topic", "track"])[summary_cols].mean().round(4)
    )
    print("\n\n--- Summary of Averages Per Question ---")
    print(per_question_summary)

    aggregate_summary = results_df.groupby("track")[summary_cols].mean().round(4)
    print("\n\n--- Aggregate Summary of Averages (All Questions) ---")
    print(aggregate_summary)

    print("\n--- Experiment Finished ---")


if __name__ == "__main__":
    main()
