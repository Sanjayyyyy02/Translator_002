"""Evaluate completed translation examples and summarize scores by language."""
import csv
from collections import defaultdict

from app import LANGUAGES, bleu_score, chrf_score, translate


INPUT_FILE = "eval_data.csv"
OUTPUT_FILE = "eval_results.csv"


def evaluate_file():
    """Translate rows with real references and save average scores per language."""
    scores = defaultdict(lambda: {"bleu": [], "chrf": []})

    with open(INPUT_FILE, newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            language = (row.get("language") or "").strip()
            english = (row.get("english") or "").strip()
            reference = (row.get("reference") or "").strip()

            # Empty references and marked examples are templates, not evaluation data.
            if not english or not reference or "[placeholder" in english.lower():
                continue
            if "[placeholder" in reference.lower():
                continue
            if language not in LANGUAGES or language == "English":
                print(f"Skipping non-Indic or unsupported target language: {language}")
                continue

            translation, _confidence, _back_text, _roundtrip_score = translate(
                english, LANGUAGES["English"], LANGUAGES[language]
            )
            scores[language]["bleu"].append(bleu_score(reference, translation))
            scores[language]["chrf"].append(chrf_score(reference, translation))

    results = []
    for language, values in sorted(scores.items()):
        count = len(values["bleu"])
        results.append([
            language,
            count,
            sum(values["bleu"]) / count,
            sum(values["chrf"]) / count,
        ])

    # Write a header even when the CSV still contains only placeholders.
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["language", "examples", "average_bleu", "average_chrf"])
        writer.writerows(results)

    if not results:
        print("No completed evaluation rows. Fill in English text and reference translations.")
        print(f"An empty results table was saved to {OUTPUT_FILE}.")
        return

    print(f"{'Language':<15} {'Examples':>8} {'Average BLEU':>14} {'Average chrF':>14}")
    for language, count, average_bleu, average_chrf in results:
        print(f"{language:<15} {count:>8} {average_bleu:>14.2f} {average_chrf:>14.2f}")
    print(f"Results saved to {OUTPUT_FILE}.")


if __name__ == "__main__":
    evaluate_file()
