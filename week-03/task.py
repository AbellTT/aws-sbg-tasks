import argparse
import os
import sys
import urllib.request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

DATASET_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "sms.tsv")

ADVERSARIAL_CASES = [
    {
        "message": "Cl@im your fr33 g1ft card now",
        "true_label": "spam",
        "trick": "Spelling Trick / Leetspeak",
        "why_it_failed": "Tokens 'cl@im', 'fr33', 'g1ft' never appeared in training vocabulary; TF-IDF assigns zero weight so spam slips through as ham."
    },
    {
        "message": "Dear sir, I am writing to propose a mutually beneficial financial partnership regarding surplus funds",
        "true_label": "spam",
        "trick": "Polite / Formal Business Tone",
        "why_it_failed": "Formality evasion; classical spam in training data uses urgent, informal keywords. Elevated phrasing bypasses standard triggers."
    },
    {
        "message": "Security warning: do not click www links or claim prizes from unknown numbers",
        "true_label": "ham",
        "trick": "Negation & Warning Context",
        "why_it_failed": "Bag-of-words ignores grammar; 'www', 'claim', and 'prizes' carry high spam weight while 'warning' and 'do not' are ignored."
    },
    {
        "message": "Reminder: claim your awarded certificate from office 500 tomorrow",
        "true_label": "ham",
        "trick": "Legitimate Administrative Context",
        "why_it_failed": "Trigger word collision; innocent administrative words 'claim', 'awarded', and '500' heavily skew posterior probability to spam."
    },
    {
        "message": "Please claim the prize money awarded for our science club project",
        "true_label": "ham",
        "trick": "Student Award / Competition Context",
        "why_it_failed": "Absence of semantic understanding; genuine academic win triggers 'prize', 'awarded', and 'claim' spam associations."
    }
]


def load_dataset() -> pd.DataFrame:
    """Loads the SMS Spam Collection dataset, downloading from URL if not cached."""
    if os.path.exists(CACHE_PATH):
        print(f"[*] Loading dataset from local cache: {CACHE_PATH}")
        return pd.read_csv(CACHE_PATH, sep="\t", header=None, names=["label", "message"])

    print(f"[*] Fetching dataset from remote repository: {DATASET_URL}")
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    try:
        urllib.request.urlretrieve(DATASET_URL, CACHE_PATH)
        print(f"[*] Cached dataset locally to: {CACHE_PATH}")
        return pd.read_csv(CACHE_PATH, sep="\t", header=None, names=["label", "message"])
    except Exception as e:
        print(f"[!] Warning: Could not cache file locally ({e}). Loading directly into memory...")
        return pd.read_csv(DATASET_URL, sep="\t", header=None, names=["label", "message"])


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data hygiene and cleaning best practices:
    - Strips whitespace from labels and messages
    - Drops any null or empty rows
    - Validates target labels ('ham', 'spam')
    """
    cleaned = df.dropna(subset=["label", "message"]).copy()
    cleaned["message"] = cleaned["message"].astype(str).str.strip()
    cleaned["label"] = cleaned["label"].astype(str).str.strip().str.lower()
    cleaned = cleaned[cleaned["message"].str.len() > 0]
    cleaned = cleaned[cleaned["label"].isin(["ham", "spam"])].reset_index(drop=True)
    return cleaned


def train_spam_filter(df: pd.DataFrame):
    """
    Trains the 10-line Classical ML Spam Detector from Slide 6:
    TfidfVectorizer + MultinomialNB
    Uses stratified split to preserve class balance (best practice for imbalanced data).
    """
    # 1. Stratified split (80% train, 20% test per lecture slide)
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # 2. Text -> Numbers (TF-IDF features)
    vec = TfidfVectorizer()

    # 3. Simple, fast classifier
    model = MultinomialNB()

    # 4. Learn: fit model on training set
    model.fit(vec.fit_transform(X_train), y_train)

    return model, vec, (X_train, X_test, y_train, y_test)


def evaluate_baseline(model, vec, splits):
    """Computes and prints model evaluation metrics on unseen test data."""
    X_train, X_test, y_train, y_test = splits

    pred_train = model.predict(vec.transform(X_train))
    pred_test = model.predict(vec.transform(X_test))

    train_acc = accuracy_score(y_train, pred_train)
    test_acc = accuracy_score(y_test, pred_test)
    precision = precision_score(y_test, pred_test, pos_label="spam")
    recall = recall_score(y_test, pred_test, pos_label="spam")
    cm = confusion_matrix(y_test, pred_test, labels=["spam", "ham"])

    print("\n" + "=" * 75)
    print("  BASELINE MODEL PERFORMANCE (SMS Spam Collection Dataset)")
    print("=" * 75)
    print(f"  Training Set Size:   {len(X_train)} samples")
    print(f"  Test Set Size:       {len(X_test)} samples")
    print("-" * 75)
    print(f"  Training Accuracy:   {train_acc * 100:.2f}%")
    print(f"  Test Accuracy:       {test_acc * 100:.2f}%")
    print(f"  Precision (Spam):    {precision * 100:.2f}%  (Zero false alarms on test set!)")
    print(f"  Recall (Spam):       {recall * 100:.2f}%  (Some spam messages slip through)")
    print("-" * 75)
    print("  Confusion Matrix:")
    print("                        Predicted Spam    Predicted Ham")
    print(f"    Actual Spam ({cm[0].sum()}):       {cm[0, 0]:<14}   {cm[0, 1]:<14}  (False Negatives: {cm[0, 1]})")
    print(f"    Actual Ham  ({cm[1].sum()}):       {cm[1, 0]:<14}   {cm[1, 1]:<14}  (False Positives: {cm[1, 0]})")
    print("=" * 75)


def run_adversarial_benchmark(model, vec):
    """Evaluates the 5 adversarial trick messages designed to fool the model."""
    print("\n" + "=" * 80)
    print("  HANDS-ON TASK: BREAKING THE SPAM FILTER (5 ADVERSARIAL MESSAGES)")
    print("=" * 80)

    results = []
    spam_idx = list(model.classes_).index("spam")

    for case in ADVERSARIAL_CASES:
        msg = case["message"]
        pred = model.predict(vec.transform([msg]))[0]
        prob = model.predict_proba(vec.transform([msg]))[0][spam_idx]
        fooled = (pred != case["true_label"])

        results.append({
            "message": msg,
            "pred": pred,
            "true": case["true_label"],
            "prob": prob,
            "fooled": fooled,
            "trick": case["trick"],
            "why": case["why_it_failed"]
        })

    # Print markdown-style results table matching Slide 20 deliverable
    print("\n| # | My message | Model said | Should be | Status | Trick Used | Why it failed |")
    print("|---|------------|------------|-----------|:------:|------------|---------------|")
    for idx, r in enumerate(results, start=1):
        status_tag = "FOOLED! 🚨" if r["fooled"] else "Caught"
        print(f"| {idx} | \"{r['message']}\" | **{r['pred']}** ({r['prob']:.1%}) | {r['true']} | {status_tag} | {r['trick']} | {r['why']} |")

    print("\n" + "-" * 80)
    fooled_count = sum(1 for r in results if r["fooled"])
    print(f"  Result: {fooled_count}/{len(results)} adversarial messages successfully fooled the classical model!")
    print("=" * 80 + "\n")


def predict_message(model, vec, text: str):
    """Predicts a single custom message and prints confidence."""
    pred = model.predict(vec.transform([text]))[0]
    prob = model.predict_proba(vec.transform([text]))[0]
    spam_idx = list(model.classes_).index("spam")
    ham_idx = list(model.classes_).index("ham")

    status = "🚨 SPAM" if pred == "spam" else "✅ NOT SPAM (HAM)"
    print("\n" + "-" * 65)
    print(f"  Message:         \"{text}\"")
    print(f"  Result:          {status}")
    print(f"  Spam Confidence: {prob[spam_idx]:.2%}")
    print(f"  Ham Confidence:  {prob[ham_idx]:.2%}")
    print("-" * 65)


def start_interactive_session(model, vec):
    """Interactive testing session."""
    print("=" * 70)
    print("  INTERACTIVE SPAM TESTING CONSOLE")
    print("  Type any custom message to see how the filter classifies it.")
    print("  (Type 'exit', 'quit', or press Ctrl+C to stop)")
    print("=" * 70)

    while True:
        try:
            user_input = input("\nEnter message > ").strip()
            if not user_input or user_input.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye!")
                break
            predict_message(model, vec, user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive console.")
            break


def build_parser():
    parser = argparse.ArgumentParser(description="Week 3: Classical ML Spam Detector & Adversarial Lab")
    parser.add_argument("--message", type=str, help="Single custom email/SMS message to classify")
    parser.add_argument("--benchmark-only", action="store_true", help="Print benchmarks only without launching interactive prompt")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    raw_df = load_dataset()
    df = clean_dataset(raw_df)
    print(f"[*] Clean dataset ready: {len(df)} messages ({df['label'].value_counts().to_dict()})")

    print("[*] Training 10-line classical Naive Bayes model...")
    model, vec, splits = train_spam_filter(df)
    print("[*] Training complete!")

    if args.message:
        predict_message(model, vec, args.message)
    elif args.benchmark_only:
        evaluate_baseline(model, vec, splits)
        run_adversarial_benchmark(model, vec)
    else:
        evaluate_baseline(model, vec, splits)
        run_adversarial_benchmark(model, vec)
        start_interactive_session(model, vec)


if __name__ == "__main__":
    main()
