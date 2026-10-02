import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


# =====================================================
# DATASET
# =====================================================

data = {

    "text": [

        # -------------------------
        # NORMAL
        # -------------------------

        "I really enjoyed this movie",
        "This is a beautiful day",
        "Thank you for your help",
        "I like this project",
        "This lesson is very useful",
        "This is very good",
        "You are a good person",
        "You are doing a great job",
        "Very good boy",
        "You are very kind",
        "Good work",
        "Excellent work",
        "Well done",
        "I like your work",
        "Thank you very much",
        "You did a great job",
        "That was a wonderful idea",
        "I am happy with your work",
        "Your project looks great",
        "This is a nice project",
        "You are helpful",
        "You are doing well",
        "Great job",
        "Good morning",
        "Have a nice day",
        "I appreciate your help",
        "That is a good idea",
        "You explained it very well",
        "I am proud of your work",
        "This is really interesting",
        "Your answer is correct",
        "I enjoyed your presentation",
        "You worked very hard",
        "Keep up the good work",
        "That was helpful",
        "Nice explanation",
        "I agree with your idea",
        "Your work is impressive",
        "This is a useful application",
        "The project is working well",

        # -------------------------
        # CYBERBULLYING
        # -------------------------

        "You are stupid",
        "Nobody likes you",
        "You are useless",
        "You are an idiot",
        "Stop bothering everyone",
        "You are so dumb",
        "You cannot do anything",
        "Everyone is laughing at you",
        "You are a loser",
        "You are annoying",
        "Nobody wants you here",
        "You are terrible at everything",
        "You always make mistakes",
        "You should be ashamed",
        "You are embarrassing",
        "Everyone hates you",
        "You are such a fool",
        "You are worthless",
        "Stop talking",
        "Go away",
        "You are bad at this",
        "Nobody wants to hear you",
        "You are pathetic",
        "You are making everyone angry",
        "You are a complete failure",
        "People are laughing at you",
        "You should leave",
        "You are not good enough",
        "You are always causing problems",
        "Nobody respects you",
        "You are really annoying",
        "You cannot do anything right",
        "Everyone thinks you are stupid",
        "You are a bad student",
        "You are embarrassing yourself",
        "Nobody wants to work with you",
        "You are always wrong",
        "You are a failure",
        "You should stop trying",
        "Everyone is tired of you",

        # -------------------------
        # HATE SPEECH
        # -------------------------

        "I hate people from that group",
        "Those people are disgusting",
        "That group should not be here",
        "I hate those people",
        "They are all terrible",
        "People from that group are bad",
        "I do not like that community",
        "That group is dangerous",
        "Those people are not welcome",
        "I hate that community",
        "That group should leave",
        "They are all bad people",
        "I dislike people from that group",
        "That community is terrible",
        "Those people should stay away",
        "I hate everyone from that group",
        "That group is the problem",
        "Those people are unwanted",
        "I do not respect that group",
        "They should not be allowed here",
        "That community is disgusting",
        "I hate people from that community",
        "That group is not welcome",
        "I do not want those people here",
        "Those people are dangerous",
        "That community should leave",
        "I strongly dislike that group",
        "Those people should not come here",
        "I hate that group",
        "People from that community are bad",
        "That group does not belong here",
        "Those people are not wanted",
        "I dislike that group",
        "That community should not exist here",
        "I do not want that group here",
        "Those people are dangerous to society",
        "That group is unwanted",
        "I hate members of that group",
        "Those people should go away",
        "That community is not welcome"
    ],

    "label": [

        # NORMAL - 40
        "Normal","Normal","Normal","Normal","Normal",
        "Normal","Normal","Normal","Normal","Normal",
        "Normal","Normal","Normal","Normal","Normal",
        "Normal","Normal","Normal","Normal","Normal",
        "Normal","Normal","Normal","Normal","Normal",
        "Normal","Normal","Normal","Normal","Normal",
        "Normal","Normal","Normal","Normal","Normal",
        "Normal","Normal","Normal","Normal","Normal",

        # CYBERBULLYING - 40
        "Cyberbullying","Cyberbullying","Cyberbullying","Cyberbullying",
        "Cyberbullying","Cyberbullying","Cyberbullying","Cyberbullying",
        "Cyberbullying","Cyberbullying","Cyberbullying","Cyberbullying",
        "Cyberbullying","Cyberbullying","Cyberbullying","Cyberbullying",
        "Cyberbullying","Cyberbullying","Cyberbullying","Cyberbullying",
        "Cyberbullying","Cyberbullying","Cyberbullying","Cyberbullying",
        "Cyberbullying","Cyberbullying","Cyberbullying","Cyberbullying",
        "Cyberbullying","Cyberbullying","Cyberbullying","Cyberbullying",
        "Cyberbullying","Cyberbullying","Cyberbullying","Cyberbullying",
        "Cyberbullying","Cyberbullying","Cyberbullying","Cyberbullying",

        # HATE SPEECH - 40
        "Hate Speech","Hate Speech","Hate Speech","Hate Speech",
        "Hate Speech","Hate Speech","Hate Speech","Hate Speech",
        "Hate Speech","Hate Speech","Hate Speech","Hate Speech",
        "Hate Speech","Hate Speech","Hate Speech","Hate Speech",
        "Hate Speech","Hate Speech","Hate Speech","Hate Speech",
        "Hate Speech","Hate Speech","Hate Speech","Hate Speech",
        "Hate Speech","Hate Speech","Hate Speech","Hate Speech",
        "Hate Speech","Hate Speech","Hate Speech","Hate Speech",
        "Hate Speech","Hate Speech","Hate Speech","Hate Speech",
        "Hate Speech","Hate Speech","Hate Speech","Hate Speech"
    ]
}


# =====================================================
# CREATE DATAFRAME
# =====================================================

df = pd.DataFrame(data)

print("======================================")
print("Cyberbullying & Hate Speech Detection")
print("======================================")

print("\nTotal samples:", len(df))

print("\nClass distribution:")
print(df["label"].value_counts())


# =====================================================
# TF-IDF FEATURES
# =====================================================

word_features = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True
)

character_features = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=1,
    sublinear_tf=True
)


# =====================================================
# COMBINE WORD + CHARACTER FEATURES
# =====================================================

features = FeatureUnion([
    ("word_tfidf", word_features),
    ("character_tfidf", character_features)
])


# =====================================================
# MACHINE LEARNING MODEL
# =====================================================

model = Pipeline([

    ("features", features),

    (
        "classifier",
        LogisticRegression(
            max_iter=3000,
            C=20,
            class_weight="balanced",
            random_state=42
        )
    )
])


# =====================================================
# TRAIN MODEL
# =====================================================

print("\nTraining model...")

model.fit(
    df["text"],
    df["label"]
)


# =====================================================
# SAVE MODEL
# =====================================================

os.makedirs("model", exist_ok=True)

model_path = "model/cyberbullying_hate_speech_model.pkl"

joblib.dump(
    model,
    model_path
)


print("\n======================================")
print("Training completed successfully!")
print("======================================")

print("\nModel saved at:")
print(model_path)


# =====================================================
# TEST SOME EXAMPLES
# =====================================================

test_texts = [
    "You are doing a great job",
    "Very good boy",
    "You are stupid",
    "Nobody likes you",
    "I hate people from that group",
    "That group is not welcome here"
]

predictions = model.predict(test_texts)

print("\nSample predictions:")

for text, prediction in zip(test_texts, predictions):
    print("--------------------------------------")
    print("Text:", text)
    print("Prediction:", prediction)