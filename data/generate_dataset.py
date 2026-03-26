"""
Dataset Generator for Hate Speech Detection
Generates 52,000 labeled text samples (hate / non-hate)
"""

import csv
import random
import os

random.seed(42)

HATE_TEMPLATES = [
    "{group} are all {negative_adj} and should be {negative_action}",
    "I hate every single {group_member}, they are {negative_adj}",
    "All {group} should {negative_action}, they're {negative_adj} {noun}",
    "These {negative_adj} {group} are ruining everything for us",
    "{group} don't deserve {positive_thing}, they are {negative_adj}",
    "Get rid of all {group}, they're nothing but {negative_adj} {noun}",
    "Why do we let {group} exist? They are so {negative_adj}",
    "No {group_member} can be trusted, they are all {negative_adj} {noun}",
    "Death to all {group}, {negative_adj} {noun} every one of them",
    "We need to eliminate all {group} from our {place}",
    "{group} make me sick, {negative_adj} {noun} all of them",
    "Every {group_member} I've met is a {negative_adj} {noun}",
    "The world would be better without {group}, they are {negative_adj}",
    "Kick out every {group_member}, they don't belong in our {place}",
    "All {group} are subhuman {negative_adj} {noun}",
    "{group} are a plague on {place}, all {negative_adj}",
    "I wish all {group} would just disappear, {negative_adj} {noun}",
    "Ban every {group_member} from our {place}, they're {negative_adj}",
    "Pure hatred for {group}, they are the worst {noun} alive",
    "Every {group_member} is a threat to our {place}",
    "{group} are inferior and {negative_adj}, everyone knows it",
    "Can't stand {group}, they are disgusting {negative_adj} {noun}",
    "Round up all the {group} and {negative_action}",
    "{group_member} spotted in {place}? Disgusting {negative_adj} {noun}",
    "Someone needs to teach {group} a lesson, {negative_adj} creatures",
]

NON_HATE_TEMPLATES = [
    "I had a wonderful day at the {place} today with my friends",
    "Just finished reading an amazing book about {topic}",
    "The weather in {city} is absolutely beautiful this morning",
    "Can anyone recommend a good {food} restaurant near {place}?",
    "Happy birthday to my best friend! Wishing you all the {positive_thing}",
    "Learning about {topic} has been really fascinating lately",
    "Great game last night! The {team} played incredibly well",
    "Just adopted a new {pet} and I couldn't be happier",
    "The sunset at {place} was breathtaking this evening",
    "Started a new {hobby} class and I'm loving every minute",
    "Congratulations to everyone who graduated this year!",
    "This {food} recipe is absolutely delicious, highly recommend",
    "Had the most productive meeting about {topic} today",
    "My {family_member} just got promoted, so proud of them",
    "The new {place} in town is really well designed",
    "Just finished a 5K run, feeling great about my progress",
    "Watching a documentary about {topic}, really eye-opening",
    "Thank you all for the support during this difficult time",
    "The {team} community is so welcoming and friendly",
    "Beautiful flowers are blooming in the {place} this spring",
    "Our neighborhood cleanup was a huge success today",
    "Just saw the most incredible art exhibit at the {place}",
    "Helping my {family_member} move into their new apartment today",
    "The new policy on {topic} seems really promising for everyone",
    "Made homemade {food} for dinner and it turned out perfect",
    "I appreciate everyone who volunteered at the {place} today",
    "Music festival this weekend was absolutely amazing",
    "Finally finished renovating the {place}, it looks fantastic",
    "Just had the best coffee at the new {place} downtown",
    "Our team collaboration on {topic} has been really productive",
    "Enjoying a peaceful morning walk through the {place}",
    "The children's program at the {place} is wonderful this year",
    "Such a kind gesture from my neighbor, really made my day",
    "Looking forward to the community gathering at {place} tomorrow",
    "The new {topic} course is well structured and informative",
]

GROUPS = ["immigrants", "foreigners", "refugees", "outsiders", "migrants",
          "those people", "them", "invaders", "illegals", "newcomers"]
GROUP_MEMBERS = ["immigrant", "foreigner", "refugee", "outsider", "migrant",
                 "one of them", "invader", "illegal", "newcomer", "alien"]
NEGATIVE_ADJ = ["disgusting", "filthy", "worthless", "vile", "pathetic",
                "horrible", "revolting", "despicable", "wretched", "abhorrent",
                "loathsome", "repulsive", "contemptible", "detestable", "deplorable"]
NEGATIVE_ACTIONS = ["be removed", "be banned", "be deported", "leave",
                    "be punished", "be silenced", "disappear", "be expelled"]
NOUNS = ["creatures", "vermin", "parasites", "scum", "trash",
         "waste", "filth", "plague", "pests", "garbage"]
POSITIVE_THINGS = ["rights", "respect", "happiness", "opportunities", "freedom",
                   "education", "healthcare", "love", "compassion", "dignity"]
PLACES = ["park", "community center", "library", "museum", "garden",
          "neighborhood", "city", "town", "country", "school"]
TOPICS = ["artificial intelligence", "climate change", "space exploration",
          "history", "technology", "renewable energy", "marine biology",
          "architecture", "philosophy", "nutrition", "photography",
          "psychology", "economics", "music theory", "robotics"]
CITIES = ["Seattle", "Austin", "Denver", "Portland", "Nashville",
          "San Diego", "Chicago", "Boston", "Atlanta", "Miami"]
FOODS = ["Italian", "Thai", "Mexican", "Japanese", "Indian",
         "Greek", "Korean", "Vietnamese", "French", "Mediterranean"]
TEAMS = ["Lakers", "Warriors", "Eagles", "Packers", "Sox",
         "Yankees", "Celtics", "Dodgers", "Chiefs", "Broncos"]
PETS = ["puppy", "kitten", "rabbit", "hamster", "dog", "cat"]
HOBBIES = ["painting", "cooking", "pottery", "yoga", "dancing",
           "guitar", "photography", "woodworking", "gardening", "chess"]
FAMILY_MEMBERS = ["sister", "brother", "mom", "dad", "cousin",
                  "aunt", "uncle", "grandmother", "grandfather", "friend"]


def fill_hate_template(template):
    return template.format(
        group=random.choice(GROUPS),
        group_member=random.choice(GROUP_MEMBERS),
        negative_adj=random.choice(NEGATIVE_ADJ),
        negative_action=random.choice(NEGATIVE_ACTIONS),
        noun=random.choice(NOUNS),
        positive_thing=random.choice(POSITIVE_THINGS),
        place=random.choice(PLACES),
    )


def fill_non_hate_template(template):
    return template.format(
        place=random.choice(PLACES),
        topic=random.choice(TOPICS),
        city=random.choice(CITIES),
        food=random.choice(FOODS),
        team=random.choice(TEAMS),
        pet=random.choice(PETS),
        hobby=random.choice(HOBBIES),
        family_member=random.choice(FAMILY_MEMBERS),
        positive_thing=random.choice(POSITIVE_THINGS),
    )


def add_noise(text):
    if random.random() < 0.15:
        text = text.upper()
    if random.random() < 0.20:
        text = text + random.choice(["!!!", "!!", "...", "???",
                                      " smh", " tbh", " fr fr", " no cap"])
    if random.random() < 0.12:
        text = text + " " + random.choice(["#truth", "#wakeup", "#facts",
                                            "#blessed", "#grateful", "#love",
                                            "#happy", "#mood", "#vibes", "#real"])
    return text


def generate_dataset(n_samples=52000, output_path="data/dataset.csv"):
    n_hate = n_samples // 2
    n_clean = n_samples - n_hate
    rows = []

    for _ in range(n_hate):
        template = random.choice(HATE_TEMPLATES)
        text = fill_hate_template(template)
        text = add_noise(text)
        rows.append({"text": text, "label": 1})

    for _ in range(n_clean):
        template = random.choice(NON_HATE_TEMPLATES)
        text = fill_non_hate_template(template)
        text = add_noise(text)
        rows.append({"text": text, "label": 0})

    random.shuffle(rows)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    hate_count = sum(1 for r in rows if r["label"] == 1)
    clean_count = sum(1 for r in rows if r["label"] == 0)
    print(f"Generated {len(rows)} samples -> {output_path}")
    print(f"  Hate: {hate_count} | Non-hate: {clean_count}")


if __name__ == "__main__":
    generate_dataset()
