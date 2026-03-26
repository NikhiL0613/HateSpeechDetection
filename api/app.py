"""
Flask API - Hate Speech Detection with Extended Threat Detection
ML model + comprehensive word/phrase detection
"""

import os
import json
import time
import re
import pickle
import logging
from datetime import datetime, timezone
from collections import deque
import numpy as np
from flask import Flask, request, jsonify

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
PROCESSED = os.environ.get("PROCESSED_DIR", "data/processed")
PORT = int(os.environ.get("PORT", 5000))
THRESHOLD = 0.75

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp


# ============================================
# COMPREHENSIVE NEGATIVE WORD DICTIONARY
# ============================================
NEGATIVE_WORDS = {
    # === PROFANITY & SLURS ===
    "fuck", "fucking", "fucked", "fucker", "fucks", "fck", "fuk", "fuq",
    "shit", "shitty", "shitting", "bullshit", "horseshit", "dipshit",
    "bitch", "bitches", "bitchy", "bitchin",
    "bastard", "bastards",
    "ass", "asshole", "assholes", "arse", "arsehole",
    "dick", "dicks", "dickhead", "dickwad",
    "cunt", "cunts",
    "damn", "damned", "dammit", "goddamn", "goddammit",
    "crap", "crappy", "crapping",
    "piss", "pissed", "pissing",
    "whore", "whores", "whorish",
    "slut", "sluts", "slutty",
    "hoe", "hoes", "ho",
    "twat", "twats",
    "wanker", "wankers",
    "tosser", "tossers",
    "prick", "pricks",
    "douche", "douchebag", "douches",
    "skank", "skanky", "skanks",
    "tramp", "tramps",
    "slag", "slags",

    # === INSULTS & DEROGATORY ===
    "idiot", "idiots", "idiotic",
    "moron", "morons", "moronic",
    "stupid", "stupidity",
    "dumb", "dumber", "dumbest", "dumbass",
    "retard", "retarded", "retards",
    "imbecile", "imbeciles",
    "fool", "foolish", "fools",
    "loser", "losers",
    "lame", "lamest",
    "pathetic", "pathetically",
    "worthless", "worthlessly",
    "useless", "uselessly",
    "hopeless", "hopelessly",
    "brainless", "braindead",
    "clueless", "cluelessly",
    "incompetent", "incompetence",
    "ignorant", "ignorance", "ignoramus",
    "illiterate",
    "dimwit", "dimwits",
    "nitwit", "nitwits",
    "halfwit", "halfwits",
    "numbskull", "numbnuts",
    "bonehead", "blockhead", "airhead",
    "dunce", "dolt", "doofus", "dork",
    "nerd", "geek", "freak", "freaks", "freaky",
    "creep", "creeps", "creepy", "creeper",
    "weirdo", "weirdos",
    "psycho", "psychos", "psychopath", "sociopath",
    "maniac", "maniacs",
    "lunatic", "lunatics",
    "nutjob", "nutcase", "nutjobs",
    "crackpot", "crackhead",
    "junkie", "junkies",
    "coward", "cowards", "cowardly",
    "weakling", "weaklings",
    "wimp", "wimps", "wimpy",
    "sissy", "sissies",
    "cry baby", "crybaby",
    "snowflake", "snowflakes",
    "Karen",
    "simp", "simps", "simping",
    "incel", "incels",
    "neckbeard", "neckbeards",
    "troll", "trolls", "trolling",
    "clown", "clowns",
    "joke", "jokes",
    "fraud", "frauds",
    "faker", "fakers", "phony", "phonies",
    "liar", "liars", "lying",
    "cheat", "cheater", "cheaters", "cheating",
    "thief", "thieves", "stealing",
    "criminal", "criminals",
    "crook", "crooks",
    "scammer", "scammers", "scamming",
    "con", "conman",
    "corrupt", "corruption",
    "parasite", "parasites",
    "leech", "leeches",
    "vermin", "vermins",
    "pest", "pests",
    "scum", "scumbag", "scumbags",
    "trash", "trashy",
    "garbage", "filth", "filthy",
    "dirt", "dirty", "dirtbag",
    "sleazy", "sleazebag", "sleazeball",
    "slimy", "slimeball",
    "nasty", "nastier", "nastiest",
    "gross", "grosser", "grossest",
    "disgusting", "disgust", "disgusted",
    "revolting", "repulsive", "repugnant",
    "vile", "vilest",
    "foul", "fouler", "foulest",
    "rotten", "rottener",
    "putrid", "rancid", "fetid",
    "horrid", "horrible", "horribly", "horrific", "horrifying",
    "terrible", "terribly", "terrifying",
    "awful", "awfully", "atrocious", "atrociously",
    "dreadful", "dreadfully",
    "appalling", "appallingly",
    "abysmal", "abysmally",
    "lousy", "lousier",
    "crummy", "crummier",
    "abhorrent", "abhorrence",
    "despicable", "despicably",
    "contemptible", "contempt",
    "detestable", "detested",
    "deplorable", "deplorably",
    "loathsome", "loathing",
    "wretched", "wretchedly",
    "miserable", "miserably",
    "abominable", "abominably",

    # === APPEARANCE INSULTS ===
    "ugly", "uglier", "ugliest", "ugliness",
    "fat", "fatty", "fatso", "fatass", "fats",
    "obese", "overweight", "chunky", "chubby",
    "skinny", "anorexic", "skeleton",
    "midget", "dwarf", "runt", "shrimp",
    "bald", "baldy",
    "fugly", "hideous", "grotesque", "repulsive",
    "deformed", "cripple", "crippled",

    # === HATE & DISCRIMINATION ===
    "hate", "hated", "hater", "haters", "hating", "hatred",
    "racist", "racists", "racism", "racial",
    "sexist", "sexists", "sexism",
    "bigot", "bigots", "bigotry", "bigoted",
    "homophobe", "homophobic", "homophobia",
    "transphobe", "transphobic", "transphobia",
    "xenophobe", "xenophobic", "xenophobia",
    "misogynist", "misogyny", "misogynistic",
    "misandrist", "misandry",
    "chauvinist", "chauvinism",
    "supremacist", "supremacy",
    "nazi", "nazis", "fascist", "fascists", "fascism",
    "extremist", "extremists", "extremism",
    "radical", "radicals", "radicalized",
    "terrorist", "terrorists", "terrorism",
    "jihadist", "jihadists",
    "thug", "thugs", "thuggish",
    "savage", "savages", "savagery",
    "barbaric", "barbarian", "barbarians",
    "primitive", "primitives",
    "uncivilized", "uncivilised",
    "inferior", "inferiority",
    "subhuman", "subhumans",
    "inhuman", "inhumane", "inhumanity",
    "animal", "animals", "beast", "beasts",
    "monster", "monsters", "monstrous",
    "demon", "demons", "demonic", "devil", "devilish",
    "evil", "evils", "evilness",
    "wicked", "wickedness",
    "sinful", "sinner", "sinners",
    "degenerate", "degenerates", "degeneracy",
    "deviant", "deviants",
    "pervert", "perverts", "perverted", "perversion",
    "predator", "predators", "predatory",
    "abuser", "abusers", "abusive", "abuse",
    "molester", "molesters",
    "rapist", "rapists",
    "pedophile", "pedophiles", "pedo", "paedo",

    # === NEGATIVE EMOTIONS ===
    "angry", "angrier", "angriest", "anger", "enraged",
    "furious", "furiously", "fury", "fuming",
    "rage", "raging", "outrage", "outraged", "outrageous",
    "hostile", "hostility",
    "aggressive", "aggressively", "aggression",
    "violent", "violently", "violence",
    "cruel", "cruelly", "cruelty",
    "brutal", "brutally", "brutality",
    "harsh", "harshly", "harshness",
    "mean", "meaner", "meanest", "meanness",
    "spiteful", "spitefully", "spite",
    "vindictive", "vindictively",
    "malicious", "maliciously", "malice",
    "vicious", "viciously", "viciousness",
    "ruthless", "ruthlessly", "ruthlessness",
    "merciless", "mercilessly",
    "heartless", "heartlessly",
    "callous", "callously", "callousness",
    "cold-blooded", "cold hearted",
    "toxic", "toxicity",
    "poison", "poisonous", "venomous",
    "bitter", "bitterly", "bitterness",
    "resentful", "resentment", "resent",
    "jealous", "jealousy", "envious", "envy",
    "greedy", "greed", "greedily",
    "selfish", "selfishly", "selfishness",
    "arrogant", "arrogantly", "arrogance",
    "cocky", "conceited", "egotistical",
    "narcissist", "narcissistic", "narcissism",
    "obnoxious", "obnoxiously",
    "annoying", "annoyingly", "annoyed",
    "irritating", "irritated", "irritable",
    "infuriating", "infuriated",
    "unbearable", "intolerable",
    "insufferable",
    "despise", "despised", "despising",
    "detest", "detested", "detesting",
    "loathe", "loathed", "loathing",
    "abhor", "abhorred",
    "condemn", "condemned", "condemnation",
    "curse", "cursed", "cursing",
    "damn", "damned", "damning",
    "scorn", "scorned", "scornful",
    "mock", "mocked", "mocking", "mockery",
    "ridicule", "ridiculed", "ridiculous",
    "humiliate", "humiliated", "humiliating", "humiliation",
    "degrade", "degraded", "degrading", "degradation",
    "demean", "demeaned", "demeaning",
    "belittle", "belittled", "belittling",
    "disrespect", "disrespected", "disrespectful",
    "insult", "insulted", "insulting", "insults",
    "offend", "offended", "offensive", "offending",
    "provoke", "provoked", "provoking",
    "taunt", "taunted", "taunting", "taunts",
    "bully", "bullied", "bullying", "bullies",
    "harass", "harassed", "harassing", "harassment",
    "intimidate", "intimidated", "intimidating", "intimidation",
    "threaten", "threatened", "threatening", "threats",
    "terrorize", "terrorized", "terrorizing",
    "torment", "tormented", "tormenting",
    "persecute", "persecuted", "persecution",
    "oppress", "oppressed", "oppression", "oppressive",
    "discriminate", "discriminated", "discrimination",
    "prejudice", "prejudiced",
    "stereotype", "stereotyped", "stereotyping",
    "stigmatize", "stigmatized",
    "marginalize", "marginalized",
    "exclude", "excluded", "exclusion",
    "reject", "rejected", "rejection",
    "abandon", "abandoned", "abandonment",
    "betray", "betrayed", "betrayal",
    "deceive", "deceived", "deception", "deceitful",
    "manipulate", "manipulated", "manipulative", "manipulation",
    "exploit", "exploited", "exploitation", "exploitative",
    "fuck", "fucking", "fucked", "fucker", "fucks", "fck", "fuk", "fuq",
    "motherfucker", "motherfucking", "motherfuckers", "mf", "mofo",
    "wtf", "stfu", "lmfao", "Sex","Sexy"

    # === VIOLENCE & THREATS ===
    "kill", "killed", "killer", "killers", "killing", "killings",
    "murder", "murdered", "murderer", "murderers", "murdering",
    "die", "died", "dying", "dies",
    "death", "deaths", "deadly", "lethal",
    "dead", "deadbeat",
    "suicide", "suicidal",
    "stab", "stabbed", "stabbing", "stabs",
    "shoot", "shot", "shooting", "shooter",
    "gun", "guns", "gunshot", "gunfire",
    "bomb", "bombed", "bombing", "bomber",
    "explode", "explosion", "explosive",
    "attack", "attacked", "attacking", "attacker",
    "assault", "assaulted", "assaulting",
    "hit", "hitting", "hits",
    "punch", "punched", "punching", "punches",
    "kick", "kicked", "kicking", "kicks",
    "beat", "beaten", "beating", "beatdown",
    "slap", "slapped", "slapping", "slaps",
    "smack", "smacked", "smacking",
    "choke", "choked", "choking",
    "strangle", "strangled", "strangling",
    "drown", "drowned", "drowning",
    "burn", "burned", "burning", "burns",
    "hang", "hanged", "hanging",
    "torture", "tortured", "torturing",
    "mutilate", "mutilated", "mutilation",
    "dismember", "dismembered",
    "decapitate", "decapitated", "behead", "beheaded",
    "massacre", "massacred", "massacring",
    "slaughter", "slaughtered", "slaughtering",
    "annihilate", "annihilated", "annihilation",
    "exterminate", "exterminated", "extermination",
    "eradicate", "eradicated", "eradication",
    "eliminate", "eliminated", "elimination",
    "destroy", "destroyed", "destroying", "destruction",
    "demolish", "demolished",
    "obliterate", "obliterated",
    "crush", "crushed", "crushing",
    "smash", "smashed", "smashing",
    "wreck", "wrecked", "wrecking",
    "ruin", "ruined", "ruining",
    "devastate", "devastated", "devastating",
    "ravage", "ravaged",
    "harm", "harmed", "harmful", "harming",
    "hurt", "hurting", "hurts",
    "wound", "wounded", "wounding", "wounds",
    "injure", "injured", "injuring", "injury",
    "damage", "damaged", "damaging",
    "suffer", "suffered", "suffering", "suffers",
    "pain", "painful", "painfully",
    "agony", "agonizing", "agonized",
    "bleed", "bleeding", "bled", "blood", "bloody", "bloodbath",
    "gore", "gory",
    "rape", "raped", "raping", "rapist",
    "kidnap", "kidnapped", "kidnapping",
    "rob", "robbed", "robbing", "robbery",
    "steal", "stealing", "stolen",
    "arson", "arsonist",

    # === COMMANDS & DISMISSALS ===
    "shut", "shutup", "stfu",
    "gtfo", "gtfoh",
    "kys", "kms",
    "die", "go die",
    "scram", "begone",

    # === INTERNET SLANG HATE ===
    "noob", "noobs", "n00b",
    "scrub", "scrubs",
    "bot", "bots",
    "cancer", "cancerous", "aids",
    "autistic",
    "triggered", "snowflake",
    "salty", "butthurt",
    "cringe", "cringy", "cringey",
    "simp", "sus", "cap",
    "ratio", "ratioed",
    "cope", "copium", "seethe", "mald",
    "touch grass", "no life",
    "delusional", "delusion",
}

POSITIVE_WORDS = {
    "hello", "hi", "hey", "howdy", "greetings", "welcome",
    "good", "great", "awesome", "amazing", "wonderful", "fantastic",
    "excellent", "brilliant", "superb", "outstanding", "perfect",
    "beautiful", "gorgeous", "pretty", "cute", "adorable", "lovely",
    "love", "loved", "loving", "adore", "cherish", "care",
    "happy", "happiness", "joy", "joyful", "glad", "cheerful",
    "delighted", "pleased", "thrilled", "excited", "ecstatic",
    "thank", "thanks", "thankful", "grateful", "appreciate",
    "please", "kindly", "sorry",
    "friend", "friends", "buddy", "pal", "mate", "bro",
    "nice", "kind", "gentle", "sweet", "warm", "tender",
    "help", "helping", "helpful", "support", "supportive",
    "inspire", "inspiring", "motivate", "encourage",
    "brave", "courageous", "strong", "strength",
    "smart", "clever", "intelligent", "wise", "brilliant",
    "creative", "talented", "gifted", "skilled",
    "honest", "loyal", "faithful", "trustworthy", "reliable",
    "generous", "compassionate", "empathetic", "thoughtful",
    "polite", "respectful", "humble", "graceful", "patient",
    "peaceful", "calm", "gentle", "quiet", "serene",
    "fun", "funny", "humor", "laugh", "smile", "giggle",
    "enjoy", "enjoying", "enjoyed", "pleasure", "delight",
    "celebrate", "congratulate", "congrats", "congratulations",
    "proud", "pride", "achievement", "success", "accomplish",
    "win", "winner", "champion", "best", "top",
    "family", "home", "together", "community", "team",
    "safe", "secure", "comfort", "comfortable",
    "hope", "hopeful", "optimistic", "positive", "bright",
    "dream", "dreams", "wish", "wishes",
    "paradise", "heaven", "blessing", "blessed",
    "sunshine", "rainbow", "flower", "garden", "nature",
    "music", "dance", "sing", "art", "beauty",
    "vacation", "holiday", "relax", "rest", "peace",
    "birthday", "wedding", "anniversary", "party", "festival",
    "hugs", "kisses", "xoxo",
    "lol", "haha", "lmao", "rofl",
    "yay", "woohoo", "hooray", "hurray",
    "cool", "dope", "sick", "lit", "fire", "goat",
    "queen", "king", "legend", "iconic", "goals",
}

THREAT_PHRASES = [
    "kill you", "kill u", "kill ya", "kill em", "kill them", "kill him", "kill her",
    "hit you", "hit u", "hit ya",
    "kick you", "kick u", "kick ya",
    "beat you", "beat u", "beat ya", "beat up",
    "punch you", "punch u", "punch ya",
    "hurt you", "hurt u", "hurt ya",
    "fight you", "fight u", "fight ya",
    "slap you", "slap u", "slap ya",
    "shoot you", "shoot u", "shoot ya",
    "stab you", "stab u", "stab ya",
    "destroy you", "destroy u",
    "smack you", "smack u",
    "choke you", "choke u",
    "strangle you", "strangle u",
    "burn you", "burn u",
    "end you", "end u",
    "get you", "come for you",
    "will hit", "will kill", "will kick", "will punch",
    "will beat", "will hurt", "will stab", "will shoot",
    "will slap", "will smack", "will choke", "will strangle",
    "gonna hit", "gonna kill", "gonna kick", "gonna beat",
    "gonna hurt", "gonna punch", "gonna slap", "gonna smack",
    "wanna fight", "want to fight", "want to kill",
    "want to hit", "want to hurt", "want to punch",
    "go die", "go kill yourself", "drop dead",
    "kill yourself", "kys", "neck yourself", "end yourself",
    "shut up", "shut the fuck", "shut your mouth",
    "go to hell", "burn in hell", "rot in hell",
    "fuck you", "fuck u", "fuck off", "fuck outta",
    "piss off", "bugger off", "sod off",
    "piece of shit", "son of a bitch", "waste of space",
    "waste of life", "waste of oxygen", "waste of air",
    "nobody likes you", "everyone hates you",
    "you deserve to", "hope you die", "wish you were dead",
    "i hope you", "you should die",
    "get lost", "get out", "go away",
    "don't belong", "go back to",
    "not welcome", "not wanted",
    "sick of you", "tired of you", "fed up with you",
    "can't stand you", "hate you", "hate u",
    "you disgust me", "you make me sick",
    "you're nothing", "you are nothing",
    "you're worthless", "you are worthless",
    "you're pathetic", "you are pathetic",
    "you're disgusting", "you are disgusting",
    "you're trash", "you are trash",
    "you're garbage", "you are garbage",
    "you're a waste", "you are a waste",
    "you suck", "u suck", "you stink",
]


def detect_threat(text):
    lower = text.lower().strip()
    words = set(re.findall(r'[a-z]+', lower))
    pos_matches = words & POSITIVE_WORDS
    if len(pos_matches) >= 1 and len(words & NEGATIVE_WORDS) == 0:
        return False, 0.0
    profanity_roots = ["fuck", "shit", "bitch", "cunt", "dick", "ass", "damn", "whore", "slut", "cock", "nigga", "nigger", "fag", "sex", "porn", "nude", "naked", "boob", "penis", "vagina"]
    for root in profanity_roots:
        if root in lower:
            return True, 0.90

    # Check threat phrases
    for phrase in THREAT_PHRASES:
        if phrase in lower:
            return True, 0.92

    # Check negative vs positive words
    neg_matches = words & NEGATIVE_WORDS
    pos_matches = words & POSITIVE_WORDS

    # Strong negative with no positive context
    if len(neg_matches) >= 2 and len(pos_matches) == 0:
        return True, 0.85

    # Single strong negative in short message
    if len(neg_matches) >= 1 and len(pos_matches) == 0 and len(words) <= 3:
        return True, 0.78

    return False, 0.0


# ============================================
# LOAD TRAINED MODEL
# ============================================
logger.info("Loading artifacts...")
with open(f"{MODEL_DIR}/best_model.json") as f:
    best_info = json.load(f)
best_name = best_info["best_model"]

with open(f"{MODEL_DIR}/{best_name}.pkl", "rb") as f:
    model = pickle.load(f)
with open(f"{PROCESSED}/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open(f"{PROCESSED}/meta_scaler.pkl", "rb") as f:
    meta_scaler = pickle.load(f)
with open(f"{MODEL_DIR}/all_metrics.json") as f:
    all_metrics = json.load(f)

logger.info(f"Loaded best model: {best_name}")
logger.info(f"Negative words: {len(NEGATIVE_WORDS)} | Positive words: {len(POSITIVE_WORDS)} | Threat phrases: {len(THREAT_PHRASES)}")

# ============================================
# TEXT PROCESSING
# ============================================
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him",
    "his", "himself", "she", "her", "hers", "herself", "it", "its",
    "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against",
    "between", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off",
    "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
SPECIAL_RE = re.compile(r"[^a-zA-Z\s]")
MULTI_SPACE = re.compile(r"\s+")
SUFFIX_RULES = [
    ("ational", "ate"), ("tional", "tion"), ("ization", "ize"),
    ("ation", "ate"), ("fulness", "ful"), ("ousness", "ous"),
    ("iveness", "ive"), ("ness", ""), ("ment", ""), ("ing", ""),
    ("ed", ""), ("ly", ""), ("ers", ""), ("er", ""), ("es", ""),
    ("s", ""),
]


def simple_stem(w):
    if len(w) <= 3:
        return w
    for sfx, rep in SUFFIX_RULES:
        if w.endswith(sfx) and len(w) - len(sfx) + len(rep) >= 3:
            return w[:-len(sfx)] + rep
    return w


def predict_single(text):
    from scipy.sparse import hstack, csr_matrix
    t0 = time.time()
    original = text

    # STEP 1: Threat detection
    is_threat, threat_conf = detect_threat(text)

    # STEP 2: ML model
    text_clean = str(text).lower()
    text_clean = URL_RE.sub("", text_clean)
    text_clean = MENTION_RE.sub("", text_clean)
    text_clean = HASHTAG_RE.sub(r"\1", text_clean)

    meta = {
        "char_count": len(text_clean),
        "word_count": len(text_clean.split()),
        "avg_word_len": len(text_clean) / (len(text_clean.split()) + 1),
        "uppercase_ratio": sum(c.isupper() for c in original) / (len(original) + 1),
        "exclamation_count": original.count("!"),
        "question_count": original.count("?"),
    }

    text_clean = SPECIAL_RE.sub(" ", text_clean)
    text_clean = MULTI_SPACE.sub(" ", text_clean).strip()
    tokens = [simple_stem(w) for w in text_clean.split()
              if w not in STOPWORDS and len(w) > 1]
    clean = " ".join(tokens)
    meta["unique_word_ratio"] = len(set(clean.split())) / (len(clean.split()) + 1) if clean else 0

    X_tfidf = tfidf.transform([clean])
    meta_arr = np.array([[
        meta["char_count"], meta["word_count"], meta["avg_word_len"],
        meta["uppercase_ratio"], meta["exclamation_count"],
        meta["question_count"], meta["unique_word_ratio"]
    ]])
    X = hstack([X_tfidf, csr_matrix(meta_scaler.transform(meta_arr))])
    ml_prob = float(model.predict_proba(X)[0][1])

    # STEP 3: Combine
    lower = original.lower()
    words = set(re.findall(r'[a-z]+', lower))
    pos_found = words & POSITIVE_WORDS
    neg_found = words & NEGATIVE_WORDS

    if is_threat:
        final_prob = max(ml_prob, threat_conf)
    elif len(pos_found) >= 1 and len(neg_found) == 0:
        # Positive message with no negative words — reduce ML score
        final_prob = ml_prob * 0.4
    elif len(neg_found) == 0 and ml_prob < 0.85:
        # No negative words and ML not very sure — reduce score
        final_prob = ml_prob * 0.6
    else:
        final_prob = ml_prob

    label = 1 if final_prob >= THRESHOLD else 0
    latency = round((time.time() - t0) * 1000, 2)

    return {
        "text": original,
        "cleaned_text": clean,
        "prediction": "hate" if label else "non-hate",
        "label": label,
        "confidence": round(final_prob, 4),
        "latency_ms": latency,
        "model": best_name,
        "threat_detected": is_threat,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ============================================
# STATS
# ============================================
request_log = deque(maxlen=100_000)
stats = {"total": 0, "hate": 0, "non_hate": 0, "avg_latency_ms": 0,
         "latencies": deque(maxlen=1000), "hourly": {},
         "start": datetime.now(timezone.utc).isoformat()}


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "Empty text"}), 400
    result = predict_single(text)
    stats["total"] += 1
    stats["hate"] += result["label"]
    stats["non_hate"] += 1 - result["label"]
    stats["latencies"].append(result["latency_ms"])
    stats["avg_latency_ms"] = round(np.mean(stats["latencies"]), 2)
    request_log.append(result)
    return jsonify(result)


@app.route("/api/predict/batch", methods=["POST", "OPTIONS"])
def predict_batch():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    texts = request.get_json(force=True).get("texts", [])
    if not texts:
        return jsonify({"error": "No texts"}), 400
    results = [predict_single(t) for t in texts[:100]]
    return jsonify({"count": len(results), "results": results})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model": best_name})


@app.route("/api/metrics", methods=["GET"])
def metrics():
    return jsonify(all_metrics)


@app.route("/api/stats", methods=["GET"])
def get_stats():
    return jsonify({
        "total_requests": stats["total"],
        "total_hate": stats["hate"],
        "total_non_hate": stats["non_hate"],
        "avg_latency_ms": stats["avg_latency_ms"],
        "recent_predictions": list(request_log)[-50:],
    })


if __name__ == "__main__":
    logger.info(f"Starting on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
