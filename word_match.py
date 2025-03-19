import nltk
from nltk.corpus import process

def get_synonym_match(user_input):
    nltk.download("wordnet")
    synonyms = set()

    # Get all synonyms from WordNet
    for synset in wordnet.synsets(user_input):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name())

    # Check if any synonym matches YOLO objects
    for word in yolo_objects:
        if word in synonyms:
            return word

    return None  # No synonym match found


hard_to_locate_objects = [
    # **Small Personal Items**
    "keys", "house keys", "car keys", "wallet", "credit card", "ID card",
    "phone", "remote control", "glasses", "sunglasses", "hearing aid",
    "earbuds", "headphones", "watch", "smartwatch", "bracelet", "ring", "necklace",

    # **Commonly Misplaced Kitchen Items**
    "spoon", "fork", "knife", "measuring spoon", "measuring cup",
    "can opener", "bottle opener", "corkscrew", "spatula", "tongs",
    "whisk", "peeler", "grater", "small cutting board", "coffee mug",
    "tea bag", "sugar packet", "salt shaker", "pepper shaker", "spice jar",

    # **Living Room Items**
    "TV remote", "lamp switch", "fan remote", "light switch", "thermostat",
    "coaster", "small decorative object", "USB cable", "charging cable",
    "phone charger", "laptop charger", "tablet", "stylus", "pen", "pencil",

    # **Bathroom Items**
    "toothbrush", "toothpaste", "razor", "hairbrush", "comb", "soap",
    "shampoo bottle", "conditioner bottle", "lotion bottle", "medicine bottle",
    "prescription bottle", "pill organizer", "deodorant", "perfume", "nail clipper",

    # **Bedroom Items**
    "alarm clock", "bedside lamp", "nightstand drawer items", "earplugs",
    "eye mask", "TV remote", "reading glasses", "hearing aid case", "USB stick",

    # **Office & Stationery Items**
    "USB drive", "external hard drive", "flash drive", "small notebook",
    "sticky notes", "highlighter", "stapler", "paperclip", "rubber band",
    "scissors", "tape dispenser", "batteries",

    # **Cleaning & Household Supplies**
    "spray bottle", "small duster", "sponges", "scrub brush", "detergent bottle",
    "bleach bottle", "laundry pod", "fabric softener", "air freshener",

    # **Miscellaneous Items**
    "coin", "cash", "ticket", "bus pass", "membership card", "earrings",
    "button", "safety pin", "thread spool", "needle", "shoelaces"
]

# YOLO object classes (from COCO dataset, can be replaced with custom classes)
yolo_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "dog", "cat", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]