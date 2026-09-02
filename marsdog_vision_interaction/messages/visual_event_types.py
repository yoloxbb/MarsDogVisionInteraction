"""Visual event names and model-output mapping."""

EVT_VISION_MASTER_HAPPY = "EVT_VISION_MASTER_HAPPY"
EVT_VISION_MASTER_SAD = "EVT_VISION_MASTER_SAD"
EVT_VISION_MASTER_NEUTRAL = "EVT_VISION_MASTER_NEUTRAL"
EVT_VISION_MASTER = "EVT_VISION_MASTER"
EVT_VISION_STRANGER = "EVT_VISION_STRANGER"
EVT_VISION_FOOD = "EVT_VISION_FOOD"
EVT_VISION_TOY = "EVT_VISION_TOY"
EVT_VISION_FALL = "EVT_VISION_FALL"
EVT_VISION_STOP_GESTURE = "EVT_VISION_STOP_GESTURE"
EVT_VISION_ANIMAL_CALM = "EVT_VISION_ANIMAL_CALM"
EVT_VISION_ANIMAL_GREET = "EVT_VISION_ANIMAL_GREET"
EVT_VISION_ANIMAL_PLAY = "EVT_VISION_ANIMAL_PLAY"
EVT_VISION_ANIMAL_BOUNDARY = "EVT_VISION_ANIMAL_BOUNDARY"

_SPECIAL_ACTION_EVENTS = {
    "fallen_down": EVT_VISION_FALL,
    "stop_gesture": EVT_VISION_STOP_GESTURE,
}

_ACTION_EMOTION = {
    "arm_raise_wave": "happy",
    "jump": "happy",
    "lean_forward_arms_open": "happy",
    "nodding": "happy",
    "clapping": "happy",
    "thumbs_up": "happy",
    "hands_on_hips": "sad",
    "rapid_wave_slap": "sad",
    "finger_pointing": "sad",
    "stomping": "sad",
    "arms_crossed": "sad",
    "head_down_slumped": "sad",
    "hands_covering_face": "sad",
    "body_curled_up": "sad",
    "hunched_back": "sad",
    "neutral_stand_sit": "neutral",
}

_EMOTION_EVENTS = {
    "happy": EVT_VISION_MASTER_HAPPY,
    "sad": EVT_VISION_MASTER_SAD,
    "neutral": EVT_VISION_MASTER_NEUTRAL,
}


def pose_action_to_vision_event(action: str, identity_confirmed: bool) -> str:
    if not action or not identity_confirmed:
        return ""
    if action in _SPECIAL_ACTION_EVENTS:
        return _SPECIAL_ACTION_EVENTS[action]
    return _EMOTION_EVENTS.get(_ACTION_EMOTION.get(action, ""), "")


def face_identity_to_vision_event(identity: str) -> str:
    return (
        EVT_VISION_MASTER
        if identity and identity != "unknown"
        else EVT_VISION_STRANGER
    )


def object_to_vision_event(label: str) -> str:
    if label in {"dog bowl", "dog food can", "dog treat bag"}:
        return EVT_VISION_FOOD
    if label in {"dog toy ball", "dog frisbee toy", "dog tug ring toy"}:
        return EVT_VISION_TOY
    return ""
