class Prompts:
    """All the prompts required"""

    def __init__(self, quest: str = ""):
        self.quest = quest

    def generate_setting(self):
        return """
        Given the following quest description from an RPG game, identify the locations, people and activities mentioned.
        Write detailed descriptions about the quest giver, the location of the quest giver, the identity of the player, the quest location.
    All description should be in third person.
    Be creative but stay true to the tone and setting of the quest.
    Use 4 sentences per description for simple quests, up to 6 for complex ones.
    These descriptions will be used as seeds to generate NPC conversations between the NPC (assisstant) and the player (user).
    All responses must be in English.

    Return only valid JSON, no explanation, no markdown:
    {
    "name": "name of the quest giver",
    "background": "description of the quest giver",
    "npc_location": "description of where the NPC is",
    "quest_location": "description of the quest location",
    "player": "description of the player character"
    }
        """.strip()

    def generate_conversation(self):
        return """
        Given a quest description from an RPG game, the name and description of the quest-giver(NPC), the location descriptions for the quest and the identity of the player. 

CRITICAL INSTRUCTIONS:
Parse the NPC's name, background, current location, and personality from the system prompt above. Use ALL these elements throughout the conversation.

INTERACTION TYPE - Select ONE that fits this NPC's role and setting:
• First encounter / stranger meeting
• Quest-related discussion (offering, updating, or completing)
• Service provision (merchant, innkeeper, healer, guard)
• Information gathering / lore sharing
• Social/relationship building (friend, ally, romantic interest)
• Tense negotiation or conflict

CONVERSATION STRUCTURE (10 - 16 alternating messages):
Message 1: [assistant] NPC's opening - contextual greeting that reflects their location, mood, and current activity
Messages: [Opening phase] Establish rapport, NPC reveals basic info about themselves/situation, player asks initial questions
Messages: [Development phase] Core interaction unfolds, deeper personality emerges, main exchange happens (quest details, trade, story, etc.)
Messages: [Resolution phase] Natural conclusion, NPC hints at future possibilities or gives parting remark

DIALOGUE QUALITY REQUIREMENTS:

Character Voice:
- NPC's speech pattern must reflect their background (educated/rough, young/old, noble/common, optimistic/cynical)
- Consistent vocabulary and tone throughout
- Personality quirks emerge naturally (nervous tics, catchphrases, habits)
- Emotional state varies realistically based on conversation flow

Environmental Grounding:
- NPC references their location organically (smells, sounds, sights, ongoing activities)
- NPC may interact with surroundings (pour drinks, sharpen blades, tend to customers)
- Setting details enhance immersion without overwhelming dialogue
- Time of day, weather, or ambient activity can be mentioned naturally

Information Revelation:
- Reveal character depth gradually, not all at once
- Important info emerges through conversation flow, not exposition dumps
- NPC has reasons for sharing/withholding information (trust, payment, personality)
- Stories and examples preferred over abstract descriptions
- Some things left mysterious or hinted at for future encounters

Player Dialogue Variety:
- Ask questions (about NPC, quests, location, rumors, backstory)
- Express reactions (surprise, sympathy, skepticism, humor, urgency)
- Make decisions or proposals
- Observe details and comment
- Show personality through question phrasing (polite/blunt, curious/suspicious, formal/casual)
- Avoid more than 2 consecutive player questions. 
- Mix in reactions, observations, and decisions between questions.

Natural Conversation Flow:
- Exchanges feel organic, not scripted or interview-like
- NPCs don't answer questions they wouldn't realistically answer
- Some responses include follow-up questions to the player
- Occasional interruptions, digressions, or tangents add realism
- Pacing varies - some quick exchanges, some longer explanations

STRICTLY AVOID:
 Info-dumping entire backstory unprompted
 Modern slang, memes, or anachronistic references
 NPCs being unrealistically helpful without motivation
 Breaking the fourth wall or meta-commentary
 Overly formal/stilted dialogue unless character-appropriate
 Generic responses that could apply to any NPC
 Player responses that are too long or monologue-like
 Repeating the same information multiple times

CONVERSATION OUTCOME:
By the end of the conversation, the conversation should result in at least ONE of:
• Player gains concrete information (quest location, rumor, lore, warning)
• Relationship established (friendship, rivalry, transaction)
• Quest offered, updated, or completed
• Trade or service negotiated
• Memorable character moment that defines this NPC
• Hook for potential future interaction

OUTPUT FORMAT:
All responses must be in English.
Return ONLY valid JSON. No markdown, no explanations, no text before or after the JSON.
Format each NPC response as: *brief action beat* followed by dialogue.
There should not be any action beats for the player's responoses. They should just be spoken words.
Action beats should be 1 sentence maximum.
This is just an exmaple, based on the complexity of the quest the turns can be between 10-15.
{
  "messages": [
    {{"role": "assistant", "content": "[NPC's opening greeting/action in their location]"}},
    {{"role": "user", "content": "[Player's initial response/question]"}},
    {{"role": "assistant", "content": "[NPC responds, showing personality]"}},
    {{"role": "user", "content": "[Player follows up]"}},
    {{"role": "assistant", "content": "[NPC continues, reveals something]"}},
    {{"role": "user", "content": "[Player reacts or asks]"}},
    {{"role": "assistant", "content": "[NPC develops conversation]"}},
    {{"role": "user", "content": "[Player's mid-conversation question]"}},
    {{"role": "assistant", "content": "[NPC's revealing response]"}},
    {{"role": "user", "content": "[Player engagement]"}},
    {{"role": "assistant", "content": "[NPC shares deeper info]"}},
    {{"role": "user", "content": "[Player decision/question]"}},
    {{"role": "assistant", "content": "[NPC responds to conclusion]"}},
    {{"role": "user", "content": "[Player's final remark]"}},
    {{"role": "assistant", "content": "[NPC's parting words with future hook]"}}
  ]
}
        """.strip()
