import os
import json

voice_settings_file = "voice_settings.json"

def load_voice_settings():
    """Načítá nastavení hlasů ze souboru JSON (pokud existuje)."""
    if os.path.exists(voice_settings_file):
        with open(voice_settings_file, "r") as f:
            return json.load(f)
    return {}

def save_voice_settings(settings):
    """Ukládá nastavení hlasů do souboru JSON."""
    with open(voice_settings_file, "w") as f:
        json.dump(settings, f, indent=4)

def change_voice(character, voice_id):
    """Změní hlas zadané postavě a uloží do souboru."""
    settings = load_voice_settings()
    settings[character] = voice_id
    save_voice_settings(settings)

def get_voice(character):
    """Načte hlas zadané postavy; pokud není uložen, vrátí výchozí hlas."""
    settings = load_voice_settings()
    return settings.get(character, "CwhRBWXzGAHq8TQ4Fs17")  # Výchozí hlas