import os
import json
import glob
from src.ollama_client import OllamaClient

# --- Configuration ---
JOURNALS_DIR = "journals"
PROFILE_OUTPUT_PATH = "user_profile.json"

def load_all_journal_entries(journals_path):
    """Loads and concatenates all text files from the specified path."""
    print(f"\nLooking for journal files in: {os.path.abspath(journals_path)}")
    all_entries_content = []
    
    # Accept any text file format
    text_files = glob.glob(os.path.join(journals_path, "*.*"))
    text_files = [f for f in text_files if os.path.isfile(f) and not f.endswith(('.py', '.json', '.db'))]
    
    if not text_files:
        print(f"❌ No journal files found in {journals_path}.")
        return None

    print(f"✅ Found {len(text_files)} journal files:")
    sorted_files = sorted(text_files)
    for file_path in sorted_files:
        print(f"  - {os.path.basename(file_path)}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Add file name as a simple separator
                all_entries_content.append(f"--- Entry from {os.path.basename(file_path)} ---\n{content}")
        except Exception as e:
            print(f"❌ Error reading file {file_path}: {e}")
    
    return "\n\n".join(all_entries_content)

def generate_json_profile(journal_content):
    """Uses an LLM to generate a user profile in JSON format based on journal content."""
    print("\nInitializing Ollama client...")
    ollama = OllamaClient()

    prompt = f"""
Analyze the following collection of journal entries and extract key information about the user.
Your response MUST be a single, valid JSON object and nothing else. Do not include any explanatory text before or after the JSON.
The JSON object should conform to the following structure:

{{
  "basic_info": {{
    "name": "User (Journal Author)", // Or infer if possible, otherwise use this default
    "age": "integer | null", // Attempt to infer from entries
    "primary_location_context": "string | null" // e.g., "Lives in SF, sometimes visits parents in SR"
  }},
  "key_goals_and_aspirations": [
    // List of strings, e.g., "Find a romantic partner", "Rehab ankle injury"
  ],
  "current_challenges_and_struggles": [
    // List of strings, e.g., "Healing persistent ankle injury", "Ambivalence about full-time job"
  ],
  "core_values_and_beliefs": [
    // List of strings, e.g., "Importance of physical health", "Value of self-improvement"
  ],
  "recurring_patterns_and_behaviors": [
    // List of strings, e.g., "Tendency towards perfectionism", "Avoidance of job searching"
  ],
  "significant_relationships_mentioned": [
    // List of objects, e.g., {{"name": "Tuhin", "context": "Potential collaboration on 'Mimica'"}}
  ],
  "key_interests": [
    // List of strings, e.g., "AI for personal development", "BJJ", "Dance", "Yoga"
  ],
  "self_reflections_on_emotions": [
    // List of strings, e.g., "Experiences emotional rollercoaster with dating apps", "Feels anxious about productivity"
  ]
}}

Here are the journal entries:
---
{journal_content}
---

Now, generate the JSON profile based on these entries. Remember, ONLY the JSON object.
"""

    print("Sending request to LLM for profile generation. This may take a moment...")
    try:
        response_text = ollama.generate(prompt)
        print("✅ Received response from LLM")
        
        # Attempt to parse the JSON from the response
        json_response_cleaned = response_text.strip()
        if json_response_cleaned.startswith("```json"):
            json_response_cleaned = json_response_cleaned[7:]
        if json_response_cleaned.endswith("```"):
            json_response_cleaned = json_response_cleaned[:-3]
        
        profile_json = json.loads(json_response_cleaned.strip())
        print("✅ Successfully parsed JSON response")
        return profile_json
    except json.JSONDecodeError as e:
        print(f"❌ Error decoding JSON from LLM response: {e}")
        print("LLM Response was:")
        print(response_text)
        return None
    except Exception as e:
        print(f"❌ An error occurred during LLM generation: {e}")
        return None

if __name__ == "__main__":
    print("\n=== Starting User Profile Generation ===")
    print(f"Current working directory: {os.getcwd()}")
    
    # 1. Load all journal entries
    print("\nStep 1: Loading journal entries...")
    journal_data = load_all_journal_entries(JOURNALS_DIR)
    
    if journal_data:
        # 2. Generate JSON profile using LLM
        print("\nStep 2: Generating profile from journal content...")
        profile = generate_json_profile(journal_data)
        
        if profile:
            # 3. Save the profile to a JSON file
            print("\nStep 3: Saving profile to file...")
            try:
                output_path = os.path.abspath(PROFILE_OUTPUT_PATH)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(profile, f, indent=2, ensure_ascii=False)
                print(f"✅ Successfully saved user profile to: {output_path}")
                print(f"   File size: {os.path.getsize(output_path)} bytes")
            except Exception as e:
                print(f"❌ Error saving profile JSON to file: {e}")
        else:
            print("❌ Failed to generate user profile.")
    else:
        print("❌ Could not load journal entries. Profile generation aborted.")
    
    print("\n=== Profile Generation Complete ===")