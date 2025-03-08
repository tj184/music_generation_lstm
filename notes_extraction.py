import os
import json
import pretty_midi

def extract_notes_from_midi(midi_file):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    notes = []
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            notes.append([note.start, note.end, note.pitch, note.velocity])
    return notes

def process_music_folder(dataset_folder, output_json):
    dataset = []
    
    for file in os.listdir(dataset_folder):
        if file.endswith('.mid'):
            midi_path = os.path.join(dataset_folder, file)
            try:
                notes = extract_notes_from_midi(midi_path)
                dataset.append({"genre": "soft music", "file": file, "notes": notes})
                print(f'Processed {file} as soft music')
            except Exception as e:
                print(f"Error processing {file}: {e}")

    with open(output_json, "w") as f:
        json.dump(dataset, f)

if __name__ == "__main__":
    dataset_folder = "output_midi"
    output_json = "soft_music_dataset.json"
    
    process_music_folder(dataset_folder, output_json)
    print(f"Dataset saved to {output_json}")
