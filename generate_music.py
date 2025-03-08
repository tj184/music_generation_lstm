import numpy as np
import tensorflow as tf
import pretty_midi
from tensorflow.keras.models import load_model

model = load_model("music_gen_model.h5")

def generate_music(start_sequence, max_duration=30.0):
    generated_sequence = list(start_sequence)
    total_time = sum(note[1] for note in generated_sequence)

    while total_time < max_duration:
        input_seq = np.array([generated_sequence[-50:]])
        pitch_probs, duration_velocity = model.predict(input_seq, verbose=0)
        pitch = np.argmax(pitch_probs)
        duration, velocity = duration_velocity[0]

        duration = max(0.1, duration)
        velocity = max(0, min(127, int(velocity)))

        if total_time + duration > max_duration:
            break
        
        generated_sequence.append([pitch, duration, velocity])
        total_time += duration

    return generated_sequence

def notes_to_midi(notes, output_midi="generated_music.mid"):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    start_time = 0
    for note in notes:
        pitch, duration, velocity = int(note[0]), note[1], int(note[2])
        end_time = start_time + duration

        midi_note = pretty_midi.Note(
            velocity=velocity, pitch=pitch, start=start_time, end=end_time
        )
        instrument.notes.append(midi_note)
        start_time = end_time

    midi.instruments.append(instrument)
    midi.write(output_midi)
    print(f"MIDI file saved: {output_midi}")

start_sequence = np.random.randint(30, 80, (50, 3))
generated_notes = generate_music(start_sequence, max_duration=30.0)
notes_to_midi(generated_notes)
print("MIDI music generation complete!")
