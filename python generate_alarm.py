import wave
import struct
import math

# ==========================================
# SETTINGS
# ==========================================

filename = "alarm.wav"

duration = 2.0      # seconds
frequency = 1000    # Hz
sample_rate = 44100
volume = 32767

# ==========================================
# GENERATE WAVE FILE
# ==========================================

wav_file = wave.open(filename, 'w')

wav_file.setparams((1, 2, sample_rate, 0, 'NONE', 'not compressed'))

for i in range(int(duration * sample_rate)):

    value = int(
        volume * math.sin(
            2 * math.pi * frequency * (i / sample_rate)
        )
    )

    data = struct.pack('<h', value)

    wav_file.writeframesraw(data)

wav_file.close()

print("alarm.wav generated successfully")