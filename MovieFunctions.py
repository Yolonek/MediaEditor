from pydub import AudioSegment
from pathlib import Path


def turn_volume_up(audio_file: AudioSegment, volume_change: int):
    return audio_file + volume_change


def cut_audio_file(audio_file: AudioSegment, start: int, end: int):
    return audio_file[start:end]


def cut_and_change_volume(audio_file_path: Path,
                          audio_file_target: Path,
                          audio_file_name: str,
                          fragment_start: tuple[int, int, int],
                          duration: int,
                          volume_change: int):
    audio = AudioSegment.from_file(str(audio_file_path))
    audio = turn_volume_up(audio, volume_change)
    time_start = (fragment_start[0] * 3600 + fragment_start[1] * 60 + fragment_start[2]) * 1000
    time_end = time_start + duration * 1000
    audio = cut_audio_file(audio, time_start, time_end)
    audio.export(str(audio_file_target / Path(audio_file_name)), format='mp3')


if __name__ == '__main__':
    audio_file = Path(r'C:\Users\thejg\Desktop\Zemdlau\nagrania\Zarządzanie czasem i kutasem1-1.mp3')
    audio_name = 'ZemdlauImprov.mp3'
    audio_target = Path('./Audio')
    cut_and_change_volume(audio_file, audio_target, audio_name, (0, 34, 53), 44, 5)
