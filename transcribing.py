import tempfile

import numpy as np
from pydub import AudioSegment

import json
import os.path

import whisper
import torch

from subprocess import DEVNULL, STDOUT, check_call


def run_transcribing_parallel(queue, video_file, language):
    Transcribing().transcribe(
        video_file,
        language=language,
        queue=queue,
    )


class Transcribing:
    def __init__(self, audio_format='wav', model='small', use_cuda=True):
        self.model = model
        self.use_cuda = use_cuda
        self.audio_format = audio_format
        self.result = []

        print(f"Initializing device and loading model (use_cuda={self.use_cuda}, model={self.model})")
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.model = whisper.load_model(name=self.model, device=self.device)

    def video2mp3(self, file):
        cmd = f"ffmpeg -i {file} -ac 2 -f {self.audio_format} {file}.{self.audio_format} -y".split(" ")

        print(f"Converting video to {self.audio_format}...", end=' ')
        if not os.path.isfile(f"{file}.{self.audio_format}"):
            check_call(cmd, stdout=DEVNULL, stderr=STDOUT)
        print("done")

        return f"{file}.{self.audio_format}"

    def dump_last(self, file):
        if self.result is None:
            print("No data found")
            return False

        json.dump({e: i['segments'] for e, i in enumerate(self.result)}, open(file, "w"))
        return True

    def transcribe(self, video_file, language='en', queue=None):
        def handler(seg):
            if queue is not None:
                queue.put(seg)

        temp_audio = self.video2mp3(video_file)
        song = AudioSegment.from_file(temp_audio)
        offset = 0
        offset_add = 10 * 60 * 1000

        print("Audio loaded")

        while len(song) > offset:
            out_mp3 = tempfile.NamedTemporaryFile(delete=False)
            song[offset:offset_add].export(out_mp3.name, format="mp3")

            self.result.append(self.model.transcribe(
                out_mp3.name,
                task='transcribe',
                verbose=True,
                language=language,
                segment_handler=handler
            ))

            offset += offset_add

            out_mp3.close()
            os.unlink(out_mp3.name)

        print("Transcribing done!")


if __name__ == '__main__':
    video = "mishka.mp4"

    transc = Transcribing()
    transc.transcribe(video)
    transc.dump_last(f"{video}.txt")
